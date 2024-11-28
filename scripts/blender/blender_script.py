"""
Copy this file into Blender
"""

import bpy
import bmesh

import numpy    as np

from math       import radians
from mathutils  import Vector 

# =============================================================================
# SETTINGS
# =============================================================================
# Paths =======================================================================
MODEL           = "ltd_v3"
# Enter the path to the project here
PROJECT_PATH    = "E:\Developer\Master-Thesis"
DIFFUSION       = "data\log\diffusion"
# Enter the heightmap file here
FILE            = "infinite_asd_11-25_15-59-24_2.npy"
HEIGHTMAP_PATH  = f"{PROJECT_PATH}\{DIFFUSION}\{MODEL}\heightmaps\{FILE}"
RENDER_PATH     = f"{PROJECT_PATH}\{DIFFUSION}\{MODEL}\\renders\{FILE}"

# Heightmap Params ============================================================
# Scale of points in x and y in meters (1u = 8092m)
HEIGHTMAP_SCALE = (111111 / 2) / 8092
# Scale of a point in z (0u = 0m, 1u = 8092m)
HEIGHT_SCALE    = 1/2 

TERRAIN_TYPE    = "European"

# Lights and Camera ===========================================================
CAMERA_POSITION     = (HEIGHTMAP_SCALE/2, -4.5, 10)
CAMERA_ROTATION     = (radians(37.5), 0, 0)
CAMERA_CLIP_START   = 0.5
CAMERA_CLIP_END     = 100
CAMERA_ASPECT_RATIO = (4, 3)

RENDER_RESOLUTION   = 1500 

SUN_POSITION        = (0, 0, 2)
SUN_ROTATION        = (radians(45), radians(45), radians(-45))
SUN_STRENGTH        = 2.6

SUBDIVS_VIEW        = 2
SUBDIVS_RENDER      = 4

BACKGROUND_COLOUR   = (0.06, 0.37, 1, 1.0)
BACKGROUND_STRENGTH = 0.2

# =============================================================================
# HELPER CLASSES
# =============================================================================
class MaterialLookupTable():
    def __init__(self, biome_entries):
        self.biome_entries = biome_entries

    def __getitem__(self, biome):
        return self.biome_entries[biome]

class BiomeEntry():
    def __init__(self, steepness_entries):
        self.steepness_entries = steepness_entries

    def __getitem__(self, steepness):
        for steepness_entry in self.steepness_entries:
            if steepness_entry.steepness >= steepness:
                return steepness_entry
            
        return self.steepness_entries(len(self.steepness_entries) - 1)

    def create_material(self):
        material            = bpy.data.materials.new(name="Material")

        material.use_nodes  = True
        nodes               = material.node_tree.nodes
        links               = material.node_tree.links

        nodes.clear()
        
        texture_coordinate      = nodes.new(type="ShaderNodeTexCoord")

        previous_shader = self.steepness_entries[0].create_shader(
            texture_coordinate, nodes, links)

        for i in range(1, len(self.steepness_entries)):
            steepness_entry     = self.steepness_entries[i]
            material_shader     = steepness_entry.create_shader(
                texture_coordinate, nodes, links)

            previous_steepness  = self.steepness_entries[i-1].steepness

            # Determine Steepness =============================================
            # Dot Product -----------------------------------------------------
            dot_node            = nodes.new(type = "ShaderNodeVectorMath")
            dot_node.operation  = "DOT_PRODUCT"
            links.new(texture_coordinate.outputs["Normal"], 
                      dot_node.inputs[0])
            dot_node.inputs[1].default_value = UP_VECTOR

            # Arccos ----------------------------------------------------------
            arccos_node             = nodes.new(type = "ShaderNodeMath")
            arccos_node.operation   = "ARCCOSINE"
            links.new(dot_node.outputs["Value"], 
                      arccos_node.inputs["Value"])

            # To Degrees ------------------------------------------------------
            to_deg_node             = nodes.new(type = "ShaderNodeMath")
            to_deg_node.operation   = "DEGREES"
            links.new(arccos_node.outputs["Value"], 
                      to_deg_node.inputs["Value"])

            # Compare to steepness threshold ==================================
            greater_than_node           = nodes.new(type = "ShaderNodeMath")
            greater_than_node.operation = "GREATER_THAN"
            links.new(to_deg_node.outputs["Value"], 
                      greater_than_node.inputs["Value"])
            greater_than_node.inputs[1].default_value = previous_steepness

            mix_shader = nodes.new(type="ShaderNodeMixShader")
            links.new(greater_than_node.outputs["Value"], mix_shader.inputs[0])
            links.new(previous_shader.outputs[0], mix_shader.inputs[1])
            links.new(material_shader.outputs[0], mix_shader.inputs[2])

            previous_shader = mix_shader

        # Shader output -------------------------------------------------------
        output_node = nodes.new(type="ShaderNodeOutputMaterial")

        links.new(previous_shader.outputs[0], 
                  output_node.inputs["Surface"])

        return material
    
class SteepnessEntry():
    def __init__(self, steepness, height_entries):
        self.steepness      = steepness
        self.height_entries = height_entries

    def __getitem__(self, height):
        for height_entry in self.height_entries:
            if height_entry.height >= height:
                return height_entry
            
        return self.height_entries(len(self.height_entries) - 1)
    
    def create_shader(self, texture_coordinate, nodes, links):

        separate_xyz_node   = nodes.new(type="ShaderNodeSeparateXYZ")
        links.new(texture_coordinate.outputs["Object"], 
                  separate_xyz_node.inputs[0])
        
        previous_shader = self.height_entries[0].create_shader(
                texture_coordinate, nodes, links)

        for i in range(1, len(self.height_entries)):
            height_entry    = self.height_entries[i]
            material_shader = height_entry.create_shader(texture_coordinate, 
                                                         nodes, 
                                                         links)


            previous_height = self.height_entries[i-1].height

            greater_than_node           = nodes.new(type = "ShaderNodeMath")
            greater_than_node.operation = "GREATER_THAN"
            links.new(separate_xyz_node.outputs["Z"], 
                      greater_than_node.inputs[0])
            greater_than_node.inputs[1].default_value = previous_height

            mix_shader = nodes.new(type="ShaderNodeMixShader")
            links.new(greater_than_node.outputs["Value"], mix_shader.inputs[0])
            links.new(previous_shader.outputs[0], mix_shader.inputs[1])
            links.new(material_shader.outputs[0], mix_shader.inputs[2])

            previous_shader = mix_shader

        return previous_shader

class HeightEntry():
    def __init__(self, height, material_values):
        self.height             = height
        self.material_values    = material_values
        self.index              = 0

        self.material           = None

    def create_shader(self, texture_coordinate, nodes, links):
        # Just a single material ==============================================
        if not issubclass(type(self.material_values), list):
            shader_node = self.material_values.create_shader(nodes)
 
            return shader_node  

        # Multiple materials so we need to blend them =========================
        # Input Geometry ------------------------------------------------------
        separate_xyz_node   = nodes.new(type="ShaderNodeSeparateXYZ")
        links.new(texture_coordinate.outputs["Object"], 
                  separate_xyz_node.inputs[0])

        # Take first material as first mixed shader ---------------------------
        previous_blended_nodes = self.material_values[0].create_shader(nodes)

        for i in range(1, len(self.material_values)):
            material_value  = self.material_values[i]
            material_shader = material_value.create_shader(nodes)

            start_height    = self.material_values[i-1].height
            end_height      = material_value.height
            range_size      = end_height - start_height

            # Subtract start height from current height -----------------------
            subtract_node           = nodes.new(type = "ShaderNodeMath")
            subtract_node.operation = "SUBTRACT"
            links.new(separate_xyz_node.outputs["Z"], subtract_node.inputs[0])
            subtract_node.inputs[1].default_value = start_height

            # Take the maximum between subtraction and 0 ----------------------
            maximum_node            = nodes.new(type = "ShaderNodeMath")
            maximum_node.operation  = "MAXIMUM"
            links.new(subtract_node.outputs["Value"], maximum_node.inputs[0])
            maximum_node.inputs[1].default_value = 0

            # Divide maximum by the size of the range to get alpha ------------
            divide_node             = nodes.new(type = "ShaderNodeMath")
            divide_node.operation   = "DIVIDE"
            links.new(maximum_node.outputs["Value"], divide_node.inputs[0])
            divide_node.inputs[1].default_value = range_size

            # Create blended shader -------------------------------------------
            mix_shader = nodes.new(type="ShaderNodeMixShader")
            links.new(divide_node.outputs["Value"], mix_shader.inputs[0])
            links.new(previous_blended_nodes.outputs[0], mix_shader.inputs[1])
            links.new(material_shader.outputs[0], mix_shader.inputs[2])

            previous_blended_nodes = mix_shader

        return previous_blended_nodes        

class MaterialValues():
    def __init__(self, colour, roughness, height = 1):
        self.colour     = colour
        self.roughness  = roughness    
        self.height     = height

    
    def create_shader(self, nodes):
        shader_node = nodes.new(type="ShaderNodeBsdfPrincipled")

        shader_node.inputs["Base Color"].default_value  = (
            self.colour)
        shader_node.inputs["Roughness"].default_value   = (
            self.roughness)

        return shader_node 

# =============================================================================
# MATERIAL LOOKUP TABLE
# =============================================================================
MATERIAL_TABLE  = MaterialLookupTable( biome_entries = 
    {
    "European": BiomeEntry(steepness_entries = [ 
        SteepnessEntry(
            steepness       = 14,
            height_entries  = [
                # Lake
                HeightEntry(
                    height          = 0.28,
                    material_values = MaterialValues(
                        colour      = (0.06, 0.16, 0.17, 1),
                        roughness   = 0.1 
                    )
                ), 
                # Grass
                HeightEntry(
                    height          = 0.4,
                    material_values = [
                        # Fresh Grass
                        MaterialValues(
                            height      = 0.3,
                            colour      = (0.03, 0.59, 0.13, 1),
                            roughness   = 0.55 
                        ),
                        # Darker Grass
                        MaterialValues(
                            height      = 0.38,
                            colour      = (0.11, 0.43, 0.15, 1),
                            roughness   = 0.6 
                        )
                    ]
                ), 
                # Snow
                HeightEntry(
                    height          = 1,
                    material_values = MaterialValues(
                        colour      = (1, 1, 1, 1),
                        roughness   = 0.55 
                    )
                ),
            ]
        ),
        SteepnessEntry(
            steepness       = 25,
            height_entries  = [
                # Rocky Moss
                HeightEntry(
                    height          = 0.35,
                    material_values = MaterialValues(
                        colour      = (0.24, 0.32, 0.24, 1),
                        roughness   = 1 
                    )
                ),
                # Gravel?
                HeightEntry(
                    height          = 1,
                    material_values = MaterialValues(
                        colour      = (0.28, 0.28, 0.28, 1),
                        roughness   = 1 
                    )
                )
            ]
        ), 
        SteepnessEntry(
            steepness       = 90,
            height_entries  = [
                # Rock
                HeightEntry(
                    height          = 1,
                    material_values = [
                        # Normal Rock
                        MaterialValues(
                            height      = 0.4,
                            colour      = (0.3, 0.3, 0.3, 1),
                            roughness   = 1 
                        ),
                        # Icy Rock
                        MaterialValues(
                            height      = 0.8,
                            colour      = (0.4, 0.4, 0.4, 1),
                            roughness   = 0.7 
                        )
                    ]
                )
            ]
        )   
    ]),
    "Coast":    None,
    "Desert":   None,
    "Polar":    None,
    "Grey": BiomeEntry(steepness_entries = [ 
        SteepnessEntry(
            steepness       = 90,
            height_entries  = [
                HeightEntry(
                    height          = 1,
                    material_values = [
                        # Fresh Grass
                        MaterialValues(
                            height      = 0.0,
                            colour      = (0, 0, 0, 1),
                            roughness   = 1 
                        ),
                        # Darker Grass
                        MaterialValues(
                            height      = 1,
                            colour      = (1, 1, 1, 1),
                            roughness   = 0.6 
                        )
                    ]
                ), 
            ]
        ),
    ])
    }
)

# =============================================================================
# Constants
# =============================================================================
UP_VECTOR = Vector((0, 0, 1))

# =============================================================================
# SCRIPT
# =============================================================================
# Clear current scene =========================================================
for obj in bpy.context.collection.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

# Load the Heightmap ==========================================================
heightmap   = np.load(HEIGHTMAP_PATH)
heightmap   = (heightmap + 1) * HEIGHT_SCALE

print (f"Min Height: {np.min(heightmap)}")
print (f"Max Height: {np.max(heightmap)}")

# Create mesh from heightmap ==================================================
def generate_heightmap_mesh(name, heightmap):
    heightmap_size_x = heightmap.shape[1]
    heightmap_size_y = heightmap.shape[0]

    # Create meshgrid for easy indexing ---------------------------------------
    x       = np.linspace(0, 1, heightmap_size_x)
    y       = np.linspace(0, 1, heightmap_size_y)
    x, y    = np.meshgrid(x, y)
    
    # Create vertices and faces -----------------------------------------------
    vertices    = []
    faces       = []
    for j in range(heightmap_size_y):
        for i in range(heightmap_size_x):
            vertices.append((x[j, i] * HEIGHTMAP_SCALE, 
                             y[j, i] * HEIGHTMAP_SCALE, 
                             heightmap[j, i]))
            
            # One less face than vertices
            if j < heightmap_size_y - 1 and i < heightmap_size_x - 1: 
                idx = j * heightmap_size_x + i
                faces.append([idx, 
                              idx + 1, 
                              idx + heightmap_size_x + 1, 
                              idx + heightmap_size_x])

    # Create the mesh and add to viewport -------------------------------------
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    return obj

heightmap_mesh = generate_heightmap_mesh("Heightmap", heightmap)

# Add subdivision surface modifier ============================================  
subsurf                 = heightmap_mesh.modifiers.new(name="Subdivision", 
                                                       type="SUBSURF")
subsurf.levels          = SUBDIVS_VIEW
subsurf.render_levels   = SUBDIVS_RENDER

for face in heightmap_mesh.data.polygons:
    face.use_smooth = True

# Add Material ================================================================
def add_materials(heightmap, table):
    mesh = heightmap.data
    
    material = table[TERRAIN_TYPE].create_material()
    mesh.materials.append(material)
        
add_materials(heightmap_mesh, MATERIAL_TABLE)

# Lighting ====================================================================
light = bpy.data.objects.new("Light", 
                             bpy.data.lights.new("Light", type="SUN"))
light.location          = SUN_POSITION
light.rotation_euler    = SUN_ROTATION
light.data.energy       = SUN_STRENGTH
bpy.context.collection.objects.link(light)

# Camera ====================================================================
camera                  = bpy.data.objects.new("Camera", 
                                               bpy.data.cameras.new("Camera"))
camera.location         = CAMERA_POSITION
camera.rotation_euler   = CAMERA_ROTATION

camera.data.clip_start  = CAMERA_CLIP_START
camera.data.clip_end    = CAMERA_CLIP_END

bpy.context.collection.objects.link(camera)
bpy.context.scene.camera                = camera
bpy.context.scene.render.resolution_x   = RENDER_RESOLUTION
bpy.context.scene.render.resolution_y   = int(
    RENDER_RESOLUTION * CAMERA_ASPECT_RATIO[1] / CAMERA_ASPECT_RATIO[0])


# Adjust the background =======================================================
# Maybe make this a different colour instead
background = bpy.context.scene.world.node_tree.nodes["Background"]
background.inputs["Color"].default_value    = BACKGROUND_COLOUR
background.inputs["Strength"].default_value = BACKGROUND_STRENGTH


print ("Finished!")
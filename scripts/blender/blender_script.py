"""
Copy this file into Blender
"""

import bpy
import bmesh

import numpy    as np

from mathutils  import Vector 

# =============================================================================
# HELPER CLASSES
# =============================================================================
class MaterialLookupTable():
    def __init__(self, biome_entries):
        self.biome_entries = biome_entries

        for _, biome in biome_entries.items():
            if biome is not None:
                biome.post_init(0)

    def __getitem__(self, biome):
        return self.biome_entries[biome]

class BiomeEntry():
    def __init__(self, steepness_entries):
        self.steepness_entries = steepness_entries

    def post_init(self, index):
        for steepness_entry in self.steepness_entries:
            index = steepness_entry.post_init(index)
        
        return index

    def __getitem__(self, steepness):
        for steepness_entry in self.steepness_entries:
            if steepness_entry.steepness >= steepness:
                return steepness_entry
            
        return self.steepness_entries(len(self.steepness_entries) - 1)
    
    def get_all_materials(self):
        all_materials = []
        for steepness_entry in self.steepness_entries:
            for height_entry in steepness_entry.height_entries:
                all_materials.append(height_entry.material)

        return all_materials
    
class SteepnessEntry():
    def __init__(self, steepness, height_entries):
        self.steepness      = steepness
        self.height_entries = height_entries

    def post_init(self, index):
        for height_entry in self.height_entries:
            index = height_entry.post_init(index)
        
        return index

    def __getitem__(self, height):
        for height_entry in self.height_entries:
            if height_entry.height >= height:
                return height_entry
            
        return self.height_entries(len(self.height_entries) - 1)

class HeightEntry():
    def __init__(self, height, material_values):
        self.height             = height
        self.material_values    = material_values
        self.index              = 0

        self.material           = None

    def post_init(self, index):
        self.material   = self.__create_material()
        self.index      = index

        return index + 1

    def __create_material(self):
        material            = bpy.data.materials.new(name="Material")

        material.use_nodes  = True
        nodes               = material.node_tree.nodes
        links               = material.node_tree.links

        nodes.clear()

        # Shader output -------------------------------------------------------
        output_node = nodes.new(type="ShaderNodeOutputMaterial")

        # Just a single material ==============================================
        if not issubclass(type(self.material_values), list):
            shader_node = self.material_values.create_shader(nodes)

            links.new(shader_node.outputs["BSDF"], 
                      output_node.inputs["Surface"])     
            return material  


        # Multiple materials so we need to blend them =========================
        
        # Input Geometry ------------------------------------------------------
        tex_coord_node      = nodes.new(type="ShaderNodeTexCoord")
        separate_xyz_node   = nodes.new(type="ShaderNodeSeparateXYZ")
        links.new(tex_coord_node.outputs["Object"], 
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


        links.new(previous_blended_nodes.outputs[0], 
                  output_node.inputs["Surface"])

        return material        

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

# Lights and Camera ===========================================================
CAMERA_POSITION     = (-1, -1, 2)
CAMERA_ROTATION     = (45,  0, -45)
CAMERA_CLIP_START   = 0.5
CAMERA_CLIP_END     = 100

SUN_POSITION        = (1, 1, 3)

SUBDIVS_VIEW        = 1
SUBDIVS_RENDER      = 4

# Material Lookup =============================================================
TERRAIN_TYPE    = "Tropical"
MATERIAL_TABLE  = MaterialLookupTable( biome_entries = 
    {
    "Tropical": BiomeEntry(steepness_entries = [ 
        SteepnessEntry(
            steepness       = 14,
            height_entries  = [
                # Lake
                HeightEntry(
                    height          = 0.28,
                    material_values = MaterialValues(
                        colour      = (0.21, 0.3, 0.4, 1),
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
                            colour      = (0.18, 0.47, 0.18, 1),
                            roughness   = 0.55 
                        ),
                        # Darker Grass
                        MaterialValues(
                            height      = 0.38,
                            colour      = (0.18, 0.3, 0.18, 1),
                            roughness   = 0.6 
                        )
                    ]
                ), 
                # Snow
                HeightEntry(
                    height          = 1,
                    material_values = MaterialValues(
                        colour      = (0.9, 0.9, 0.9, 1),
                        roughness   = 0.4 
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

bpy.context.view_layer.objects.active = heightmap_mesh
bpy.ops.object.modifier_apply(modifier=subsurf.name)

for face in heightmap_mesh.data.polygons:
    face.use_smooth = True

# Add Material ================================================================
def add_materials(heightmap, table):
    mesh = heightmap.data

    for material in table[TERRAIN_TYPE].get_all_materials():
        mesh.materials.append(material)

    for polygon in mesh.polygons:
        heights = [mesh.vertices[vertex].co.z for vertex in polygon.vertices]

        height                  = np.mean(heights)
        steepness               = polygon.normal.angle(UP_VECTOR) * 180 / np.pi 

        polygon.material_index  = (table[TERRAIN_TYPE]
                                        [steepness]
                                        [height].index)
        
add_materials(heightmap_mesh, MATERIAL_TABLE)

# Lighting ====================================================================
light = bpy.data.objects.new("Light", 
                             bpy.data.lights.new("Light", type="SUN"))
light.location = SUN_POSITION
bpy.context.collection.objects.link(light)

# Camera ====================================================================
camera                  = bpy.data.objects.new("Camera", 
                                               bpy.data.cameras.new("Camera"))
camera.location         = CAMERA_POSITION
camera.rotation_euler   = CAMERA_ROTATION

camera.data.clip_start  = CAMERA_CLIP_START
camera.data.clip_end    = CAMERA_CLIP_END

bpy.context.collection.objects.link(camera)
bpy.context.scene.camera = camera

# Adjust the background =======================================================
# Maybe make this a different colour instead
scene = bpy.context.scene
scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.5

print ("Finished!")
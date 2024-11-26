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
        for i, steepness_entry in enumerate(self.steepness_entries):
            if (steepness_entry.steepness >= steepness
                or i + 1 == len(self.steepness_entries)):

                return steepness_entry
            
        return None
    
    def get_all_materials(self):
        all_materials = []
        for steepness_entry in self.steepness_entries:
            for height_entry in steepness_entry.height_entries:
                all_materials.append(height_entry.material_values.material)

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
        for i, height_entry in enumerate(self.height_entries):
            if (height_entry.height >= height
                or i + 1 == len(self.height_entries)):

                return height_entry.material_values
            
        return None

class HeightEntry():
    def __init__(self, height, material_values):
        self.height             = height
        self.material_values    = material_values

    def post_init(self, index):
        index = self.material_values.post_init(index)

        return index

class MaterialValues():
    def __init__(self, colour, roughness):
        self.colour     = colour
        self.roughness  = roughness
        self.material   = None
        self.index      = 0
    
    def post_init(self, index):
        self.material   = self.__create_material()
        self.index      = index

        return index + 1

    def __create_material(self):
        material = bpy.data.materials.new(name="Material")

        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links

        nodes.clear()

        # Shader properties ---------------------------------------------------
        shader_node = nodes.new(type="ShaderNodeBsdfPrincipled")
        shader_node.inputs["Base Color"].default_value  = self.colour
        shader_node.inputs["Roughness"].default_value   = self.roughness

        # Shader output -------------------------------------------------------
        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        links.new(shader_node.outputs["BSDF"], output_node.inputs["Surface"])
        
        return material


# =============================================================================
# SETTINGS
# =============================================================================
MODEL           = "ltd_v3"
# Enter the path to the project here
PROJECT_PATH    = "E:\Developer\Master-Thesis"
DIFFUSION       = "data\log\diffusion"
# Enter the heightmap file here
FILE            = "infinite_asd_11-25_15-59-24_2.npy"
HEIGHTMAP_PATH  = f"{PROJECT_PATH}\{DIFFUSION}\{MODEL}\heightmaps\{FILE}"
RENDER_PATH     = f"{PROJECT_PATH}\{DIFFUSION}\{MODEL}\\renders\{FILE}"

# Scale of points in x and y in meters (1u = 8092m)
HEIGHTMAP_SCALE = (111111 / 2) / 8092
# Scale of a point in z (0u = 0m, 1u = 8092m)
HEIGHT_SCALE    = 1/2 

TERRAIN_TYPE = "Tropical"

MATERIAL_TABLE = MaterialLookupTable( biome_entries = 
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
                # Fresh Grass
                HeightEntry(
                    height          = 0.3,
                    material_values = MaterialValues(
                        colour      = (0.18, 0.47, 0.18, 1),
                        roughness   = 0.55 
                    )
                ), 
                # Darker Grass
                HeightEntry(
                    height          = 0.4,
                    material_values = MaterialValues(
                        colour      = (0.2, 0.4, 0.2, 1),
                        roughness   = 0.6 
                    )
                ), 
                # Snow
                HeightEntry(
                    height          = 1,
                    material_values = MaterialValues(
                        colour      = (0.9, 0.9, 0.9, 1),
                        roughness   = 0.2 
                    )
                ),
            ]
        ),
        SteepnessEntry(
            steepness       = 25,
            height_entries  = [
                # Rocky Moss
                HeightEntry(
                    height          = 0.4,
                    material_values = MaterialValues(
                        colour      = (0.24, 0.32, 0.24, 1),
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
                    height          = 0.4,
                    material_values = MaterialValues(
                        colour      = (0.3, 0.3, 0.3, 1),
                        roughness   = 1 
                    )
                ),
                # Icy Rock
                HeightEntry(
                    height          = 1,
                    material_values = MaterialValues(
                        colour      = (0.4, 0.4, 0.4, 1),
                        roughness   = 0.7 
                    )
                )
            ]
        )   
    ]),
    "Coast":    None,
    "Desert":   None,
    "Polar":    None,
    }
)

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
subsurf.levels          = 2
subsurf.render_levels   = 4

for face in heightmap_mesh.data.polygons:
    face.use_smooth = True

# Add Material ================================================================
def add_materials(heightmap, table):
    mesh    = heightmap.data
    # bm      = bpy.data.meshes.new_from_object(heightmap)

    for material in table[TERRAIN_TYPE].get_all_materials():
        mesh.materials.append(material)

    for polygon in mesh.polygons:
        heights = []
        for vertex in polygon.vertices:
            heights.append(mesh.vertices[vertex].co.z)

        height                  = np.mean(heights)
        steepness               = polygon.normal.angle(UP_VECTOR) * 180 / np.pi 

        polygon.material_index  = table[TERRAIN_TYPE][steepness][height].index
        
add_materials(heightmap_mesh, MATERIAL_TABLE)

# Lighting ====================================================================
light = bpy.data.objects.new("Light", 
                             bpy.data.lights.new("Light", type="SUN"))
light.location = (1, 1, 3)
bpy.context.collection.objects.link(light)

# Camera ====================================================================
camera                  = bpy.data.objects.new("Camera", 
                                               bpy.data.cameras.new("Camera"))
camera.location         = (-1,   -1,  2)
camera.rotation_euler   = (45,  0, -45)

camera.data.clip_start  = 0.1
camera.data.clip_end    = 5

bpy.context.collection.objects.link(camera)
bpy.context.scene.camera = camera

# Adjust the background =======================================================
# Maybe make this a different colour instead
scene = bpy.context.scene
scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.5

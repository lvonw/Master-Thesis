"""
Copy this file into Blender
"""

import bpy
import bmesh

import numpy    as np

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

# Scale of points in x and y in meters (1u = 8092m)
# Each heightmap is 0.5 
HEIGHTMAP_SCALE = (111111 / 2) / 8092
# Scale of a point in z (0u = 0m, 1u = 8092m)
HEIGHT_SCALE    = 1/2 

# =============================================================================
# SCRIPT
# =============================================================================

# Clear current scene =========================================================
for obj in bpy.context.collection.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

# Load the Heightmap ==========================================================
heightmap       = np.load(HEIGHTMAP_PATH) + 1

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

# for face in mesh.data.polygons:
#     face.use_smooth = True

# Add Material ================================================================
# TODO

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

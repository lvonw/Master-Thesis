import numpy as np

class FractalPerlinGenerator():
    """
    Adapted version of the algorithm detailed in 
    https://en.wikipedia.org/wiki/Perlin-noise
    """

    def __init__(self, configuration):
        self.seed                   = configuration["seed"]
        self.lacunarity             = configuration["lacunarity"]
        self.persistance            = configuration["persistance"]
        self.octaves                = configuration["octaves"]

        self.image_side_resolution  = configuration["image_side_resolution"]
        self.chunks_per_image_side  = configuration["chunks_per_image_side"]
        self.cells_per_chunk_side   = configuration["cells_per_chunk_side"]

        self.chunk_side_resolution  = (self.image_side_resolution
                                       // self.chunks_per_image_side)
        
        self.cell_side_resolution   = (self.chunk_side_resolution
                                       // self.cells_per_chunk_side)
        
        self.center_offset = 0 #np.ceil(self.chunks_per_image_side / 2) - 1

    def generate_image(self, center_coordinate=(0, 0)):
        image_chunks = []
        
        # Generate chunks =====================================================
        offset_x = center_coordinate[0] - self.center_offset
        offset_y = center_coordinate[1] - self.center_offset

        for chunk_y in range(self.chunks_per_image_side):
             image_chunks.append([])
             for chunk_x in range(self.chunks_per_image_side):
                chunk_coordinate = (chunk_x + offset_x,
                                    chunk_y + offset_y)
                image_chunks[chunk_y].append(self.generate_chunk(
                    chunk_coordinate))

        # Stitch chunks =======================================================
        rows = []
        for row in image_chunks:
            rows.append(np.hstack(row))
            
        image = np.vstack(rows)
        return image

    def generate_chunk(self, coordinate=(0, 0)):
        current_frequency = self.cells_per_chunk_side
        current_amplitude = 1

        chunk = np.zeros((self.chunk_side_resolution, 
                          self.chunk_side_resolution))

        for _ in range(self.octaves):            
            gradients       = self.__generate_gradients(
                int(np.ceil(current_frequency)),
                coordinate)
            
            octave_chunk = np.empty((self.chunk_side_resolution, 
                                     self.chunk_side_resolution))
            
            cell_side_resolution   = (self.chunk_side_resolution
                                      // current_frequency)
                                
            for y in range(self.chunk_side_resolution):
                for x in range(self.chunk_side_resolution):

                    cell_x = x // cell_side_resolution
                    cell_y = y // cell_side_resolution

                    relative_x = ((x % cell_side_resolution) 
                                  / cell_side_resolution)
                    
                    relative_y = ((y % cell_side_resolution) 
                                  / cell_side_resolution)

                    octave_chunk[y][x] = self.__perlin(gradients,
                                                       cell_x,
                                                       cell_y,
                                                       relative_x,
                                                       relative_y)
            
            chunk += current_amplitude * octave_chunk 
            
            current_frequency *= self.lacunarity
            current_amplitude *= self.persistance

        return chunk    
    
    def __generate_gradients(self, resolution, coordinate):
        grid_side_length = resolution + 1
        angles = np.empty((grid_side_length, grid_side_length))

        for cell_y in range(grid_side_length):
            string = ""
            for cell_x in range(grid_side_length):

                vertex_x = (cell_x / resolution) + coordinate[0]
                vertex_y = (cell_y / resolution) + coordinate[1]

                vertex_seed = (self.seed + abs(hash((vertex_x, vertex_y))))

                generator = np.random.default_rng(vertex_seed)

                angles [cell_y][cell_x] = generator.uniform(0, 2 * np.pi)

        return np.stack((np.cos(angles), np.sin(angles)), axis = -1)

    def __perlin(self, 
                 gradients, 
                 cell_x, 
                 cell_y, 
                 relative_x, 
                 relative_y):

        top_left        = self.__dot_with_gradient(relative_x,
                                                   relative_y,
                                                   cell_x,
                                                   cell_y,
                                                   0,
                                                   0,
                                                   gradients)
        top_right       = self.__dot_with_gradient(relative_x,
                                                   relative_y,
                                                   cell_x,
                                                   cell_y,
                                                   1,
                                                   0,
                                                   gradients)
        
        top = self.__smoothstep(relative_x, top_left, top_right)

        bottom_left     = self.__dot_with_gradient(relative_x,
                                                   relative_y,
                                                   cell_x,
                                                   cell_y,
                                                   0,
                                                   1,
                                                   gradients)
        bottom_right    = self.__dot_with_gradient(relative_x,
                                                   relative_y,
                                                   cell_x,
                                                   cell_y,
                                                   1,
                                                   1,
                                                   gradients)
        
        bottom  = self.__smoothstep(relative_x, bottom_left, bottom_right)

        return self.__smoothstep(relative_y, top, bottom)


    def __dot_with_gradient(self, 
                            x, 
                            y,
                            cell_x,
                            cell_y, 
                            gradient_x, 
                            gradient_y, 
                            gradients):
        
        distance_x = x - gradient_x
        distance_y = y - gradient_y

        gradient = gradients[cell_y + gradient_y][cell_x + gradient_x]

        return distance_x * gradient[0] + distance_y * gradient[1]


    def __smoothstep(self, x, a, b):
        """5th degree polynomial is steady in 2nd derivative"""
        x = np.pow(x, 3) * (10 + x * (x * 6.0 - 15.0))
        return (b - a) * x + a


import torch
from data.data_util import DataVisualizer

import matplotlib.pyplot as plt

def test(config):
    
    perlin = FractalPerlinGenerator(config)
    data = perlin.generate_image((0,0))
    # data = np.log((data / 2) + 1)

    plt.imshow(data, cmap="gray", interpolation="nearest")
    plt.colorbar()
    plt.title("perlin")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

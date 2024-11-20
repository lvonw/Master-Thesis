import torch
import util

import numpy as np

from enum           import Enum

class MaskInterpolation(Enum):
    NONE                = "None"
    LINEAR              = "Linear"
    LEFT_COSINE         = "Left Cosine"
    RIGHT_COSINE        = "Right Cosine"
    LEFT_EXPONENTIAL    = "Left Exponential"
    RIGHT_EXPONENTIAL   = "Right Exponential"

class OverlapArea(Enum):
    TOP             = "Top"
    TOP_LEFT        = "Top Left"
    TOP_RIGHT       = "Top Right"

    LEFT            = "Left"
    RIGHT           = "Right"

    BOTTOM          = "Bottom"
    BOTTOM_LEFT     = "Bottom Left"
    BOTTOM_RIGHT    = "Bottom Right"

class GenerationGrid():
    def __init__(self, image_shape, overlap_size):
        self.grid               = [[None]]
        self.insertion_order    = []
        self.image_shape        = image_shape
        self.overlap_size       = overlap_size

        # Mapping of the grid space origin to the array space coordinate
        self.origin_coordinate = (0, 0)

    def insert(self, element, coordinate):
        array_coordinate = self.__grid_to_array(coordinate)
        
        # Adjust Y shape ======================================================
        if array_coordinate[1] < 0:
            for _ in range(0 - array_coordinate[1]):
                self.origin_coordinate = (self.origin_coordinate[0],
                                          self.origin_coordinate[1] + 1)
                self.grid =  [None] * len(self.grid) + self.grid

        if array_coordinate[1] >= len(self.grid):
            for _ in range(1 + array_coordinate[1] - len(self.grid)):
                self.grid = self.grid + [None] * len(self.grid)

        # Adjust X shape ======================================================
        if array_coordinate[0] < 0:
            deficit = 0 - array_coordinate[0]

            self.origin_coordinate = (self.origin_coordinate[0] + deficit,
                                      self.origin_coordinate[1])
            
            for idx, row in enumerate(self.grid):
                self.grid[idx] = [None] * deficit + row

        if array_coordinate[0] >= len(self.grid[0]):
            deficit = 1 + array_coordinate[0] - len(self.grid)
            for idx, row in enumerate(self.grid):
                self.grid[idx] = row + [None] * deficit

        # Insert ==============================================================
        new_cell = GridCell(len(self.insertion_order), 
                            element.to("cpu").unsqueeze(0), 
                            coordinate)
        self.insertion_order.append(new_cell)
        self.grid[array_coordinate[1]][array_coordinate[0]] = new_cell        


    def get_mask_for_coordinate(self, 
                                coordinate, 
                                alpha,
                                device=util.get_device()):

        array_coordinate = self.__grid_to_array(coordinate)
        
        # Determine the overlapping regions ===================================        
        overlapping_cells   = [[None, None, None],
                               [None, None, None],
                               [None, None, None]]
        overlapping_order   = []
        no_neighbours       = True
        
        for j, y in enumerate(range(array_coordinate[1] - 1, 
                                    array_coordinate[1] + 2)):
            for i, x in enumerate(range(array_coordinate[0] - 1, 
                                        array_coordinate[0] + 2)):

                if (0 <= y < len(self.grid)
                    and 0 <= x < len(self.grid[y])
                    and self.grid[y][x] is not None):
                    
                    overlapping_cells[j][i] = (self.grid[y][x])
                    overlapping_order.append((i, j))

                    no_neighbours = False
        
        if no_neighbours:
            return None, None
        
        overlapping_order = sorted(
            overlapping_order,
            key=lambda x: overlapping_cells[x[1]][x[0]].insertion_index)
        
        # Create alpha mask ===================================================
        # Use this to fix corner interpolations if we use two adjacent sides
        top_left        = 0
        top_right       = 0
        bottom_left     = 0
        bottom_right    = 0
        
        mask            = torch.zeros(self.image_shape, 
                                      dtype=torch.float32, 
                                      device=device)
                
        # Sides ---------------------------------------------------------------
        # top
        if overlapping_cells[0][1] is not None:
            top_left        += 1
            top_right       += 1
            overlapping_area = self.__get_overlapping_area(
                mask, 
                OverlapArea.TOP) 
            overlapping_area[:] = alpha.unsqueeze(-1) 
        # bottom
        if overlapping_cells[2][1] is not None:
            bottom_left     += 1
            bottom_right    += 1
            overlapping_area = self.__get_overlapping_area(
                mask, 
                OverlapArea.BOTTOM) 
            overlapping_area[:] = alpha.flip([0]).unsqueeze(-1) 
        # left
        if overlapping_cells[1][0] is not None:   
            top_left        += 1
            bottom_left     += 1
            overlapping_area = self.__get_overlapping_area(
                mask, 
                OverlapArea.LEFT) 
            overlapping_area[:] = alpha
        # right
        if overlapping_cells[1][2] is not None:
            top_right       += 1
            bottom_right    += 1
            overlapping_area = self.__get_overlapping_area(
                mask, 
                OverlapArea.RIGHT) 
            overlapping_area[:] = alpha.flip([0]) 
        
        # Corners -------------------------------------------------------------
        corner = torch.zeros((self.overlap_size, self.overlap_size), 
                              dtype =torch.float32, 
                              device=device)
        
        for diagonal, alpha_value in enumerate(alpha):
            corner[diagonal, diagonal:] = alpha_value
            corner[diagonal:, diagonal] = alpha_value

        if overlapping_cells[0][0] is not None or top_left == 2:
            overlapping_area = self.__get_overlapping_area(
                mask, 
                OverlapArea.TOP_LEFT) 
            overlapping_area[:] = corner

        if overlapping_cells[0][2] is not None or top_right == 2:
            overlapping_area = self.__get_overlapping_area(
                mask, 
                OverlapArea.TOP_RIGHT) 
            overlapping_area[:] = corner.flip([3])

        if overlapping_cells[2][0] is not None or bottom_left == 2:
            overlapping_area = self.__get_overlapping_area(
                mask, 
                OverlapArea.BOTTOM_LEFT) 
            overlapping_area[:] = corner.flip([2])
        
        if overlapping_cells[2][2] is not None or bottom_right == 2:
            overlapping_area = self.__get_overlapping_area(
                mask, 
                OverlapArea.BOTTOM_RIGHT) 
            overlapping_area[:] = corner.flip([2, 3])
            
        
        # Create masked image =================================================
        # Perhaps perlin background?
        masked_image = torch.zeros(self.image_shape, 
                                   dtype    = torch.float32,
                                   device   = device)

        for overlap_coordinate in overlapping_order:
            image = overlapping_cells[
                overlap_coordinate[1]][overlap_coordinate[0]].image
            
            # Top ------------------------------------------------------------- 
            if overlap_coordinate   == (0, 0):
                overlapping_area    = self.__get_overlapping_area(
                    masked_image, OverlapArea.TOP_LEFT) 
                overlapping_area[:] = self.__get_overlapping_area(
                    image, OverlapArea.BOTTOM_RIGHT)     

            elif overlap_coordinate == (1, 0):
                overlapping_area    = self.__get_overlapping_area(
                    masked_image, OverlapArea.TOP) 
                overlapping_area[:] = self.__get_overlapping_area(
                    image, OverlapArea.BOTTOM)
                
            elif overlap_coordinate == (2, 0):
                overlapping_area    = self.__get_overlapping_area(
                    masked_image, OverlapArea.TOP_RIGHT) 
                overlapping_area[:] = self.__get_overlapping_area(
                    image, OverlapArea.BOTTOM_LEFT)
            
            # Middle ----------------------------------------------------------     
            elif overlap_coordinate == (0, 1):
                overlapping_area    = self.__get_overlapping_area(
                    masked_image, OverlapArea.LEFT)         
                overlapping_area[:] = self.__get_overlapping_area(
                    image, OverlapArea.RIGHT)
                
            elif overlap_coordinate == (2, 1):
                overlapping_area    = self.__get_overlapping_area(
                    masked_image, OverlapArea.RIGHT) 
                overlapping_area[:] = self.__get_overlapping_area(
                    image, OverlapArea.LEFT)
            
            # Bottom ----------------------------------------------------------     
            elif overlap_coordinate == (0, 2):
                overlapping_area    = self.__get_overlapping_area(
                    masked_image, OverlapArea.BOTTOM_LEFT) 
                overlapping_area[:] = self.__get_overlapping_area(
                    image, OverlapArea.TOP_RIGHT)
                
            elif overlap_coordinate == (1, 2):
                overlapping_area    = self.__get_overlapping_area(
                    masked_image, OverlapArea.BOTTOM) 
                overlapping_area[:] = self.__get_overlapping_area(
                    image, OverlapArea.TOP)
                
            elif overlap_coordinate == (2, 2):
                overlapping_area    = self.__get_overlapping_area(
                    masked_image, OverlapArea.BOTTOM_RIGHT) 
                overlapping_area[:] = self.__get_overlapping_area(
                    image, OverlapArea.TOP_LEFT)
                
        return mask, masked_image
    
    def create_final_image(self, generated_image, masked_image, mask):
        if masked_image is None or mask is None: 
            return generated_image
        
        return mask * masked_image + (1 - mask) * generated_image

    def stitch_image(self):
        height  = (len(self.grid) * self.image_shape[-2] 
                   - (len(self.grid) - 1) * self.overlap_size)
        width   = (len(self.grid[0]) * self.image_shape[-1] 
                   - (len(self.grid[0]) - 1) * self.overlap_size)

        full_image = np.full((1, height, width), -1, dtype=np.float32)

        for sub_image in self.insertion_order:
            array_coordinate = self.__grid_to_array(sub_image.coordinate) 
            image_coordinate = self.__array_to_image(array_coordinate)
            
            full_image[
                :, 
                image_coordinate[1]:image_coordinate[1]+self.image_shape[-2], 
                image_coordinate[0]:image_coordinate[0]+self.image_shape[-1]]=(
                    sub_image.image[:,:].numpy()
                )

        return full_image

    def __str__(self):
        string = ""
        for row in self.grid:
            for column in row:
                if column is None:
                    string += "none "
                else:
                    string += "lmnt "
            string += "\n"
        
        return string

    def __grid_to_array(self, coordinate):
        array_x = coordinate[0] + self.origin_coordinate[0]
        array_y = coordinate[1] + self.origin_coordinate[1]

        return (array_x, array_y)

    def __array_to_grid(self, coordinate):
        grid_x = coordinate[0] - self.origin_coordinate[0]
        grid_y = coordinate[1] - self.origin_coordinate[1]

        return (grid_x, grid_y)

    def __array_to_image(self, coordinate):
        image_x = coordinate[0] * (self.image_shape[-2] - self.overlap_size)
        image_y = coordinate[1] * (self.image_shape[-1] - self.overlap_size)

        return (image_x, image_y)
    
    def __get_overlapping_area(self, tensor, area):
        match area:
            case OverlapArea.TOP_LEFT:
                return tensor[:,:, :self.overlap_size,  :self.overlap_size]
            case OverlapArea.TOP:
                return tensor[:,:, :self.overlap_size]
            case OverlapArea.TOP_RIGHT:
                return tensor[:,:, :self.overlap_size,  -self.overlap_size:]
            
            case OverlapArea.LEFT:
                return tensor[:,:, :,                   :self.overlap_size]
            case OverlapArea.RIGHT:
                return tensor[:,:, :,                   -self.overlap_size:]
            
            case OverlapArea.BOTTOM_LEFT:
                return tensor[:,:, -self.overlap_size:, :self.overlap_size]
            case OverlapArea.BOTTOM:
                return tensor[:,:, -self.overlap_size:]
            case OverlapArea.BOTTOM_RIGHT:
                return tensor[:,:, -self.overlap_size:, -self.overlap_size:]

class GridCell():
    def __init__(self, insertion_index, image, coordinate):
        self.insertion_index    = insertion_index
        self.image              = image
        self.coordinate         = coordinate

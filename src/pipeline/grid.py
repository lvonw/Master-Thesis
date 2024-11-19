import constants
import os
import torch
import util

import numpy        as np

from enum           import Enum

OVERLAP_AREA = 128 #64

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
                self.grid = [] + self.grid

        if array_coordinate[1] >= len(self.grid):
            for _ in range(1 + array_coordinate[1] - len(self.grid)):
                self.grid = self.grid + []

        # Adjust X shape ======================================================
        if array_coordinate[0] < 0:
            deficit = 0 - array_coordinate[0]

            self.origin_coordinate = (self.origin_coordinate[0] + deficit,
                                      self.origin_coordinate[1])
            
            for idx, row in enumerate(self.grid()):
                self.grid[idx] = [None] * deficit + row

        if array_coordinate[0] >= len(self.grid[0]):
            deficit = 1 + array_coordinate[0] - len(self.grid)
            for idx, row in enumerate(self.grid()):
                self.grid[idx] = row + [None] * deficit

        # Insert ==============================================================
        new_cell = GridCell(len(self.insertion_order), 
                            element, 
                            array_coordinate)
        self.insertion_order.append(new_cell)
        self.grid[array_coordinate[1]][array_coordinate[0]] = new_cell        


    def get_mask_for_coordinate(self, 
                                coordinate, 
                                alpha,
                                device=util.get_device()):

        array_coordinate = self.__grid_to_array(coordinate)
        
        # Determine the overlapping regions ===================================        
        overlapping_cells = [[None, None, None],
                             [None, None, None],
                             [None, None, None]]
        overlapping_order = []
        
        for j, y in enumerate(range(array_coordinate[1] - 1, 
                                    array_coordinate[1] + 2)):
            for i, x in enumerate(range(array_coordinate[0] - 1, 
                                        array_coordinate[0] + 2)):
                
                if (y in self.grid and x in self.grid[y] 
                    and self.grid[y][x] is not None):
                    
                    overlapping_cells.append(self.grid[y][x])
                    overlapping_order.append()

        overlapping_order = sorted(
            overlapping_order,
            key=lambda x: overlapping_cells[x[1]][x[0]].insertion_index)

        # Create alpha mask ===================================================
        # Use this to fix corner interpolations if we use two adjacent sides
        top_left        = 0
        top_right       = 0
        bottom_left     = 0
        bottom_right    = 0
        
        mask            = torch.zeros(self.shape, 
                                      dtype=torch.float32, 
                                      device=device)
                
        # Sides ---------------------------------------------------------------
        # top
        if overlapping_cells[0][1] is not None:
            top_left        += 1
            top_right       += 1
            self.__get_overlapping_area(
                tensor, OverlapArea.TOP) = alpha.unsqueeze(-1) 
        # bottom
        if overlapping_cells[2][1] is not None:
            bottom_left     += 1
            bottom_right    += 1
            self.__get_overlapping_area(
                tensor, OverlapArea.BOTTOM) = alpha.flip([0]).unsqueeze(-1) 
        # left
        if overlapping_cells[1][0] is not None:   
            top_left        += 1
            bottom_left     += 1
            self.__get_overlapping_area(
                tensor, OverlapArea.LEFT) = alpha
        # right
        if overlapping_cells[1][2] is not None:
            top_right       += 1
            bottom_right    += 1
            self.__get_overlapping_area(
                tensor, OverlapArea.RIGHT) = alpha.flip([0]) 
        
        # Corners -------------------------------------------------------------
        corner = torch.zeros((self.overlap_size, self.overlap_size), 
                              dtype =torch.float32, 
                              device=device)
        
        for diagonal, alpha_value in enumerate(alpha):
            corner[:,:, diagonal, diagonal:] = alpha_value
            corner[:,:, diagonal:, diagonal] = alpha_value

        if overlapping_cells[0][0] is not None or top_left == 2:
            self.__get_overlapping_area(
                tensor, OverlapArea.TOP_LEFT) = corner

        if overlapping_cells[0][2] is not None or top_right == 2:
            self.__get_overlapping_area(
                tensor, OverlapArea.TOP_RIGHT) = corner.flip([3])

        if overlapping_cells[2][0] is not None or bottom_left == 2:
            self.__get_overlapping_area(
                tensor, OverlapArea.BOTTOM_LEFT) = corner.flip([2])
        
        if overlapping_cells[2][2] is not None or bottom_right == 2:
            self.__get_overlapping_area(
                tensor, OverlapArea.BOTTOM_RIGHT) = corner.flip([2, 3])
            
        
        # Create masked image =================================================
        # Perhaps perlin background?
        masked_image = torch.zeros(self.image_shape)

        for overlap_coordinate in overlapping_order():
            image = overlapping_cells[
                overlap_coordinate[1]][overlap_coordinate[0]].image
            
            if overlap_coordinate   == (0, 0):
                self.__get_overlapping_area(
                    masked_image, OverlapArea.TOP_LEFT) = (
                    self.__get_overlapping_area(
                        image, OverlapArea.BOTTOM_RIGHT))
                
            elif overlap_coordinate == (0, 1):
                self.__get_overlapping_area(
                    masked_image, OverlapArea.TOP) = (
                    self.__get_overlapping_area(
                        image, OverlapArea.BOTTOM))
                
            elif overlap_coordinate == (0, 2):
                self.__get_overlapping_area(
                    masked_image, OverlapArea.TOP_RIGHT) = (
                    self.__get_overlapping_area(
                        image, OverlapArea.BOTTOM_LEFT))
                
            elif overlap_coordinate == (1, 0):
                self.__get_overlapping_area(
                    masked_image, OverlapArea.LEFT) = (
                    self.__get_overlapping_area(
                        image, OverlapArea.RIGHT))
                
            elif overlap_coordinate == (1, 2):
                self.__get_overlapping_area(
                    masked_image, OverlapArea.RIGHT) = (
                    self.__get_overlapping_area(
                        image, OverlapArea.LEFT))
                
            elif overlap_coordinate == (2, 0):
                self.__get_overlapping_area(
                    masked_image, OverlapArea.BOTTOM_LEFT) = (
                    self.__get_overlapping_area(
                        image, OverlapArea.TOP_RIGHT))
                
            elif overlap_coordinate == (2, 1):
                self.__get_overlapping_area(
                    masked_image, OverlapArea.BOTTOM) = (
                    self.__get_overlapping_area(
                        image, OverlapArea.TOP))
                
            elif overlap_coordinate == (2, 2):
                self.__get_overlapping_area(
                    masked_image, OverlapArea.BOTTOM_RIGHT) = (
                    self.__get_overlapping_area(
                        image, OverlapArea.TOP_LEFT))
                
        return mask, masked_image
    
    def create_final_image(self, generated_image, masked_image, mask):
        return mask * masked_image + (1 - mask) * generated_image

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
    def __init__(self, insertion_index, image, coordinates):
        self.insertion_index    = insertion_index
        self.image              = image
        self.coordinates        = coordinates

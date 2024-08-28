import constants
import os

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data   import Dataset

from osgeo      import gdal

def cell_to_geo_coordinates(dataset, x, y):
    geo_transform = dataset.GetGeoTransform()

    x_origin        = geo_transform[0]
    pixel_width     = geo_transform[1]
    rotation_x      = geo_transform[2]
    y_origin        = geo_transform[3]
    rotation_y      = geo_transform[4]
    pixel_height    = geo_transform[5]

    longitude   = x_origin + x * pixel_width  + y * rotation_x
    latitude    = y_origin + y * pixel_height + x * rotation_y 

    return longitude, latitude

def geo_coordinates_to_cell(dataset, longitude, latitude):
    geo_transform       = dataset.GetGeoTransform()

    # The maps used are all north oriented so the rotation will always be 0
    # which simplifies the calculation
    if (not geo_transform[2] == 0.0) or (not geo_transform[4] == 0.0):
        return (0, 0)

    x_origin        = geo_transform[0]
    pixel_width     = geo_transform[1]
    y_origin        = geo_transform[3]
    pixel_height    = geo_transform[5]


    x = (longitude - x_origin) / pixel_width
    y = (latitude - y_origin) / pixel_height
    return int(x), int(y)

def show_dataset(dataset):
    dataset_array = dataset.GetRasterBand(1).ReadAsArray()

    plt.figure(figsize=(10, 10))
    plt.imshow(dataset_array)

    x,y = geo_coordinates_to_cell(dataset, 9.993682, 53.551086)
    plt.scatter(x, y, color='red', s=20, zorder=5)

    x,y = geo_coordinates_to_cell(dataset, -118.243683, 34.052235)
    plt.scatter(x, y, color='red', s=20, zorder=5)

    x,y = geo_coordinates_to_cell(dataset, 144.946457, -37.840935)
    plt.scatter(x, y, color='red', s=20, zorder=5)

    x,y = geo_coordinates_to_cell(dataset, -51.72157, 64.18347)
    plt.scatter(x, y, color='red', s=20, zorder=5)

    x,y = geo_coordinates_to_cell(dataset, 18.423300, -33.918861)
    plt.scatter(x, y, color='red', s=20, zorder=5)

    plt.title('Raster Image')
    plt.xlabel('Column (x)')
    plt.ylabel('Row (y)')
    #plt.colorbar(label='Pixel Values')
    plt.show()

class FullDataset(Dataset): 
    def __init__(self, compileList=False):
        if compileList:
            self.__compile_DEM_list()

    def open_DEM_list(self):
        dem_list = []
        with open(constants.DATA_PATH_DEM_LIST, 'r') as file:
            for line in file:
                dem_list.append(line.strip())
        return dem_list    

    def open_DEM(self, name):
        return gdal.Open(constants.DATA_PATH_DEMS + name)

    def open_GLiM(self):
        return gdal.Open(constants.DATA_PATH_GLIM)

    def open_climate(self):
        return gdal.Open(constants.DATA_PATH_CLIMATE)
    
    def open_DSMW(self):
        return gdal.Open(constants.DATA_PATH_DSMW)
    
    def open_GTC(self):
        return gdal.Open(constants.DATA_PATH_GTC)
    
    def __compile_DEM_list(self):
        files = []          
        with os.scandir(constants.DATA_PATH_DEMS) as entries:
            for entry in entries:
                if entry.is_file():
                    files.append(entry.name) 
        with open(constants.DATA_PATH_DEM_LIST, 'w') as file:
            for name in files:
                file.write(f"{name}\n") 



# Testing TODO delete
def main():
    r = FullDataset()

    #gds = r.open_GLiM()
    #gt = gds.GetGeoTransform()
    # p = gds.GetProjection()
    # rc = gds.RasterCount
    # band = gds.GetRasterBand(1)
    # arr = band.ReadAsArray()
    # arr = np.clip(arr, 0, 17)


    #print(gt)

    # show_dataset(r.open_GLiM())
    # show_dataset(r.open_DSMW())
    # show_dataset(r.open_climate())
    show_dataset(r.open_GTC())

    
    
if __name__ == "__main__":
    main()
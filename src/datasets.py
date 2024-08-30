import constants
import os
import torch

import matplotlib.pyplot    as plt
import numpy                as np

from configuration          import Section
from data_access            import DataAccessor
from mpl_toolkits.mplot3d   import Axes3D
from osgeo                  import gdal
from torchvision            import transforms
from torch.utils.data       import Dataset


class GeoUtil():
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
    
def show_dataset_2D(dataset):
    dataset_array = dataset.GetRasterBand(1).ReadAsArray()

    plt.figure(figsize=(10, 10))
    plt.imshow(dataset_array)

    x,y = GeoUtil.geo_coordinates_to_cell(dataset, 9.993682, 53.551086)

    plt.scatter(0, 
                0, color='blue', s=20, zorder=5)
    plt.scatter(dataset_array.shape[0], 
                dataset_array.shape[1], color='red', s=20, zorder=5)

    # x,y = geo_coordinates_to_cell(dataset, -118.243683, 34.052235)
    # plt.scatter(x, y, color='red', s=20, zorder=5)

    # x,y = geo_coordinates_to_cell(dataset, 144.946457, -37.840935)
    # plt.scatter(x, y, color='red', s=20, zorder=5)

    # x,y = geo_coordinates_to_cell(dataset, -51.72157, 64.18347)
    # plt.scatter(x, y, color='red', s=20, zorder=5)

    # x,y = geo_coordinates_to_cell(dataset, 18.423300, -33.918861)
    # plt.scatter(x, y, color='red', s=20, zorder=5)

    plt.title('Raster Image')
    plt.xlabel('Column (x)')
    plt.ylabel('Row (y)')
    #plt.colorbar(label='Pixel Values')
    plt.show()

def show_dataset_3D(dataset):
    dataset_array = dataset.GetRasterBand(1).ReadAsArray()

    x = np.arange(dataset_array.shape[1])
    y = np.arange(dataset_array.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, dataset_array, cmap='viridis')

    plt.title('Raster Image')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #plt.colorbar(label='Pixel Values')
    ax.set_zlim(-1000, +1000)
    plt.show()

class TerrainDataset(Dataset): 
    def __init__(self, DEM_list, data_cache, size, transform=None):
        self.transform  = transform
        self.DEM_list   = DEM_list
        self.data_cache = data_cache

    def __len__(self):
        return len(self.DEM_list)
    
    def __getitem__(self, index):
        DEM_dataset = DataAccessor.open_DEM(self.DEM_list[index])

        if DEM_dataset.RasterCount == 0:
            return None, None
        
        DEM_array = DEM_dataset.GetRasterBand(1).ReadAsArray()

        channels = [DEM_array]

        if self.data_cache: 
            top_left = GeoUtil.cell_to_geo_coordinates(
                DEM_dataset, 0, 0)
            bot_right = GeoUtil.cell_to_geo_coordinates(
                DEM_dataset, DEM_array.shape[1], DEM_array.shape[0])
             
            for cache in self.data_cache:
                cache_array = cache.GetRasterBand(1).ReadAsArray()
                channels.append(cache_array[top_left[0]:bot_right[0]+1, 
                                            top_left[1]:bot_right[1]+1])
                
            

        data_entry = torch.tensor(DEM_array, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            data_entry = self.transform(data_entry)

        return data_entry, data_entry
    
class DatasetFactory():

    def create_dataset(data_configuration: Section) -> TerrainDataset:
        DEM_List = DatasetFactory.__open_DEM_list()

        data_cache = []

        if data_configuration["GLiM"]:
            data_cache.append(DataAccessor.open_gdal_dataset(
                constants.DATA_PATH_GLIM))
        
        if data_configuration["climate"]:
            data_cache.append(DataAccessor.open_gdal_dataset(
                constants.DATA_PATH_CLIMATE))
        
        if data_configuration["DSMW"]:
            data_cache.append(DataAccessor.open_gdal_dataset(
                constants.DATA_PATH_DSMW))
        
        if data_configuration["GTC"]:
            data_cache.append(DataAccessor.open_gdal_dataset(
                constants.DATA_PATH_GTC))
            
        transform_list = []

        if data_configuration["RandomCrop"]:
            transform_list.append(transforms.RandomCrop(
                size=data_configuration["Size"]))
            
        if data_configuration["FiveCrop"]:
            transform_list.append(transforms.FiveCrop(
                size=data_configuration["Size"]))
            
        if data_configuration["TenCrop"]:
            transform_list.append(transforms.TenCrop(
                size=data_configuration["Size"]))
            
        if data_configuration["RandomHorizontalFlip"]:
            transform_list.append(transforms.RandomHorizontalFlip(
                p=data_configuration["RandomHorizontalFlip"]))
            
        if data_configuration["RandomVerticalFlip"]:
            transform_list.append(transforms.RandomHorizontalFlip(
                p=data_configuration["RandomVerticalFlip"]))
        
        # This probably needs to be the very first operation
        if data_configuration["RandomRotation"]:
            transform_list.append(transforms.RandomRotation(
                degrees=data_configuration["RandomRotation"]))

        transform = transforms.Compose(transform_list)

        return TerrainDataset(DEM_List, data_cache, transform)
        
    def __compile_DEM_list():
        files = []          
        with os.scandir(constants.DATA_PATH_DEMS) as entries:
            for entry in entries:
                if entry.is_file():
                    files.append(entry.name) 

        with open(constants.DATA_PATH_DEM_LIST, 'w') as file:
            for name in files:
                file.write(f"{name}\n")

        return files
    
    def __open_DEM_list(compile_list=False):
        if compile_list or not os.path.exists(constants.DATA_PATH_DEM_LIST):
            return DatasetFactory.__compile_DEM_list()

        dem_list = []
        with open(constants.DATA_PATH_DEM_LIST, 'r') as file:
            for line in file:
                dem_list.append(line.strip())
        
        return dem_list    



# Testing TODO delete
def main():
    # r = FullDataset()

    # gds = r.open_DEM("N53E009.tif")
    # gt = gds.GetGeoTransform()
    # p = gds.GetProjection()
    # rc = gds.RasterCount
    # band = gds.GetRasterBand(1)
    # arr = band.ReadAsArray()
    # arr = np.clip(arr, 0, 17)


    # # print(gt)

    # show_dataset(r.open_GLiM())
    # # show_dataset(r.open_DSMW())
    # # show_dataset(r.open_climate())
    # # show_dataset(r.open_GTC())

    show_dataset_2D(DataAccessor.open_DEM("N53E009.tif"))
    # show_dataset_3D(r.open_DEM("N53E009.tif"))
    pass

    
    
if __name__ == "__main__":
    main()
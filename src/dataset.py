import constants
import math
import torch

import matplotlib.pyplot    as plt
import numpy                as np

from configuration          import Section
from data_access            import DataAccessor
from mpl_toolkits.mplot3d   import Axes3D
from osgeo                  import gdal
from torchvision            import transforms
from torch.utils.data       import Dataset, random_split


class GeoUtil():
    def cell_to_geo_coordinates(geo_transform, x, y):
        x_origin        = geo_transform[0]
        pixel_width     = geo_transform[1]
        rotation_x      = geo_transform[2]
        y_origin        = geo_transform[3]
        rotation_y      = geo_transform[4]
        pixel_height    = geo_transform[5]

        longitude       = y_origin + y * pixel_height + x * rotation_y 
        latitude        = x_origin + x * pixel_width  + y * rotation_x

        return longitude, latitude

    def geo_coordinates_to_cell(geo_transform, longitude, latitude):

        # The maps used are all north oriented so the rotation will always be 0
        # which simplifies the calculation
        if geo_transform[2] != 0.0 or geo_transform[4] != 0.0:
            return (0, 0)

        x_origin        = geo_transform[0]
        pixel_width     = geo_transform[1]
        y_origin        = geo_transform[3]
        pixel_height    = geo_transform[5]

        x = (latitude  - x_origin) / pixel_width
        y = (longitude - y_origin) / pixel_height
        
        return int(x), int(y)

    def get_normalized_raster_band(raster_band,
                                   nodata_behaviour = constants.NoDataBehaviour.LOCAL_MINIMUM, 
                                   global_min = None, 
                                   global_max = None):
        
        band_array      = raster_band.ReadAsArray()
        nodata_value    = raster_band.GetNoDataValue()
        
        local_min = np.min(band_array)
        local_max = np.max(band_array)

        global_min = local_min if global_min is None else global_min
        global_max = local_max if global_max is None else global_max

        if nodata_value is not None:
            match nodata_behaviour:
                case constants.NoDataBehaviour.LOCAL_MINIMUM:
                    band_array[band_array == nodata_value] = local_min 
                case constants.NoDataBehaviour.GLOBAL_MINIMUM:
                    band_array[band_array == nodata_value] = global_min

        band_array = band_array - global_min
        band_array = band_array.astype(np.float32) 
        band_array = band_array / np.float32(global_max - global_min)

        return band_array
    
    def get_min_max(DEM_list):
        min = np.iinfo(np.int64).max
        max = np.iinfo(np.int64).min

        for dem in DEM_list:
            d = DataAccessor.open_DEM(dem)

            a = d.GetRasterBand(1).ReadAsArray()
            n = d.GetRasterBand(1).GetNoDataValue()

            max_v = np.max(a)

            if n is not None:
                a[a == n] = max_v
            
            min_v = np.min(a)
           
            if min > min_v:
                min = min_v
            if max < max_v:
                max = max_v
        
        return min, max
        
    
def show_dataset_2D(dataset):
    dataset_array = dataset.GetRasterBand(1).ReadAsArray()

    plt.figure(figsize=(10, 10))
    plt.imshow(dataset_array)
    
    plt.title('Raster Image')
    plt.xlabel('Column (x)')
    plt.ylabel('Row (y)')
    plt.colorbar(label='Pixel Values')
    plt.show()

def show_array(array):
    plt.figure(figsize=(10, 10))
    plt.imshow(array)
    
    plt.title('Raster Image')
    plt.xlabel('Column (x)')
    plt.ylabel('Row (y)')
    plt.colorbar(label='Pixel Values')
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
    def __init__(self, 
                 DEM_list, 
                 channel_data_cache,
                 label_data_cache, 
                 transform=None):
        self.transform              = transform
        self.DEM_list               = DEM_list
        self.channel_data_cache     = channel_data_cache
        self.label_data_cache       = label_data_cache

    def __len__(self):
        return len(self.DEM_list)
    
    def __getitem__(self, index):
        DEM_dataset = DataAccessor.open_DEM(self.DEM_list[index])
       
        # band = gds.GetRasterBand(1)
        # p = gds.GetProjection()
        # show_dataset_2D(DEM_dataset)
        label = None

        if DEM_dataset.RasterCount == 0:
            return None, None
        
        DEM_array   = GeoUtil.get_normalized_raster_band(
            DEM_dataset.GetRasterBand(1),
            global_min = constants.DEM_GLOBAL_MIN,
            global_max = constants.DEM_GLOBAL_MAX
        )

        DEM_tensor  = torch.tensor(DEM_array, dtype=torch.float32).unsqueeze(0)
        channels    = [DEM_tensor]

        # Extract all the specified additional data from other maps
        if self.channel_data_cache: 
            # Get the reference coordinates for the respective data frame
            top_left_geo    = GeoUtil.cell_to_geo_coordinates(
                DEM_dataset.GetGeoTransform(), 
                0, 
                0)
            bot_right_geo   = GeoUtil.cell_to_geo_coordinates(
                DEM_dataset.GetGeoTransform(), 
                DEM_array.shape[1], 
                DEM_array.shape[0])

            resize          = transforms.Resize(DEM_array.shape)
            
            for cache in self.channel_data_cache:
                cache_array = cache[1]

                top_left_cell   = GeoUtil.geo_coordinates_to_cell(
                    cache[0], 
                    top_left_geo[0],
                    top_left_geo[1]) 
                
                bot_right_cell  = GeoUtil.geo_coordinates_to_cell(
                    cache[0],
                    bot_right_geo[0],
                    bot_right_geo[1])
                
                # Extract the data frame
                data_frame = cache_array[top_left_cell[1]:bot_right_cell[1]+1, 
                                         top_left_cell[0]:bot_right_cell[0]+1]
        
                data_tensor = torch.from_numpy(data_frame).unsqueeze(0)
                data_tensor = resize(data_tensor)

                channels.append(data_tensor)
      
        data_entry = torch.cat(channels, dim=0)
        print (torch.min(data_entry))
        print (torch.max(data_entry))

        if self.transform:
            data_entry = self.transform(data_entry)

        return data_entry, label
    
class DatasetFactory():
    def create_dataset(data_configuration: Section) -> tuple[TerrainDataset, 
                                                             TerrainDataset]:
        DEM_List = DataAccessor.open_DEM_list()

        channel_cache   = []
        label_cache     = []
        transform_list  = []

        # TODO make it so that the preprocess happens immediately after every load
        # or parallelize it
        # TODO make it so you can configure what becomes a label and what a 
        # channel
        if data_configuration["GLiM"]:
            channel_cache.append(DataAccessor.open_gdal_dataset(
                constants.DATA_PATH_GLIM))
        
        if data_configuration["climate"]:
            channel_cache.append(DataAccessor.open_gdal_dataset(
                constants.DATA_PATH_CLIMATE))
        
        if data_configuration["DSMW"]:
            channel_cache.append(DataAccessor.open_gdal_dataset(
                constants.DATA_PATH_DSMW))
        
        if data_configuration["GTC"]:
            # channel_cache.append(DataAccessor.open_gdal_dataset(
            #     constants.DATA_PATH_GTC))

            label_cache.append(DataAccessor.open_gdal_dataset(
                constants.DATA_PATH_GTC))
  
        channel_cache = DatasetFactory.__pre_process_data_cache(channel_cache)
            
        # This probably needs to be the very first operation
        if data_configuration["RandomRotation"]:
            transform_list.append(transforms.RandomRotation(
                degrees=data_configuration["RandomRotation"]))

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
        
        transform = transforms.Compose(transform_list)

        return DatasetFactory.__get_data_splits(
            TerrainDataset(DEM_List, 
                           channel_cache,
                           label_cache, 
                           transform),
            data_configuration["Data_Split"])
    
    def __get_data_splits(dataset, training_data_split):
        total_data      = len(dataset)
        training_split  = math.ceil( total_data * training_data_split)
        
        return random_split(dataset, 
                            [training_split, total_data - training_split],
                            generator=torch.Generator()
                                .manual_seed(constants.DATALOADER_SEED))

    
    def __pre_process_data_cache(data_cache):
        processed_cache = []

        for i, cache in enumerate(data_cache, 0):
            if cache.RasterCount == 0:
                    continue

            geo_transform = cache.GetGeoTransform()

            processed_cache.append(
                (geo_transform, 
                 GeoUtil.get_normalized_raster_band(cache.GetRasterBand(1)))
            )

        return processed_cache
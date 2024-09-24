import constants
import math
import torch

from configuration          import Section
from data.data_access       import DataAccessor
from data.data_util         import GeoUtil
from mpl_toolkits.mplot3d   import Axes3D
from osgeo                  import gdal
from torchvision            import transforms
from torch.utils.data       import Dataset, random_split

class DatasetFactory():
    def create_dataset(data_configuration: Section):
        DEM_List = DataAccessor.open_DEM_list()

        channel_cache   = []
        label_cache     = []
        transform_list  = []

        # TODO make it so that the preprocess happens immediately after every load
        # or parallelize it
        if data_configuration["GLiM"]["cache"]:
            if data_configuration["GLiM"]["use_as_channel"]:
                channel_cache.append(DataAccessor.open_gdal_dataset(
                    constants.DATA_PATH_GLIM))
            elif data_configuration["GLiM"]["use_as_label"]:
                pass
            else:
                pass
        
        if data_configuration["climate"]["cache"]:
            if data_configuration["climate"]["use_as_channel"]:
                channel_cache.append(DataAccessor.open_gdal_dataset(
                    constants.DATA_PATH_CLIMATE))
            elif data_configuration["climate"]["use_as_label"]:
                pass
            else:
                pass
        
        if data_configuration["DSMW"]["cache"]:
            if data_configuration["DSMW"]["use_as_channel"]:
                channel_cache.append(DataAccessor.open_gdal_dataset(
                    constants.DATA_PATH_DSMW))
            elif data_configuration["DSMW"]["use_as_label"]:
                pass
            else:
                pass
        
        if data_configuration["GTC"]["cache"]:
            if data_configuration["GTC"]["use_as_channel"]:
                channel_cache.append(DataAccessor.open_gdal_dataset(
                    constants.DATA_PATH_GTC))
            elif data_configuration["GTC"]["use_as_label"]:
                e = DataAccessor.open_gdal_dataset(
                    constants.DATA_PATH_GTC)
                
                geo_transform = e.GetGeoTransform()
                asd = e.GetRasterBand(1).ReadAsArray()

                label_cache.append((geo_transform, asd))
            else:
                pass
  
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
        label = []

        if DEM_dataset.RasterCount == 0:
            return None, None
        
        DEM_array   = GeoUtil.get_normalized_raster_band(
            DEM_dataset.GetRasterBand(1),
            global_min = constants.DEM_GLOBAL_MIN,
            global_max = constants.DEM_GLOBAL_MAX
        )

        DEM_tensor  = torch.tensor(DEM_array, dtype=torch.float32).unsqueeze(0)
        channels    = [DEM_tensor]

        # Extract all the specified additional channel data from other maps
        if self.channel_data_cache or self.label_data_cache: 
            # Get the reference coordinates for the respective data frame
            top_left_geo, bot_right_geo = GeoUtil.get_geo_frame_coordinates(
                DEM_dataset.GetGeoTransform(), 
                (0, 0),
                (DEM_array.shape[1], DEM_array.shape[0]))
            resize = transforms.Resize(DEM_array.shape)
            
            for cache in self.channel_data_cache:
                # Extract the data frame
                data_frame = GeoUtil.get_geo_frame_array(
                    cache[1], 
                    cache[0],
                    top_left_geo,
                    bot_right_geo)
        
                data_tensor = torch.from_numpy(data_frame).unsqueeze(0)
                data_tensor = resize(data_tensor)

                channels.append(data_tensor)
            
            for cache in self.label_data_cache:
                # TODO figure out a way to do multiclass and stuff also One Hot
                # data_frame = GeoUtil.get_geo_frame_array(
                #     cache[1], 
                #     cache[0],
                #     top_left_geo,
                #     bot_right_geo)

                middle_geo = ((bot_right_geo[0]- top_left_geo[0]) 
                              / 2 + top_left_geo[0], 
                              (bot_right_geo[1]- top_left_geo[1]) 
                              / 2 + top_left_geo[1])
                 
                middle_coord = GeoUtil.geo_coordinates_to_cell(cache[0],
                                                               middle_geo[0], 
                                                               middle_geo[1])

                label = [cache[1][middle_coord[1], middle_coord[0]]]
      
        data_entry = torch.cat(channels, dim=0)

        if self.transform:
            data_entry = self.transform(data_entry)

        return data_entry, label
    
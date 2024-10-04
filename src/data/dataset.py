import constants
import math
import multiprocessing
import os
import torch

from configuration          import Section
from data.data_access       import DataAccessor
from data.data_util         import GeoUtil
from debug                  import Printer
from torchvision            import transforms
from torch.utils.data       import Dataset, random_split

class DatasetFactory():
    def create_dataset(data_configuration: Section):
        printer = Printer()
        
        DEM_List_path = data_configuration["DEM_List"]
        printer.print_log(f"Using DEM-list: {DEM_List_path}")
        DEM_List = DataAccessor.open_DEM_list(DEM_List_path)

        if data_configuration["DEM_Dataset"]:
            source_dataset = os.path.join(constants.DATA_PATH_DEM,
                                          data_configuration["DEM_Dataset"])
        else:
            source_dataset = constants.DATA_PATH_DEMS
        
        printer.print_log(f"Using dataset: {os.path.basename(source_dataset)}")

        channel_cache   = []
        label_cache     = []
        transform_list  = []

        # TODO make it so that the preprocess happens immediately 
        # after every load or parallelize it
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
        if data_configuration["Resize"]["active"]:
            printer.print_log("Activating Resize transform")
            transform_list.append(transforms.Resize(
                size=data_configuration["Resize"]["size"]))

        if data_configuration["RandomRotation"]["active"]:
            printer.print_log("Activating RandomRotation transform")
            transform_list.append(transforms.RandomRotation())

        if data_configuration["RandomCrop"]["active"]:
            printer.print_log("Activating RandomCrop transform")
            transform_list.append(transforms.RandomCrop(
                size=data_configuration["RandomCrop"]["size"]))
            
        if data_configuration["FiveCrop"]["active"]:
            printer.print_log("Activating FiveCrop transform")
            transform_list.append(transforms.FiveCrop(
                size=data_configuration["FiveCrop"]["Size"]))
            
        if data_configuration["TenCrop"]["active"]:
            printer.print_log("Activating TenCrop transform")
            transform_list.append(transforms.TenCrop(
                size=data_configuration["TenCrop"]["Size"]))
            
        if data_configuration["RandomHorizontalFlip"]["active"]:
            printer.print_log("Activating RandomHorizontalFlip transform")
            transform_list.append(transforms.RandomHorizontalFlip(
                p=data_configuration["RandomHorizontalFlip"]["probability"]))
            
        if data_configuration["RandomVerticalFlip"]["active"]:
            printer.print_log("Activating RandomVerticalFlip transform")
            transform_list.append(transforms.RandomHorizontalFlip(
                p=data_configuration["RandomVerticalFlip"]["probability"]))
        
        transform = transforms.Compose(transform_list)

        return DatasetFactory.__get_data_splits(
            TerrainDataset(DEM_List, 
                           channel_cache,
                           label_cache, 
                           transform,
                           source_dataset,
                           cache_dems=data_configuration["Cache_DEMs"]),
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
    def __create_shared_cache():
        manager = multiprocessing.Manager()
        shared_cache = manager.dict()
        return shared_cache

    def __init__(self, 
                 DEM_list, 
                 channel_data_cache,
                 label_data_cache, 
                 transform=None,
                 source_dataset=constants.DATA_PATH_DEMS,
                 cache_dems = True):
        self.transform              = transform
        self.DEM_list               = DEM_list
        self.channel_data_cache     = channel_data_cache
        self.label_data_cache       = label_data_cache
        self.source_dataset         = source_dataset
        
        manager                     = multiprocessing.Manager()
        self.dem_cache              = manager.list([None] * len(DEM_list))
        self.cache_dems             = cache_dems

    def __len__(self):
        return len(self.DEM_list)
    
    def __getitem__(self, index):
        metadata    = {}
        label       = []
        filename    = self.DEM_list[index]

        metadata["filename"] = filename

        if self.cache_dems and self.dem_cache[index]:
            DEM_geo_transform, DEM_tensor = self.dem_cache[index]   
        else:   
            DEM_dataset = DataAccessor.open_DEM(filename, self.source_dataset)
            DEM_geo_transform = None

            if DEM_dataset.RasterCount == 0:
                return None, None    

            DEM_tensor = GeoUtil.get_normalized_raster_band(
                DEM_dataset.GetRasterBand(1),
                nodata_val = None, #constants.DEM_NODATA_VAL,
                global_min = constants.DEM_GLOBAL_MIN,
                global_max = constants.DEM_GLOBAL_MAX
            )
            
            if self.cache_dems:
                self.dem_cache[index] = (DEM_dataset.GetGeoTransform(), 
                                         DEM_tensor)

    
        DEM_shape   = DEM_tensor.shape
        DEM_tensor  = DEM_tensor.unsqueeze(0)
        
        
        channels    = [DEM_tensor]

        # Extract all the specified additional channel data from other maps
        if self.channel_data_cache or self.label_data_cache: 
            if DEM_geo_transform is None: 
                DEM_geo_transform = DEM_dataset.GetGeoTransform()
            
            # Get the reference coordinates for the respective data frame
            top_left_geo, bot_right_geo = GeoUtil.get_geo_frame_coordinates(
                DEM_geo_transform, 
                (0, 0),
                (DEM_shape[1], DEM_shape[0]))
            resize = transforms.Resize(DEM_shape)
            
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

        return data_entry, label, metadata
    
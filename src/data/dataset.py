import constants
import math
import multiprocessing
import os
import torch
import numpy                as np

from configuration          import Section
from data.data_access       import DataAccessor
from data.data_util         import GeoUtil, NoDataBehaviour, NormalizationMethod
from debug                  import Printer
from torchvision            import transforms
from torch.utils.data       import Dataset, random_split
from tqdm                   import tqdm

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


from data.data_util import DataVisualizer

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
        DatasetFactory.__append_to_cache(data_configuration["GLiM"], 
                                        channel_cache, 
                                        label_cache)

        DatasetFactory.__append_to_cache(data_configuration["Climate"], 
                                        channel_cache, 
                                        label_cache)

        DatasetFactory.__append_to_cache(data_configuration["DSMW"], 
                                        channel_cache, 
                                        label_cache)
        
        DatasetFactory.__append_to_cache(data_configuration["GTC"], 
                                         channel_cache, 
                                         label_cache)
    
  
        channel_cache   = DatasetFactory.__pre_process_data_cache(channel_cache)
        amount_classes  = DatasetFactory.__get_amount_classes(label_cache)
            
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
        
        transform           = transforms.Compose(transform_list)
        complete_dataset    = TerrainDataset(
            DEM_List, 
            channel_cache,
            label_cache, 
            transform,
            source_dataset,
            cache_dems=data_configuration["Cache_DEMs"],
            amount_classes=amount_classes) 
        
        training_dataset, eval_dataset = DatasetFactory.__get_data_splits(
            complete_dataset,
            data_configuration["Data_Split"])
        
        #complete_dataset.preprocess_dataset()
        #complete_dataset.analyse_dataset()

        return training_dataset, eval_dataset, amount_classes
    
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
    

    def __append_to_cache(dataset_configuration, channel_cache, label_cache):
        if not dataset_configuration["cache"]:
            return 

        if dataset_configuration["use_as_channel"]:
            channel_cache.append(DataAccessor.open_gdal_dataset(
                constants.DATA_PATH_GTC))
        elif dataset_configuration["use_as_label"]:
            dataset = DataAccessor.open_gdal_dataset(constants.DATA_PATH_GTC)
            geo_transform   = dataset.GetGeoTransform()
            geo_array       = dataset.GetRasterBand(1).ReadAsArray()

            label_cache.append((geo_transform, geo_array))

    # TODO Multilabel will not work with this
    def __get_amount_classes(label_cache):
        amount_classes = 0
        
        # Might have to consider nodata later
        for geo_transform, geo_array in label_cache:
            amount_classes += len(np.unique(geo_array))            

        return amount_classes
    
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
                 cache_dems = True,
                 amount_classes = 0):
        manager                     = multiprocessing.Manager()

        self.transform              = transform
        self.DEM_list               = DEM_list
        self.source_dataset         = source_dataset

        self.channel_data_cache     = channel_data_cache
        self.label_data_cache       = manager.list(label_data_cache)
        
        self.dem_cache              = manager.list([None] * len(DEM_list))
        self.cache_dems             = cache_dems

        self.amount_classes         = amount_classes

    def __len__(self):
        return len(self.DEM_list)
    
    def __getitem__(self, index):
        metadata        = {}
        filename        = self.DEM_list[index]
        label           = []

        metadata["filename"] = filename

        if self.cache_dems and self.dem_cache[index]:
            DEM_geo_transform   = self.dem_cache[index].dem_geo_transform
            DEM_tensor          = self.dem_cache[index].dem_tensor 
            label               = self.dem_cache[index].label_tensors

            DEM_shape = DEM_tensor.shape[-3:]
            channels    = [DEM_tensor]
            
            if self.label_data_cache:
                self.label_data_cache = None

            data_entry = torch.cat(channels, dim=0)

            if self.transform:
                data_entry = self.transform(data_entry)

            return data_entry, label, metadata


        else:   
            DEM_dataset = DataAccessor.open_DEM(filename, self.source_dataset)
            DEM_geo_transform = None

            if DEM_dataset.RasterCount == 0:
                return None, None    

            DEM_tensor = GeoUtil.get_normalized_raster_band(
                DEM_dataset.GetRasterBand(1),
                nodata_behaviour        = NoDataBehaviour.NONE,
                normalization_method    = NormalizationMethod.CLIPPED_LINEAR,
                nodata_value            = constants.DEM_NODATA_VAL,
                global_min              = constants.DEM_GLOBAL_MIN,
                global_max              = constants.DEM_GLOBAL_MAX
            )

            DEM_shape   = DEM_tensor.shape
            DEM_tensor  = DEM_tensor.unsqueeze(0)
        
        channels    = [DEM_tensor]

        # Extract all the specified additional channel data from other maps
        if (not label) and (self.channel_data_cache or self.label_data_cache): 
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
                data_frame = GeoUtil.get_geo_frame_array(
                    cache[1], 
                    cache[0],
                    top_left_geo,
                    bot_right_geo)

                label = torch.tensor(np.median(data_frame), dtype=torch.int32)

        if self.cache_dems and not self.dem_cache[index]:
            self.dem_cache[index] = GeoDatasetCache(
                DEM_dataset.GetGeoTransform(), 
                DEM_tensor,
                label_tensors = label)
      
        data_entry = torch.cat(channels, dim=0)

        if self.transform:
            data_entry = self.transform(data_entry)

        return data_entry, label, metadata
    
    # def preprocess_dataset(self):
    #     for index in tqdm(range(len(self.DEM_list) - 1),
    #                       total = len(self.DEM_list),
    #                       desc = "Preprocessing Dataset"):
    #         self.__getitem__(index)

    def preprocess_dataset(self):
        with ThreadPoolExecutor() as executor:
            indices = range(len(self.DEM_list) - 1)
            list(tqdm(executor.map(self.__getitem__, indices),
                    total=len(indices), 
                    desc="Preprocessing Dataset"))
    
    def analyse_dataset(self):
        analysis_result = AnalysisResult(self.amount_classes)
        
        for idx, cache in tqdm(enumerate(self.dem_cache), 
                               total=len(self.dem_cache),
                               desc="Analysing data"):
            if cache is None: 
                continue
            label = cache.label_tensors.item()
            
            analysis_result.label_bucket[label] += 1
            analysis_result.std_devs[label].append(
                np.std(cache.dem_tensor.numpy()))
            
        analysis_result.post_process()
        print (analysis_result)


class AnalysisResult():
    def __init__(self, amount_classes):
        self.amount_classes     = amount_classes
        self.label_bucket       = [0]  * amount_classes
        self.std_devs           = [[] for _ in range(amount_classes)]
        self.std_dev_bucket     = [0]  * amount_classes     

    def post_process(self):
        for idx in range(len(self.std_devs)):
            self.std_dev_bucket[idx] = np.mean(self.std_devs[idx])

    def __str__(self):
        string = ""
        for idx, (amount, sigma) in enumerate(zip(self.label_bucket, self.std_dev_bucket)):
            string += f"\nLabel {idx}; Amount: {amount}, \tSigma {sigma}"

        return string

class GeoDatasetCache():
    def __init__(self,
                 dem_geo_transform, 
                 dem_tensor,
                 label_geo_transforms   = None,
                 label_tensors          = [], 
                 channel_geo_transforms = None,
                 channel_tensors        = None):
        
        self.dem_geo_transform      = dem_geo_transform
        self.dem_tensor             = dem_tensor
        self.label_geo_transforms   = label_geo_transforms
        self.label_tensors          = label_tensors
        self.channel_geo_transforms = channel_geo_transforms
        self.channel_tensors        = channel_tensors

    def set_label_tensors(self, value):
        self.label_tensors = value



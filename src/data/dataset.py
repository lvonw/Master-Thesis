import constants
import multiprocessing
import os
import torch

import numpy                                as np
import torchvision.transforms.functional    as tf

from concurrent.futures import ThreadPoolExecutor
from configuration      import Section
from data.data_access   import DataAccessor
from data.data_util     import GeoUtil, NoDataBehaviour, NormalizationMethod
from debug              import Printer
from torchvision        import transforms
from torch.utils.data   import Dataset
from tqdm               import tqdm


class DatasetFactory():
    def create_dataset(data_configuration: Section):
        printer = Printer()
        
        DEM_List_path = data_configuration["DEM_List"]
        printer.print_log(f"Using DEM-list: {DEM_List_path}")

        dems_to_take = None
        if data_configuration["amount_dems"] > 0:
            dems_to_take = data_configuration["amount_dems"]
        
        DEM_List = DataAccessor.open_DEM_list(DEM_List_path)[:dems_to_take]

        if data_configuration["DEM_Dataset"]:
            source_dataset = os.path.join(constants.DATA_PATH_DEM,
                                          data_configuration["DEM_Dataset"])
        else:
            source_dataset = constants.DATA_PATH_DEMS
        
        printer.print_log(f"Using dataset: {os.path.basename(source_dataset)}")

        channel_cache   = []
        label_cache     = []
        transform_list  = []

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
            transform_list.append(Random90DegreeRotation())
            
        if data_configuration["RandomCrop"]["active"]:
            printer.print_log("Activating RandomCrop transform")
            random_crop = RandomCropWithFrame(
                size=data_configuration["RandomCrop"]["size"])   
             
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
            amount_classes=amount_classes,
            random_crop=random_crop) 
        
        complete_dataset.prepare_dataset(data_configuration["loader_workers"])

        return complete_dataset, amount_classes
    
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
    def __init__(self, 
                 DEM_list, 
                 channel_cache,
                 label_cache, 
                 transform=None,
                 source_dataset=constants.DATA_PATH_DEMS,
                 cache_dems = True,
                 amount_classes = 0,
                 random_crop = None):
        self.printer                = Printer()

        self.DEM_list               = DEM_list
        self.transform              = transform
        self.source_dataset         = source_dataset
        
        self.cache_dems             = cache_dems
        self.amount_classes         = amount_classes

        # Single-Process cache
        self.dataset_cache          = [None] * len(DEM_list)
        self.channel_cache          = channel_cache
        self.label_cache            = label_cache

        # Thread-Safe cache, introduces massive overheads        
        self.shared_channel_cache   = None
        self.shared_label_cache     = None
        self.shared_dataset_cache   = None

        self.loss_weights = {None: 1}

        self.random_crop = random_crop

    def __len__(self):
        return len(self.DEM_list)

    def __getitem__(self, index):        
        cache       = self.shared_dataset_cache[index]
        metadata    = cache.metadata

        if cache.did_not_cache_dem:
            pass
        else:
            dem_tensor  = cache.dem_tensor 

        # Channels ============================================================
        if cache.did_not_cache_channels:
            pass
        if cache.channel_tensor is not None:
            channels    = [dem_tensor, cache.label_tensor]        
            data_entry = torch.cat(channels, dim=0)
        else:
            data_entry = dem_tensor

        
        # Labels ==============================================================
        if cache.did_not_cache_labels:
            pass
        else:
            label_frame = cache.label_frame

        # Transforms ==========================================================
        if self.random_crop is not None:
            data_entry, label_frame = self.random_crop(data_entry, 
                                                        label_frame)
        if self.transform:
            data_entry  = self.transform(data_entry)
            
        if label_frame is not None and label_frame.numel() > 0:
            label_frame = self.transform(label_frame)
            label = torch.median(label_frame)
        else:
            label = cache.label_tensor

        return data_entry, label, metadata
    
    def analyse_dataset(self):
        analysis_result = AnalysisResult(self.amount_classes)
        
        for idx, cache in tqdm(enumerate(self.dataset_cache), 
                               total=len(self.dataset_cache),
                               desc="Analysing data"):
            if cache is None or cache.label_tensor is None: 
                continue
            label = cache.label_tensor.item()
            
            analysis_result.label_bucket[label] += 1
            analysis_result.std_devs[label].append(
                np.std(cache.dem_tensor.numpy()))
            
        analysis_result.post_process()
        return analysis_result

    def prepare_dataset(self, loader_workers=1):
        with ThreadPoolExecutor(max_workers=loader_workers) as executor:
            indices = range(len(self.DEM_list))
            list(tqdm(executor.map(self.__prefetch_cache, indices),
                    total=len(indices), 
                    desc="Preparing Dataset"))
            

        analysis_result = self.analyse_dataset()
        #self.printer.print_log(analysis_result)

        for label, label_amount in enumerate(analysis_result.label_bucket):
            # self.loss_weights[label] = len(self.DEM_list) / label_amount 
            # self.loss_weights[label] = np.log(len(self.DEM_list) 
            # / label_amount) 
            # self.loss_weights[label] = 1 
            # + analysis_result.std_dev_bucket[label]
            # self.loss_weights[label] = len(self.DEM_list) / (label_amount)
            pass
        self.loss_weights = None

        # Transfer our single process cache to the shared cache
        manager                     = multiprocessing.Manager()
        self.shared_dataset_cache   = manager.list(self.dataset_cache)
        self.dataset_cache.clear()

        
            
    def __prefetch_cache(self, index):
        filename                    = self.DEM_list[index]
        geo_cache                   = GeoDatasetCache()
        self.dataset_cache[index]   = geo_cache
        
        metadata            = {"filename": filename}
        geo_cache.metadata  = metadata

        # DEMs ================================================================
        dem_tensor, dem_shape, dem_dataset  = self.__load_dem(filename)  
        geo_cache.dem_tensor                = dem_tensor

        if self.channel_cache or self.label_cache:
            (top_left_geo, bot_right_geo), dem_geo_transform = (
                self.__load_dem_geo_coordinates(dem_dataset, dem_shape))
            
            geo_cache.dem_geo_coordinates   = (top_left_geo, bot_right_geo)
            geo_cache.dem_geo_transform     = dem_geo_transform
        else:
            return
        
        # Channels ============================================================
        if self.channel_cache:
            channels            = []
            resize_to_dem_shape = transforms.Resize(dem_shape)
            
            for cache in self.channel_cache:
                channels.append(self.__load_channel(cache, 
                                                    top_left_geo, 
                                                    bot_right_geo,
                                                    resize_to_dem_shape))
                
                geo_cache.channel_geo_transforms.append(cache[0])

            channels = torch.cat(channels, dim=0)
            geo_cache.channel_tensor = channels

        # Labels ==============================================================
        if self.label_cache:
            for cache in self.label_cache:
                label, label_data_frame = self.__load_label(cache, 
                                                            top_left_geo, 
                                                            bot_right_geo)

                geo_cache.label_geo_transforms.append(cache[0]) 
                geo_cache.label_tensor  = label
                geo_cache.label_frame   = label_data_frame

    
    def __load_dem(self, filename):
        dem_dataset = DataAccessor.open_DEM(filename, self.source_dataset) 
        
        if dem_dataset.RasterCount == 0:
            return    

        dem_tensor = GeoUtil.get_normalized_raster_band(
            dem_dataset.GetRasterBand(1),
            nodata_behaviour        = NoDataBehaviour.NONE,
            normalization_method    = NormalizationMethod.CLIPPED_LINEAR,
            nodata_value            = constants.DEM_NODATA_VAL,
            global_min              = constants.DEM_GLOBAL_MIN,
            global_max              = constants.DEM_GLOBAL_MAX,
            new_min                 = -1.,
            new_max                 = 1.            
        )

        dem_shape               = dem_tensor.shape
        dem_tensor              = dem_tensor.unsqueeze(0)

        return dem_tensor, dem_shape, dem_dataset

    def __load_dem_geo_coordinates(self, dem_dataset, dem_shape):
        dem_geo_transform = dem_dataset.GetGeoTransform()
            
        # Get the reference coordinates for the respective data frame
        top_left_geo, bot_right_geo = GeoUtil.get_geo_frame_coordinates(
            dem_geo_transform, 
            (0, 0),
            (dem_shape[1], dem_shape[0]))

        return (top_left_geo, bot_right_geo), dem_geo_transform

    def __load_channel(self, 
                       cache, 
                       top_left_geo, 
                       bot_right_geo, 
                       resize_transform):
        # Extract the data frame
        data_frame = GeoUtil.get_geo_frame_array(
            cache[1], 
            cache[0],
            top_left_geo,
            bot_right_geo)

        data_tensor = torch.from_numpy(data_frame).unsqueeze(0)
        data_tensor = resize_transform(data_tensor)
        
        return data_tensor

    def __load_label(self, cache, top_left_geo, bot_right_geo):
        data_frame  = GeoUtil.get_geo_frame_array(cache[1], 
                                                  cache[0],
                                                  top_left_geo,
                                                  bot_right_geo)
        label       = torch.tensor(np.median(data_frame), dtype=torch.int32)
        data_frame  = torch.tensor(data_frame, dtype=torch.int32)

        return label, data_frame

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
        for idx, (amount, sigma) in enumerate(zip(self.label_bucket, 
                                                  self.std_dev_bucket)):
            string += f"\nLabel {idx}; Amount: {amount}, \tSigma {sigma}"

        return string

class GeoDatasetCache():
    def __init__(self,
                 dem_tensor             = None,            
                 dem_geo_transform      = None,
                 dem_geo_coordinates    = None,
                 label_geo_transforms   = [],
                 label_tensor           = None, 
                 channel_geo_transforms = [],
                 channel_tensor         = None,
                 metadata               = {}):
        
        self.metadata               = metadata

        self.dem_geo_transform      = dem_geo_transform
        self.dem_tensor             = dem_tensor
        self.dem_geo_coordinates    = dem_geo_coordinates

        self.label_geo_transforms   = label_geo_transforms
        self.label_frame            = None
        self.label_tensor           = label_tensor

        self.channel_geo_transforms = channel_geo_transforms
        self.channel_tensor         = channel_tensor

        self.did_not_cache_dem      = False
        self.did_not_cache_channels = False
        self.did_not_cache_labels   = False

        self.uncached_dem           = None
        self.uncached_channels      = []        
        self.uncached_labels        = []


class RandomCropWithFrame():
    def __init__(self, size):
        self.cropped_size = size

    def __call__(self, image_tensor, label_frame = None):
        _, height, width = image_tensor.shape

        top     = np.random.randint(0, height - self.cropped_size)
        left    = np.random.randint(0, width  - self.cropped_size)

        cropped_img = tf.crop(image_tensor, 
                              top, 
                              left, 
                              self.cropped_size, 
                              self.cropped_size)
        
        if label_frame is None:
            return cropped_img, None
        
        label_shape     = label_frame.shape 
        scaling_factor  = label_shape[0] / height
        
        top     *= scaling_factor
        left    *= scaling_factor
        size    = self.cropped_size * scaling_factor
        height  = min(size, label_shape[0])
        width   = min(size, label_shape[1])

        label_frame = tf.crop(label_frame,
                              int(top),
                              int(left),
                              int(height),
                              int(width))
        
        return cropped_img, label_frame

class Random90DegreeRotation():
    def __call__(self, image_tensor):
        angle = 90 * np.random.randint(0, 4)

        return tf.rotate(image_tensor, angle)
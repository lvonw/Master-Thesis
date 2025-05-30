import constants
import math
import multiprocessing
import os
import torch

import numpy                                as np
import torchvision.transforms.functional    as tf

from concurrent.futures     import ThreadPoolExecutor
from configuration          import Section
from data.data_access       import DataAccessor
from data.data_util         import (GeoUtil, 
                                    NoDataBehaviour, 
                                    NormalizationMethod)
from debug                  import Printer
from torchvision            import transforms
from torchvision.datasets   import MNIST
from torch.utils.data       import Dataset, random_split
from tqdm                   import tqdm


class DatasetFactory():
    def create_dataset(data_configuration: Section, 
                       prepare = False):
        printer = Printer()

        if data_configuration["use_MNIST"]:
            printer.print_log("Using MNIST")
            return DatasetFactory.__get_mnist()
        
        printer.print_log("Using DEMs")

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
                                         constants.DATA_PATH_GLIM,
                                         channel_cache, 
                                         label_cache)

        DatasetFactory.__append_to_cache(data_configuration["DSMW"], 
                                         constants.DATA_PATH_DSMW,
                                         channel_cache, 
                                         label_cache)
        
        DatasetFactory.__append_to_cache(data_configuration["GTC"], 
                                         constants.DATA_PATH_GTC,
                                         channel_cache, 
                                         label_cache)
        
        DatasetFactory.__append_to_cache(data_configuration["Climate"], 
                                         constants.DATA_PATH_CLIMATE,
                                         channel_cache, 
                                         label_cache)    
  
        channel_cache  = DatasetFactory.__pre_process_data_cache(channel_cache)
        amount_classes = DatasetFactory.get_label_amounts(data_configuration)

        if data_configuration["RandomCrop"]["active"]:
            printer.print_log("Activating RandomCrop transform")
            transform_list.append(RandomCropWithFrame(
                size=data_configuration["RandomCrop"]["size"]))   
        
        if data_configuration["RandomRotation"]["active"]:
            printer.print_log("Activating RandomRotation transform")
            transform_list.append(Random90DegreeRotation())
             
        if data_configuration["RandomHorizontalFlip"]["active"]:
            printer.print_log("Activating RandomHorizontalFlip transform")
            transform_list.append(RandomHorizontalFlip(
                p=data_configuration["RandomHorizontalFlip"]["probability"]))
            
        if data_configuration["RandomVerticalFlip"]["active"]:
            printer.print_log("Activating RandomVerticalFlip transform")
            transform_list.append(RandomVerticalFlip(
                p=data_configuration["RandomVerticalFlip"]["probability"]))
        
        transform           = CompositeMultiTensorTransform(transform_list)
        terrain_dataset     = TerrainDataset(
            DEM_List, 
            channel_cache,
            label_cache, 
            transform,
            source_dataset,
            cache_dems=data_configuration["Cache_DEMs"],
            amount_classes=amount_classes) 
        
        if prepare:
            loss_weights        = terrain_dataset.prepare_dataset(
                data_configuration["loader_workers"])
        else: 
            loss_weights        = {}
        
        return DatasetWrapper(dataset           = terrain_dataset, 
                              amount_classes    = amount_classes,
                              loss_weights      = loss_weights)
    
    def get_label_amounts(configuration):
        if configuration["use_MNIST"]:
            return [10]
        
        labels_amounts = []

        if configuration["GLiM"]["use_as_label"]:
            pass
        if configuration["DSMW"]["use_as_label"]:
            pass
        if configuration["GTC"]["use_as_label"]:
            labels_amounts.append(constants.LABEL_AMOUNT_GTC)
        if configuration["Climate"]["use_as_label"]:
            labels_amounts.append(constants.LABEL_AMOUNT_CLIMATE)

        return labels_amounts
    
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
    
    def __append_to_cache(dataset_configuration, 
                          dataset_path,
                          channel_cache, 
                          label_cache):
        if not dataset_configuration["cache"]:
            return 

        if dataset_configuration["use_as_channel"]:
            channel_cache.append(DataAccessor.open_gdal_dataset(dataset_path))
        elif dataset_configuration["use_as_label"]:
            dataset = DataAccessor.open_gdal_dataset(dataset_path)
            geo_transform   = dataset.GetGeoTransform()
            geo_array       = dataset.GetRasterBand(1).ReadAsArray()

            label_cache.append((geo_transform, geo_array))

    def __get_amount_classes(label_cache):
        # Deprecated use get_label_amounts() instead
        amount_classes = 0
        
        # Might have to consider nodata later
        for geo_transform, geo_array in label_cache:
            amount_classes += len(np.unique(geo_array))            

        return amount_classes
    
    def __get_mnist():
        mnist_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(1, 0.5),
                                              transforms.Resize(size=32)])

        training_set    = MNIST(root=constants.DATASET_PATH_MASTER, 
                                train=True, 
                                download=True, 
                                transform=mnist_transform)

        validation_set  = MNIST(root=constants.DATASET_PATH_MASTER, 
                                train=False, 
                                download=True, 
                                transform=mnist_transform)
        
        return DatasetWrapper(training_dataset      = training_set, 
                              validation_dataset    = validation_set,
                              amount_classes        = [10])
    

class DatasetWrapper():
    def __init__(self, 
                 dataset            = None,
                 amount_classes     = 0,
                 loss_weights       = {},
                 training_dataset   = None,
                 validation_dataset = None):
        
        self.dataset            = dataset
        self.amount_classes     = amount_classes
        self.loss_weights       = loss_weights

        self.training_dataset   = training_dataset
        self.validation_dataset = validation_dataset

    def get_splits(self, split=0.95):
        if self.dataset is None:
            return self.training_dataset, self.validation_dataset
        else:
            return self.__get_data_splits(self.dataset, split)

    def __get_data_splits(self, dataset, training_data_split):
        total_data      = len(dataset)
        training_split  = math.ceil( total_data * training_data_split)
        
        return random_split(dataset, 
                            [training_split, total_data - training_split],
                            generator=torch.Generator()
                                .manual_seed(constants.DATALOADER_SEED))
    
class TerrainDataset(Dataset): 
    def __init__(self, 
                 DEM_list, 
                 channel_cache,
                 label_cache, 
                 transform      = None,
                 source_dataset = constants.DATA_PATH_DEMS,
                 cache_dems     = True,
                 amount_classes = 0):
        self.printer                = Printer()

        self.dem_list               = DEM_list
        self.transform              = transform
        self.source_dataset         = source_dataset
        
        self.cache_dems             = cache_dems
        self.amount_classes         = amount_classes

        # Single-Process cache
        self.dataset_cache          = [None] * len(DEM_list)
        self.channel_cache          = channel_cache
        self.label_cache            = label_cache

        # Thread-Safe cache, introduces massive overheads for single threads
        manager                     = multiprocessing.Manager()
        self.shared_dataset_cache   = manager.list()       

        # For when we only want to load as we go
        self.prepared               = False 

    def __len__(self):
        return len(self.dem_list)

    def __getitem__(self, index):
        if self.prepared:        
            cache       = self.shared_dataset_cache[index]
            metadata    = cache.metadata
        else: 
            cache       = GeoDatasetCache()

        # Channels ============================================================
        if not cache.did_cache_dem:
            dem_file                            = self.dem_list[index]
            cache.metadata["filename"]          = dem_file
            dem_tensor, dem_shape, dem_dataset  = self.__load_dem(dem_file)  

            if self.channel_cache or self.label_cache:
                (top_left_geo, bot_right_geo), dem_geo_transform = (
                    self.__load_dem_geo_coordinates(dem_dataset, dem_shape))
                
                cache.dem_geo_coordinates   = (top_left_geo, bot_right_geo)
                cache.dem_geo_transform     = dem_geo_transform
        else:
            dem_tensor  = cache.dem_tensor 

        # Channels ============================================================
        if not cache.did_cache_channels:
            data_entry  = dem_tensor
        if len(cache.channel_tensor) > 0:
            channels    = [dem_tensor, cache.label_tensor]        
            data_entry  = torch.cat(channels, dim=0)
        else:
            data_entry  = dem_tensor

        
        # Label Frames ========================================================
        if not cache.did_not_cache_labels:
            for cache in self.label_cache:
                label, label_data_frame = self.__load_label(cache, 
                                                            top_left_geo, 
                                                            bot_right_geo)

                cache.label_geo_transforms.append(cache[0]) 
                cache.label_tensor = label
                cache.label_frames.append(label_data_frame)
    
        label_frames = cache.label_frames

        # Transforms ==========================================================
        if self.transform:
            data_entry, label_frames = self.transform(data_entry, 
                                                      label_frames,
                                                      fast=True)
        
        # Label ===============================================================
        if label_frames is not None and len(label_frames) > 0:
            labels = []
            for _, label_frame in enumerate(label_frames):
                label = torch.median(label_frame)
                label = torch.tensor(0) if label.item() < 0 else label
                labels.append(label)
            label = torch.stack(labels, dim=0)
            
        else:
            label = cache.label_tensor

        return data_entry, label, metadata

    def get_data_by_name(self, filename):
        try:
            index = self.dem_list.index(filename)
            return self[index]

        except ValueError:
            self.printer.print_log("Item not in list")

            return None, None, None
        
    def prepare_dataset(self, loader_workers=1, analyse = False):
        with ThreadPoolExecutor(max_workers=loader_workers) as executor:
            indices = range(len(self.dem_list))
            list(tqdm(executor.map(self.__prefetch_cache, indices),
                    total=len(indices), 
                    desc="Preparing Dataset"))
        
        self.printer.print_log(f"Cached {len(self.dataset_cache)} Items")
        
        # Analysis ============================================================
        loss_weights = None
        if analyse:
            analysis_result = self.__analyse_dataset()
            self.printer.print_log(analysis_result)
            for label, label_amount in enumerate(analysis_result.label_bucket):
                self.loss_weights[label] = len(self.dem_list) / (label_amount)
        

        # Transfer our single process cache to the shared cache
        self.printer.print_log("Transferring to shared cache...")

        manager                     = multiprocessing.Manager()
        self.shared_dataset_cache   = manager.list(self.dataset_cache)
        
        self.dataset_cache.clear()
        self.printer.print_log("Finished")

        self.prepared = True

        return loss_weights
        
    def __prefetch_cache(self, index):
        filename                    = self.dem_list[index]
        geo_cache                   = GeoDatasetCache()
        self.dataset_cache[index]   = geo_cache
        
        metadata            = {"filename": filename}
        geo_cache.metadata  = metadata

        # DEMs ================================================================
        dem_tensor, dem_shape, dem_dataset  = self.__load_dem(filename)  
        geo_cache.dem_tensor                = dem_tensor
        geo_cache.did_cache_dem             = True

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
            geo_cache.channel_tensor        = channels
            geo_cache.did_cache_channels    = True

        # Labels ==============================================================
        if self.label_cache:
            for cache in self.label_cache:
                label, label_data_frame = self.__load_label(cache, 
                                                            top_left_geo, 
                                                            bot_right_geo)

                geo_cache.label_geo_transforms.append(cache[0]) 
                geo_cache.label_tensor  = label
                geo_cache.label_frames.append(label_data_frame)
            
            geo_cache.did_cache_labels    = True


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
        data_frame  = torch.tensor(data_frame, 
                                   dtype=torch.int32).unsqueeze(dim=0)

        return label, data_frame
      
    def __analyse_dataset(self):
        analysis_result = AnalysisResult(self.amount_classes[0])
        
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
                 label_tensor           = [], 
                 channel_geo_transforms = [],
                 channel_tensor         = [],
                 metadata               = {}):
        
        self.metadata               = metadata

        self.dem_geo_transform      = dem_geo_transform
        self.dem_tensor             = dem_tensor
        self.dem_geo_coordinates    = dem_geo_coordinates

        self.label_geo_transforms   = label_geo_transforms
        self.label_frames           = []
        self.label_tensor           = label_tensor

        self.channel_geo_transforms = channel_geo_transforms
        self.channel_tensor         = channel_tensor

        self.did_cache_dem          = False
        self.did_cache_channels     = False
        self.did_cache_labels       = False

        self.uncached_dem           = None
        self.uncached_channels      = []        
        self.uncached_labels        = []

class RandomCropWithFrame():
    def __init__(self, size):
        self.cropped_size = size

    def __call__(self, image_tensor, label_frames = None, fast=False):
        _, height, width = image_tensor.shape

        top     = np.random.randint(0, height - self.cropped_size)
        left    = np.random.randint(0, width  - self.cropped_size)

        cropped_img = tf.crop(image_tensor, 
                              top, 
                              left, 
                              self.cropped_size, 
                              self.cropped_size)
        
        if label_frames is None or not len(label_frames):
            return cropped_img, res_label_frames
        
        res_label_frames = [None] * len(label_frames)
        for idx, label_frame in enumerate(label_frames):
            label_shape     = label_frame.shape 
            scaling_factor  = label_shape[0] / height
            
            top     *= scaling_factor
            left    *= scaling_factor
            size    = math.ceil(self.cropped_size * scaling_factor)
            height  = min(size, label_shape[0])
            width   = min(size, label_shape[1])

            res_label_frames[idx] = tf.crop(label_frame,
                                            int(top),
                                            int(left),
                                            height,
                                            width)
        
        return cropped_img, res_label_frames

class Random90DegreeRotation():
    def __call__(self, image_tensor, label_frames = None, fast=False):
        angle = 90 * np.random.randint(0, 4)

        image_tensor = tf.rotate(image_tensor, angle)

        if fast or label_frames is None or not len(label_frames):
            return image_tensor, label_frames
        
        res_label_frames = [None] * len(label_frames)
        for idx, label_frame in enumerate(label_frames):
            res_label_frames[idx] = tf.rotate(label_frame, angle=angle)

        return image_tensor, res_label_frames
        
class RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.probability = p

    def __call__(self, image_tensor, label_frames = None, fast=False):
        should_flip = np.random.rand()

        if should_flip < self.probability:
            image_tensor = tf.hflip(image_tensor)
        
        if fast or label_frames is None or not len(label_frames):
            return image_tensor, label_frames
        
        res_label_frames = [None] * len(label_frames)
        for idx, label_frame in enumerate(label_frames):
            if should_flip < self.probability:
                res_label_frames[idx] = tf.hflip(label_frame)

        return image_tensor, res_label_frames
    
class RandomVerticalFlip():
    def __init__(self, p=0.5):
        self.probability = p

    def __call__(self, image_tensor, label_frames = None, fast=False):
        should_flip = np.random.rand()

        if should_flip < self.probability:
            image_tensor = tf.vflip(image_tensor)
        
        if fast or label_frames is None or not len(label_frames):
            return image_tensor, label_frames
        
        res_label_frames = [None] * len(label_frames)
        for idx, label_frame in enumerate(label_frames):
            if should_flip < self.probability:
                res_label_frames[idx] = tf.vflip(label_frame)

        return image_tensor, res_label_frames
    
class CompositeMultiTensorTransform():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_tensor, label_frames = None, fast=False):
        for transform in self.transforms:
            image_tensor, label_frames = transform(image_tensor, 
                                                  label_frames,
                                                  fast=fast)

        return image_tensor, label_frames
    
    
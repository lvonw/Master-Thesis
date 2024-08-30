import constants
import os

from osgeo import gdal


class DataAccessor():
    def open_gdal_dataset(dataset_path):
        dataset = gdal.Open(dataset_path) 

        if not dataset:
            print(f"gdal dataset {dataset_path} could not be loaded")
        
        return dataset
    

    def open_DEM(name):
        return DataAccessor.open_gdal_dataset(
             os.path.join(constants.DATA_PATH_DEMS, name))
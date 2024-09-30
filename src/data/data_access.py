import constants
import os

from osgeo import gdal


class DataAccessor():
    def open_gdal_dataset(dataset_path, dataset=constants.DATA_PATH_DEMS):
        dataset = gdal.Open(dataset_path) 

        if not dataset:
            print(f"gdal dataset {dataset_path} could not be loaded")
        
        return dataset
    
    def open_DEM(name, dataset=constants.DATA_PATH_DEMS):
        return DataAccessor.open_gdal_dataset(
             os.path.join(dataset, name))
    
    def compile_DEM_list():
        files = []          
        with os.scandir(constants.DATA_PATH_DEMS) as entries:
            for entry in entries:
                if entry.is_file():
                    files.append(entry.name) 

        with open(constants.DATA_PATH_DEM_LIST, 'w') as file:
            for name in files:
                file.write(f"{name}\n")

        return files

    def open_DEM_list(compile_list=False):
        if compile_list or not os.path.exists(constants.DATA_PATH_DEM_LIST):
            return DataAccessor.compile_DEM_list()

        dem_list = []
        with open(constants.DATA_PATH_DEM_LIST, 'r') as file:
            for line in file:
                dem_list.append(line.strip())
        
        return dem_list    
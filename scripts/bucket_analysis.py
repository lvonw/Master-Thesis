import numpy as np
import matplotlib.pyplot as plt
from    osgeo   import gdal
from    tqdm    import tqdm
import os


PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH_MASTER        = os.path.join(PROJECT_PATH, "data")
DATA_PATH_DEM           = os.path.join(DATA_PATH_MASTER, "DEMs")
DATA_PATH_DEM_LIST      = os.path.join(DATA_PATH_DEM, "SRTM_GL1_list.txt")

DATA_PATH_SOURCE_DEMs   = os.path.join(DATA_PATH_DEM, "SRTM_GL1_64x64")

# refer to dem_analytics for these values
MIN = -1503
MAX = 8092

def plot_histogram_from_dems(dem_names, min_value, max_value):
    pixel_buckets = np.zeros(max_value - min_value)

    for dem_name in tqdm(dem_names, desc="Going over all dems"):
        dem_file = os.path.join(DATA_PATH_SOURCE_DEMs, dem_name)
        dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
        
        if dataset is None:
            print(f"Failed to open {dem_name}")
            continue
        
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray().flatten()

        for value in array:
            pixel_buckets[int(value-min_value) - 1 ] += 1
        
    # plt.plot(pixel_buckets)
    # plt.title("Histogram of Pixel Values")
    # plt.xlabel("Pixel Value")
    # plt.ylabel("Frequency")    
    # plt.show()

    print (f"mean {np.mean(pixel_buckets) - min_value}")
    print (f"std {np.std(pixel_buckets) - min_value}")
    print (f"median {np.median(pixel_buckets) - min_value}")


def open_DEM_list():
        dem_list = []
        with open(DATA_PATH_DEM_LIST, 'r') as file:
            for line in file:
                dem_list.append(line.strip())
        
        return dem_list 


if __name__ == "__main__":
    dems = open_DEM_list()

    plot_histogram_from_dems(dems, MIN, MAX)

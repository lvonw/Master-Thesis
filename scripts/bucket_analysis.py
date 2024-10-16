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
MIN = 0 # -1503
MAX = 16 # 8092

MIN_RANGE = 0
MAX_RANGE = 7879

SIGMA = 4
PERCENTILE = 0.50
SHOW_HIST = True

def plot_histogram_from_dems(dem_names, min_value, max_value):
    pixel_buckets = np.zeros((max_value - min_value) + 1)
    range_buckets = np.zeros((MAX_RANGE - MIN_RANGE) + 1)

    # for dem_name in tqdm(dem_names, desc="Going over all dems"):
    #     dem_file = os.path.join(DATA_PATH_SOURCE_DEMs, dem_name)
    #     dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
        
    #     if dataset is None:
    #         print(f"Failed to open {dem_name}")
    #         continue


    dem_file    = os.path.join(DATA_PATH_SOURCE_DEMs, 
                               DATA_PATH_MASTER,
                               "GTC",
                                "Iwahashi_etal_2018",
                                "3600x1800_GlobalTerrainClassification_Iwahashi_etal_2018.tif")
    dataset     = gdal.Open(dem_file, gdal.GA_ReadOnly)

    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray().flatten()
    
    for value in array:
        pixel_buckets[int(value-min_value)] += 1

    # pixel_range = int(np.max(array) - np.min(array))
    # range_buckets[pixel_range] += 1
    
    if SHOW_HIST:
        plt.plot(pixel_buckets)
        plt.title("Histogram of Pixel Values")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")    
        plt.show()

    cum_buckets = np.cumsum(pixel_buckets)
    total_pixels = cum_buckets[-1]


    # cum_ranges = np.cumsum(range_buckets)
    # total_range = cum_ranges[-1]

    # plt.plot(cum_buckets)
    # plt.title("Histogram of Pixel Values")
    # plt.xlabel("Pixel Value")
    # plt.ylabel("Frequency")    
    # plt.show()


    # plt.plot(cum_ranges)
    # plt.title("Cumulative Ranges")
    # plt.xlabel("Range")
    # plt.ylabel("Frequency")    
    # plt.show()

    # lower_percentile = (1 - PERCENTILE) / 2
    # upper_percentile = 1 - lower_percentile

    # lower_bound = np.searchsorted(
    #     cum_buckets, total_pixels * lower_percentile) + min_value
    # upper_bound = np.searchsorted(
    #     cum_buckets, total_pixels * upper_percentile) + min_value

    # print (f"{PERCENTILE*100}% pixel range: {lower_bound} - {upper_bound}")

    # lower_bound = np.searchsorted(
    #     cum_ranges, total_range * lower_percentile) + MIN_RANGE
    # upper_bound = np.searchsorted(
    #     cum_ranges, total_range * upper_percentile) + MIN_RANGE
    
    # print (f"{PERCENTILE}% range range: {lower_bound} - {upper_bound}")

def open_DEM_list():
    dem_list = []
    with open(DATA_PATH_DEM_LIST, 'r') as file:
        for line in file:
            dem_list.append(line.strip())
    
    return dem_list 


if __name__ == "__main__":
    dems = open_DEM_list()

    plot_histogram_from_dems(dems, MIN, MAX)

import numpy as np
import matplotlib.pyplot as plt

from    osgeo   import gdal
from    tqdm    import tqdm
import os


PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH_MASTER        = os.path.join(PROJECT_PATH, "data", "datasets")
DATA_PATH_DEM           = os.path.join(DATA_PATH_MASTER, "DEMs")


DATA_PATH_DEM_LIST      = os.path.join(DATA_PATH_DEM, "SRTM_GL1_list.txt")

DATA_PATH_SOURCE_DEMs   = os.path.join(DATA_PATH_DEM, "SRTM_GL1_256x256")
#DATA_PATH_SOURCE_DEMs   = os.path.join(PROJECT_PATH, "data", "SRTM Master", "SRTM_GL1_srtm")

# refer to dem_analytics for these values
MIN = -12269 # -1503
MAX = 22894 # 8092

MIN_RANGE = 0
MAX_RANGE = 7879

SIGMA = 4
PERCENTILE = 0.50
SHOW_HIST = True

def plot_histogram_from_dems(dem_names, min_value, max_value):
    pixel_buckets = np.zeros((max_value - min_value) + 1)
    range_buckets = np.zeros((MAX_RANGE - MIN_RANGE) + 1)

    for dem_name in tqdm(dem_names, desc="Going over all dems"):
        dem_file = os.path.join(DATA_PATH_SOURCE_DEMs, dem_name)
        dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
        
        if dataset is None:
            print(f"Failed to open {dem_name}")
            continue

        # dataset     = gdal.Open(dem_file, gdal.GA_ReadOnly)

        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray().flatten()
        
        for value in array:
            pixel_buckets[int(value-min_value)] += 1

        # break

        # pixel_range = int(np.max(array) - np.min(array))
        # range_buckets[pixel_range] += 1
    
    x = np.arange(7000, 8092)

    
    pixel_buckets = pixel_buckets[(-MIN) + 7000: -MIN + 8092:]

    if SHOW_HIST:
        plt.plot(x, pixel_buckets / pixel_buckets.sum())
        plt.xlabel("Höhenwert")
        plt.ylabel("Anteil")    
        plt.xlim(left=7000) 
        plt.show()

    cum_buckets = np.cumsum(pixel_buckets)
    total_pixels = cum_buckets[-1]


    # cum_ranges = np.cumsum(range_buckets)
    # total_range = cum_ranges[-1]

    plt.plot(x, cum_buckets / pixel_buckets.sum())
    plt.xlabel("Höhenwert")
    plt.ylabel("Kumul. Anteil") 
    plt.xlim(left=7000) 
    plt.show()


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


DATA_PATH_MASTER        = os.path.join(PROJECT_PATH, "data", "datasets")
# DATA_PATH = os.path.join(DATA_PATH_MASTER, 
#                          "Climate", 
#                          "peel-et-al_2007",
#                          "koeppen_wgs84_0point1deg.txt.asc")

DATA_PATH = os.path.join(DATA_PATH_MASTER, 
                         "GTC", 
                         "Iwahashi_etal_2018",
                         "3600x1800_GlobalTerrainClassification_Iwahashi_etal_2018.tif")

def plot_buckets():
    pixel_buckets = np.zeros(16)

    file = os.path.join(DATA_PATH)
    dataset = gdal.Open(file, gdal.GA_ReadOnly)

    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray().flatten()
    
    for value in tqdm(array, desc="Analyzing Dataset"):
        if value < 1:
            pixel_buckets[0] += 1
            continue
        pixel_buckets[value] += 1

    pixel_buckets = pixel_buckets[1::] 

    pixel_buckets = pixel_buckets / pixel_buckets.sum()
    for value in pixel_buckets:
        print (f"{value*100:.2f}")

    return 

if __name__ == "__main__":
    # dems = open_DEM_list()
    # plot_histogram_from_dems(dems, MIN, MAX)

    plot_buckets()
    # dataset     = gdal.Open(dem_file, gdal.GA_ReadOnly)

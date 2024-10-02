import  os
import  matplotlib.pyplot   as plt
import  numpy               as np

from osgeo                  import gdal
from tqdm                   import tqdm

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH_MASTER        = os.path.join(PROJECT_PATH, "data")
DATA_PATH_DEM           = os.path.join(DATA_PATH_MASTER, "DEMs")
DEM_LIST_PREFIX         = "SRTM_GL1_"
DEM_LIST_POSTFIX         = "_list.txt" 

# Determines which dataset we are analysing
DATA_PATH_SOURCE_DEMs   = os.path.join(DATA_PATH_DEM, "SRTM_GL1_64x64")
DATA_PATH_DEM_LIST      = os.path.join(DATA_PATH_DEM, "SRTM_GL1_list.txt")

PLOT_SIGMA              = False
PLOT_RANGES             = False

PRINT_SIGMA_LIST        = False
PRINT_RANGE_LIST        = True

SEA_LEVEL               = 5
SIGMA_THRESHOLD         = 90

RANGE_MIN               = 0 #-2
RANGE_MAX               = 3213 #4993 

def analyze_DEM(dem_path):
    dataset = gdal.Open(dem_path)
    if not dataset or dataset.RasterCount != 1:
        print(f"Error opening file: {dem_path}")
        return None
    
    analysis            = {}
    analysis["file"]    = os.path.basename(dem_path)
    
    band            = dataset.GetRasterBand(1)
    band_array      = band.ReadAsArray()
    nodata_value    = band.GetNoDataValue()
    
    stats = band.GetStatistics(True, True)
    analysis["min"]             = stats[0]
    analysis["max"]             = stats[1]
    analysis["mean"]            = stats[2]
    analysis["std_dev"]         = stats[3]
    analysis["range"]           = analysis["max"] - analysis["min"]
    analysis["nodata_value"]    = nodata_value
    analysis["median"]          = np.median(band_array)

    nodata_count = 0
    if nodata_value is not None:
        nodata_count = np.sum(band_array == nodata_value)

    analysis["nodata_percentage"] = (nodata_count / band_array.size) * 100
    
    return analysis

def analyze_DEMs(dems):
    global_min              = np.iinfo(np.int64).max
    global_max              = np.iinfo(np.int64).min

    individual_metrics      = []
    all_nodata_percentages  = []
    all_means               = []
    all_medians             = []
    all_mins                = []
    all_maxs                = []
    all_stds                = []
    all_ranges              = []
    all_files_within_range  = []
    all_affected_by_range   = []
    all_unaffected_by_range = []
    all_excluded_by_range   = []

    negative_mean           = 0
    negative_median         = 0
    point3_sigma_negative   = 0
    one_sigma_negative      = 0
    sigma_over_threshold    = 0
    
    all_negative_mean_means         = []
    all_negative_mean_median_mean   = []
    all_negative_mean_stds          = []

    min_sigma_over_threshold        = float("inf")
    min_sigma_file                  = ""
    all_min_sigma_files             = []


    for dem_name in tqdm(dems,
                         total=len(dems),
                         desc="Analysing DEMs"):
        
        dem_file = os.path.join(DATA_PATH_SOURCE_DEMs, dem_name)
        
        metric = analyze_DEM(dem_file) 
        if not metric:
            continue
        
        individual_metrics.append(metric)
        
        global_min = min(global_min, metric["min"])
        global_max = max(global_max, metric["max"])
        
        all_nodata_percentages.append(metric["nodata_percentage"])
        all_means.append(metric["mean"])
        all_medians.append(metric["median"])
        all_mins.append(metric["min"])
        all_maxs.append(metric["max"])
        all_stds.append(metric["std_dev"])
        all_ranges.append(metric["range"])

        if metric["median"] < SEA_LEVEL: 
            negative_median += 1
        if metric["mean"] < SEA_LEVEL: 
            negative_mean += 1
            all_negative_mean_means.append(metric["mean"])
            all_negative_mean_stds.append(metric["std_dev"])
            all_negative_mean_median_mean.append(metric["median"])

        if metric["mean"] - 0.43 * metric["std_dev"] < SEA_LEVEL: 
            point3_sigma_negative += 1
        if metric["mean"] - metric["std_dev"] < SEA_LEVEL: 
            one_sigma_negative += 1

        if metric["std_dev"] > SIGMA_THRESHOLD: 
            sigma_over_threshold += 1
            all_min_sigma_files.append(metric["file"])
            
            if min_sigma_over_threshold > metric["std_dev"]:
                min_sigma_over_threshold = metric["std_dev"]
                min_sigma_file = metric["file"]

        if metric["max"] >= RANGE_MIN and metric["min"] <= RANGE_MAX:
            all_files_within_range.append(metric["file"])
        else:
            all_excluded_by_range.append(metric["file"])
        
        # if (metric["max"] > RANGE_MAX) != (metric["min"] < RANGE_MIN):
        #     all_affected_by_range.append(metric["file"])
        if metric["max"] <= RANGE_MAX and metric["min"] >= RANGE_MIN:
            all_unaffected_by_range.append(metric["file"])
        


    mean_min    = np.mean(all_mins)
    std_min     = np.std(all_mins)       
    mean_max    = np.mean(all_maxs)
    std_max     = np.std(all_maxs)       

    aggregate_metrics = {
        "global_min": global_min,
        "global_max": global_max,
        "mean_nodata_percentage": np.mean(all_nodata_percentages),
        "std_deviation_nodata_percentage": np.std(all_nodata_percentages),
        "mean_mean": np.mean(all_means),
        "std_deviation_of_mean": np.std(all_means),
        "mean_std_dev": np.mean(all_stds),
        "std_std_dev": np.std(all_stds),
        "mean_median": np.mean(all_medians),
        "std_deviation_of_medians": np.std(all_medians),
        "mean_min": mean_min,
        "mean_max": mean_max,
        "2_sigma_min": mean_min - 2 * std_min,
        "2_sigma_max": mean_max + 2 * std_max,
        "3_sigma_max": mean_max + 3 * std_max,
        "4_sigma_max": mean_max + 4 * std_max,
        "negative_mean": negative_mean,
        "negative_median": negative_median,
        "point3_sigma_negative": point3_sigma_negative,
        "one_sigma_negative": one_sigma_negative,
        "all_negative_mean_means_mean": np.mean(all_negative_mean_means),
        "all_negative_mean_stds_mean": np.mean(all_negative_mean_stds),
        "all_negative_mean_median_mean": np.mean(all_negative_mean_median_mean),
        "sigma_over_threshold": sigma_over_threshold,
        "min_sigma_file": min_sigma_file,
        "min_sigma_over_threshold": min_sigma_over_threshold,
        "mean_range": np.mean(all_ranges),
        "min_range": np.min(all_ranges),
        "max_range": np.max(all_ranges),
        "amount_in_range": len(all_files_within_range),
        #"all_affected_by_range": len(all_affected_by_range),
        "all_unaffected_by_range": len(all_unaffected_by_range),
        "all_excluded_by_range": len(all_excluded_by_range),
    }

    if PLOT_SIGMA:
        plt.plot(all_stds)
        plt.title("Sigma of each DEM")
        plt.xlabel("Image IDX")
        plt.ylabel("Sigma")    
        plt.show()

    if PLOT_RANGES:
        plt.plot(all_ranges)
        plt.title("Ranges of each DEM")
        plt.xlabel("Image IDX")
        plt.ylabel("Range")    
        plt.show()    

    if PRINT_SIGMA_LIST:
        dem_list_path = os.path.join(DATA_PATH_DEM, 
                                    (DEM_LIST_PREFIX 
                                     + str(SIGMA_THRESHOLD) 
                                     + DEM_LIST_POSTFIX))
        
        with open(dem_list_path, 'w') as file:
            for name in all_min_sigma_files:
                file.write(f"{name}\n")

    if PRINT_RANGE_LIST:
        dem_list_path = os.path.join(DATA_PATH_DEM, 
                                    (DEM_LIST_PREFIX 
                                     + str(RANGE_MIN)
                                     + "-"
                                     + str(RANGE_MAX)
                                     + DEM_LIST_POSTFIX))
        
        with open(dem_list_path, 'w') as file:
            for name in all_files_within_range:
                file.write(f"{name}\n")
    
    return individual_metrics, aggregate_metrics

def open_DEM_list():
        dem_list = []
        with open(DATA_PATH_DEM_LIST, 'r') as file:
            for line in file:
                dem_list.append(line.strip())
        
        return dem_list 

if __name__ == "__main__":
    dems = open_DEM_list()
    
    individual_metrics, aggregate_metrics = analyze_DEMs(dems)

    print(f"Aggregate Metrics for {os.path.basename(DATA_PATH_SOURCE_DEMs)}")
    print(f"List: {os.path.basename(DATA_PATH_DEM_LIST)}")

    for key, value in aggregate_metrics.items():
        print(f"{key}: {value}")
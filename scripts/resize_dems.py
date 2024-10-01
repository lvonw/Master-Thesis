import  os
from    osgeo   import gdal
from    tqdm    import tqdm

NEW_WIDTH   = 64
NEW_HEIGHT  = 64

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH_MASTER        = os.path.join(PROJECT_PATH, "data")
DATA_PATH_DEM           = os.path.join(DATA_PATH_MASTER, "DEMs")
DATA_PATH_SOURCE_DEMs   = os.path.join(DATA_PATH_DEM, "SRTM_GL1_srtm")
DATA_PATH_DEM_LIST      = os.path.join(DATA_PATH_DEM, "SRTM_GL1_list.txt")
DEM_FOLDER_PREFIX       = "SRTM_GL1"  


def resize_DEM(input_dem, output_dem, new_width, new_height):
    dataset = gdal.Open(input_dem)

    if not dataset:
        print(f"Error opening file: {input_dem}")
        return False

    geo_transform   = dataset.GetGeoTransform()
    projection      = dataset.GetProjection()

    original_width  = dataset.RasterXSize
    original_height = dataset.RasterYSize
    band_count      = dataset.RasterCount

    driver = gdal.GetDriverByName('GTiff')
    resized_dataset = driver.Create(output_dem, 
                                    new_width, 
                                    new_height, 
                                    band_count, 
                                    gdal.GDT_Float32)

    scale_x = original_width    / new_width
    scale_y = original_height   / new_height
    new_geo_transform = (geo_transform[0],
                         geo_transform[1] * scale_x,
                         geo_transform[2],
                         geo_transform[3],
                         geo_transform[4],
                         geo_transform[5] * scale_y)

    resized_dataset.SetGeoTransform(new_geo_transform)
    resized_dataset.SetProjection(projection)

    for band_idx in range(1, band_count + 1):
        band = dataset.GetRasterBand(band_idx)
        data = band.ReadAsArray(buf_xsize=new_width, 
                                buf_ysize=new_height,
                                resample_alg=gdal.GRIORA_Bilinear)
        resized_dataset.GetRasterBand(band_idx).WriteArray(data)

    resized_dataset.FlushCache()
    resized_dataset = None
    dataset = None

    return True


def process_DEMs(dems, output_base_folder, new_width, new_height):
    resolution_folder = os.path.join(
        output_base_folder,
        f"{DEM_FOLDER_PREFIX}_{new_width}x{new_height}")
    if not os.path.exists(resolution_folder):
        os.makedirs(resolution_folder)

    for dem in tqdm(dems, total=len(dems), desc="Resizing DEMs"):
        filename = os.path.basename(dem)
        input_file = os.path.join(DATA_PATH_SOURCE_DEMs, filename)
        output_file = os.path.join(resolution_folder, filename)

        if not resize_DEM(input_file, output_file, new_width, new_height):
            print ("Aborting")
            return
        

def open_DEM_list():
        dem_list = []
        with open(DATA_PATH_DEM_LIST, 'r') as file:
            for line in file:
                dem_list.append(line.strip())
        
        return dem_list 

if __name__ == "__main__":
    dems = open_DEM_list()

    process_DEMs(dems, DATA_PATH_DEM, NEW_WIDTH, NEW_HEIGHT)
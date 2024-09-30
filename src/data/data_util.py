import constants

import matplotlib.pyplot    as plt
import numpy                as np

from data.data_access       import DataAccessor
from mpl_toolkits.mplot3d   import Axes3D
from osgeo                  import gdal


class GeoUtil():
    def cell_to_geo_coordinates(geo_transform, x, y):
        x_origin        = geo_transform[0]
        pixel_width     = geo_transform[1]
        rotation_x      = geo_transform[2]
        y_origin        = geo_transform[3]
        rotation_y      = geo_transform[4]
        pixel_height    = geo_transform[5]

        latitude        = x_origin + x * pixel_width  + y * rotation_x
        longitude       = y_origin + y * pixel_height + x * rotation_y 

        return latitude, longitude

    def geo_coordinates_to_cell(geo_transform, latitude, longitude):

        # The maps used are all north oriented so the rotation will always be 0
        # which simplifies the calculation
        if geo_transform[2] != 0.0 or geo_transform[4] != 0.0:
            return (0, 0)

        x_origin        = geo_transform[0]
        pixel_width     = geo_transform[1]
        y_origin        = geo_transform[3]
        pixel_height    = geo_transform[5]

        x = (latitude  - x_origin) / pixel_width
        y = (longitude - y_origin) / pixel_height
        
        return int(x), int(y)

    def get_normalized_raster_band(raster_band,
                                   nodata_behaviour = constants.NoDataBehaviour.LOCAL_MINIMUM, 
                                   global_min = None, 
                                   global_max = None):
        
        band_array      = raster_band.ReadAsArray()
        nodata_value    = raster_band.GetNoDataValue()
        
        local_min = np.min(band_array)
        local_max = np.max(band_array)

        global_min = local_min if global_min is None else global_min
        global_max = local_max if global_max is None else global_max

        if nodata_value is not None:
            match nodata_behaviour:
                case constants.NoDataBehaviour.LOCAL_MINIMUM:
                    band_array[band_array == nodata_value] = local_min 
                case constants.NoDataBehaviour.GLOBAL_MINIMUM:
                    band_array[band_array == nodata_value] = global_min

        band_array = band_array - global_min
        band_array = band_array.astype(np.float32) 
        band_array = band_array / np.float32(global_max - global_min)

        return band_array
    
    def get_min_max(DEM_list):
        min = np.iinfo(np.int64).max
        max = np.iinfo(np.int64).min

        for dem in DEM_list:
            d = DataAccessor.open_DEM(dem)

            a = d.GetRasterBand(1).ReadAsArray()
            n = d.GetRasterBand(1).GetNoDataValue()

            max_v = np.max(a)

            if n is not None:
                a[a == n] = max_v
            
            min_v = np.min(a)
           
            if min > min_v:
                min = min_v
            if max < max_v:
                max = max_v
        
        return min, max
    
    def get_geo_frame_coordinates(geo_transform, top_left, bottom_right):
        """ Expecting the coordinates in x,y """
        top_left_geo    = GeoUtil.cell_to_geo_coordinates(
            geo_transform, 
            top_left[0], 
            top_left[1])
        bot_right_geo   = GeoUtil.cell_to_geo_coordinates(
            geo_transform, 
            bottom_right[0], 
            bottom_right[1])
        
        return top_left_geo, bot_right_geo

    def get_geo_frame_array(geo_array, 
                            geo_transform, 
                            top_left_geo, 
                            bot_right_geo):
        """ Expecting the coordinates in lat, long """
        top_left_cell   = GeoUtil.geo_coordinates_to_cell(
            geo_transform, 
            top_left_geo[0],
            top_left_geo[1]) 
        
        bot_right_cell  = GeoUtil.geo_coordinates_to_cell(
            geo_transform,
            bot_right_geo[0],
            bot_right_geo[1])
        
        # Extract the data frame
        data_frame = geo_array[top_left_cell[1] : bot_right_cell[1] + 1, 
                               top_left_cell[0] : bot_right_cell[0] + 1]

        return data_frame
        

class DataVisualizer():
    def show_geo_dataset_2D(dataset):
        dataset_array = dataset.GetRasterBand(1).ReadAsArray()

        plt.figure(figsize=(10, 10))
        plt.imshow(dataset_array)
        
        plt.title('Raster Image')
        plt.xlabel('Column (x)')
        plt.ylabel('Row (y)')
        plt.colorbar(label='Pixel Values')
        plt.show()

    def show_array(array):
        plt.figure(figsize=(10, 10))
        plt.imshow(array)
        
        plt.title('Raster Image')
        plt.xlabel('Column (x)')
        plt.ylabel('Row (y)')
        plt.colorbar(label='Pixel Values')
        plt.show()


    def show_dataset_3D(dataset):
        dataset_array = dataset.GetRasterBand(1).ReadAsArray()

        x = np.arange(dataset_array.shape[1])
        y = np.arange(dataset_array.shape[0])
        x, y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, dataset_array, cmap='viridis')

        plt.title('Raster Image')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plt.colorbar(label='Pixel Values')
        ax.set_zlim(-1000, +1000)
        plt.show()

    def show_image_tensor(tensor):
        if len(tensor.shape) == 4:
            image_tensor = tensor[0].permute(1, 2, 0)
        else:
            image_tensor = tensor.permute(1, 2, 0)

        image = image_tensor.numpy()
        
        plt.imshow(image)
        plt.title(f"Label")
        plt.axis('off')
        plt.show()

    def show_image_tensors(tensors):
        for tensor in tensors:
            DataVisualizer.show_image_tensor(tensor)
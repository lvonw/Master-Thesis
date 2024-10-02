import constants

import matplotlib.pyplot    as plt
import numpy                as np

from data.data_access       import DataAccessor
from mpl_toolkits.mplot3d   import Axes3D
from osgeo                  import gdal
from debug                  import Printer


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
                                   nodata_val = None,
                                   global_min = None, 
                                   global_max = None):
        local_min = None
        band_array = raster_band.ReadAsArray()
        nodata_value = (raster_band.GetNoDataValue() if nodata_val is None 
                        else nodata_val)


        if global_min is not None:
            band_array = np.clip(band_array, global_min, None)
            
            if np.min(band_array) < global_min:
                print(np.min(band_array))
        else:
            local_min = np.min(band_array)
            global_min = local_min

        if global_max is not None:
            band_array = np.clip(band_array, None, global_max)
        else:
            global_max = np.max(band_array)
        

        if nodata_value is not None:
            if nodata_behaviour == constants.NoDataBehaviour.LOCAL_MINIMUM:
                np.copyto(band_array, 
                          np.min(band_array) if local_min is None else local_min, 
                          where=(band_array == nodata_value))
            elif nodata_behaviour == constants.NoDataBehaviour.GLOBAL_MINIMUM:
                np.copyto(band_array, 
                          global_min, 
                          where=(band_array == nodata_value))

        band_array = (band_array - global_min).astype(np.float32) 
        band_array /= global_max - global_min

        # if np.min(band_array) < 0.0:
        #     print(np.min(band_array))

        # if np.max(band_array) > 1.0:
        #     print(np.max(band_array))

        return band_array
    
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
    def create_simple_plot(x, title):
        fig, ax = plt.subplots()
        ax.plot(x)
        ax.set_title(title)
        return fig
    
    def create_image_plot(image_data, 
                          title     = None,
                          xlabel    = None,
                          ylabel    = None, 
                          cmap      = "binary"):
        
        fig, ax = plt.subplots()
        cax = ax.imshow(image_data, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(cax)
        return fig

    def create_array_figure(array):
        if len(array.shape) == 1:
            return DataVisualizer.create_simple_plot(array)
        else:
            return DataVisualizer.create_image_plot(array)
        
    def create_geo_dataset_2D_figure(dataset):
        dataset_array = dataset.GetRasterBand(1).ReadAsArray()

        return DataVisualizer.create_array_figure(dataset_array)

    def create_geo_dataset_3D_figure(dataset):
        dataset_array = dataset.GetRasterBand(1).ReadAsArray()

        x = np.arange(dataset_array.shape[1])
        y = np.arange(dataset_array.shape[0])
        x, y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, dataset_array, cmap="binary")

        plt.title('Raster Image')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plt.colorbar(label='Pixel Values')
        ax.set_zlim(-1000, +1000)
        plt.show()

    def create_image_tensor_figure(tensor):
        tensor = tensor.to("cpu")
        if len(tensor.shape) == 4:
            image_tensor = tensor[0].permute(1, 2, 0)
        else:
            image_tensor = tensor.permute(1, 2, 0)

        image = image_tensor.numpy()
        
        DataVisualizer.create_image_plot(image)
        plt.show()

    def show_image_tensors(tensors):
        for tensor in tensors:
            DataVisualizer.show_image_tensor(tensor)

    def show_figures(figure_tuples, save_path=None):
        rows = len(figure_tuples)
        cols = max(len(tup) for tup in figure_tuples)

        fig, axes   = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        axes        = axes.reshape(rows, cols) 

        for row, figure_tuple in enumerate(figure_tuples):
            for col, figure in enumerate(figure_tuple):
                ax = axes[row, col] if rows > 1 else axes[col]

                for figure_ax in figure.axes:
                    if (isinstance(figure_ax.images, list) 
                        and len(figure_ax.images) > 0):
                        
                        im = figure_ax.images[0]
                        ax.imshow(im.get_array(), cmap=im.get_cmap())
                        
                    else:
                        for line in figure_ax.get_lines():
                            ax.plot(line.get_xdata(), line.get_ydata())
                        
                    ax.set_title(figure_ax.get_title())
        
        
        for row in range(rows):
            for col in range(len(figure_tuples[row]), cols):
                axes[row, col].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


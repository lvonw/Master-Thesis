import constants
import enum

import torch
import matplotlib.pyplot    as plt
import numpy                as np

from data.data_access       import DataAccessor
from mpl_toolkits.mplot3d   import Axes3D
from osgeo                  import gdal
from debug                  import Printer

class NormalizationMethod(enum.Enum):
    NONE                = 0
    LINEAR              = 1
    CLIPPED_LINEAR      = 2
    SIGMOID             = 3
    SIGMOID_COMBINATION = 4
    ASYMMETRIC_SIGMOID  = 5

class NoDataBehaviour(enum.Enum):
    NONE            = "None" 
    GLOBAL_MINIMUM  = "Global_Minimum"
    LOCAL_MINIMUM   = "Local_Minimum"


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
    
    def __rescale_sigmoid_image(image):
        image   = torch.logit(image)
        image   *= 1300
        image   += 2500
        return image
    
    def __sigmoid_image(image_tensor):
        image_tensor -= 2500
        image_tensor /= 1300
        return torch.sigmoid(image_tensor)
    
    def __asymmetric_sigmoid(x):
        a = 0.04
        b = 0.0012
        c = 0.06
        d = -50

        x = (d - torch.exp(-c * x)) * a + b * x

        return torch.sigmoid(x) 
    
    def __section_1 (x, phi):
        y = x - phi
        y /= 350.0
        y += GeoUtil.p2(phi)

        return y

    def __section_2 (x):
        y = x - 1000.0
        y /= 780.0
        y -= 0.8
        return y

    def __section_3 (x, lam):
        y = x - lam
        y /= 1800.0
        y += GeoUtil.p2(lam)
        return y    

    def __sigmoid_combination_2_sections(x):
        j = 3
        x = torch.where(x < 0, x/50 - j, x /900 - j)

        return torch.sigmoid(x) 
    
    def __sigmoid_combination_3_sections(x):
        phi = 600.0
        lam = 3213.0

        x = torch.where(x < phi, 
                        GeoUtil.__section_1(x, phi),
                        torch.where(x < lam, 
                                    GeoUtil.__section_2(x), 
                                    GeoUtil.__section_3(x, lam))) 
        return torch.sigmoid(x) 

    def __linear(x, min_value, max_value, clip=False):
        if min_value is None:
            min_value = np.min(x)

        if max_value is None:
            max_value = np.max(x)

        if clip:
            x = np.clip(x, min_value, max_value)

        x = (x - min_value).astype(np.float32) 
        x /= max_value - min_value
        return x
    

    def get_normalized_raster_band(
            raster_band,
            normalization_method    = NormalizationMethod.SIGMOID,
            nodata_behaviour        = NoDataBehaviour.NONE, 
            nodata_val = None,
            global_min = None, 
            global_max = None):

        band_array = raster_band.ReadAsArray()

        if NoDataBehaviour != NoDataBehaviour.NONE:
            nodata_value = (raster_band.GetNoDataValue() if nodata_val is None 
                            else nodata_val)
            
            if nodata_behaviour == constants.NoDataBehaviour.LOCAL_MINIMUM:
                global_min = np.min(band_array)
                np.copyto(band_array, 
                          global_min, 
                          where=(band_array == nodata_value))
            elif nodata_behaviour == constants.NoDataBehaviour.GLOBAL_MINIMUM:
                np.copyto(band_array, 
                          global_min, 
                          where=(band_array == nodata_value))
                
        band_tensor = torch.tensor(band_array, dtype=torch.float32)

        match normalization_method:
            case NormalizationMethod.NONE:
                pass 
            case NormalizationMethod.LINEAR:
                band_tensor = GeoUtil.__linear(band_tensor,
                                               global_min,
                                               global_max,
                                               False)
            case NormalizationMethod.CLIPPED_LINEAR:
                band_tensor = GeoUtil.__linear(band_tensor,
                                               global_min,
                                               global_max,
                                               True)
            case NormalizationMethod.SIGMOID:
                band_tensor = GeoUtil.__sigmoid_image(band_tensor)
            case NormalizationMethod.SIGMOID_COMBINATION:
                band_tensor = GeoUtil.__sigmoid_combination_3_sections(
                    band_tensor)
            case NormalizationMethod.ASYMMETRIC_SIGMOID:
                band_tensor = GeoUtil.__asymmetric_sigmoid(band_tensor)

        return band_tensor
        

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
                          cmap      = "gray"):
        
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
        ax.plot_surface(x, y, dataset_array, cmap="gray")

        plt.title('Raster Image')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plt.colorbar(label='Pixel Values')
        ax.set_zlim(-1000, +1000)
        plt.show()

    def create_image_tensor_figure(tensor, title="Image"):
        tensor = tensor.to("cpu")
        if len(tensor.shape) == 4:
            image_tensor = tensor[0].permute(1, 2, 0)
        else:
            image_tensor = tensor.permute(1, 2, 0)

        image = image_tensor.numpy()
        
        DataVisualizer.create_image_plot(image, title=title)
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


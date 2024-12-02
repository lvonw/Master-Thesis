import constants
import os

import torch
import matplotlib.pyplot    as plt
import numpy                as np

from data.data_access       import DataAccessor
from enum                   import Enum
from matplotlib.colors      import LightSource
from mpl_toolkits.mplot3d   import Axes3D
from osgeo                  import gdal
from debug                  import Printer

class NormalizationMethod(Enum):
    NONE                = 0
    LINEAR              = 1
    CLIPPED_LINEAR      = 2
    SIGMOID             = 3
    SIGMOID_COMBINATION = 4
    ASYMMETRIC_SIGMOID  = 5

class NoDataBehaviour(Enum):
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

        x = x - min_value
        x /= max_value - min_value
        return x
    

    def get_normalized_raster_band(
            raster_band,
            normalization_method    = NormalizationMethod.LINEAR,
            nodata_behaviour        = NoDataBehaviour.LOCAL_MINIMUM, 
            nodata_value            = None,
            global_min              = None, 
            global_max              = None,
            new_min                 = -1.,
            new_max                 = 1.):

        band_array = raster_band.ReadAsArray()

        if NoDataBehaviour != NoDataBehaviour.NONE:
            if nodata_value is None:  
                nodata_value = raster_band.GetNoDataValue()
            
            if nodata_behaviour == NoDataBehaviour.LOCAL_MINIMUM:
                global_min = np.min(band_array)
                np.copyto(band_array, 
                          global_min, 
                          where=(band_array == nodata_value))
            elif nodata_behaviour == NoDataBehaviour.GLOBAL_MINIMUM:
                np.copyto(band_array, 
                          global_min, 
                          where=(band_array == nodata_value))
        
        band_tensor = GeoUtil.get_normalized_array(band_array,
                                                   normalization_method,
                                                   global_min,
                                                   global_max,
                                                   new_min,
                                                   new_max)

        return band_tensor
        
    def get_normalized_array(
            array,
            normalization_method    = NormalizationMethod.LINEAR,
            global_min              = None, 
            global_max              = None,
            new_min                 = -1.,
            new_max                 = 1.):
        
        band_tensor = torch.tensor(array, dtype=torch.float32)

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

        range_mulitplier = new_max - new_min
        return band_tensor * range_mulitplier + new_min


class DataVisualizer():
    def __init__(self):
        self.plot_tuples = []

    def create_simple_plot(self, x, title):
        self.plot_tuples.append((Plot(data=x, title=title),))
    
    def create_image_plot(self,
                          image_data, 
                          title             = None,
                          x_label           = None,
                          y_label           = None, 
                          cmap              = "gray",
                          latent_space      = True,
                          append            = True,
                          three_dimensional = False):
        
        min_value, max_value = (-1., 1.) if latent_space else (None, None)

        plot_type = PlotType.PLANE if three_dimensional else PlotType.IMAGE

        image_plot = Plot(data      = image_data, 
                          title     = title,
                          x_label   = x_label,
                          y_label   = y_label,
                          cmap      = cmap,
                          min_value = min_value,
                          max_value = max_value,
                          plot_type = plot_type)
        
        if append:
            self.plot_tuples.append((image_plot,))
        else: 
            return image_plot
        

    def create_image_plot_tuple(self,
                          image_datas, 
                          title             = None,
                          x_label           = None,
                          y_label           = None,  
                          cmap              = "gray",
                          latent_space      = True, 
                          three_dimensional = False):
        
        image_plots = []
        for image_data in image_datas:
            image_plots.append(self.create_image_plot(
                image_data, 
                title               = title,
                x_label             = x_label,
                y_label             = y_label,
                cmap                = cmap,
                latent_space        = latent_space,
                append              = False,
                three_dimensional   = three_dimensional))
            
        self.plot_tuples.append(tuple(image_plots))
            
    def create_array_figure(self, array):
        if len(array.shape) == 1:
            return DataVisualizer.create_simple_plot(array)
        else:
            return DataVisualizer.create_image_plot(array)
        
    def create_geo_dataset_2D_figure(self, dataset):
        dataset_array = dataset.GetRasterBand(1).ReadAsArray()

        return DataVisualizer.create_array_figure(dataset_array)

    def create_array_from_tensor(self, tensor):
        if not isinstance(tensor, np.ndarray):
            tensor  = tensor.to("cpu")
            image   = tensor.numpy()
        else:
            image   = tensor

        if len(image.shape) == 4:
            image = image[0][0]
        else:
            image = image[0]

        return image
        

    def create_image_tensor_tuple(self, 
                                  tensors, 
                                  title=None, 
                                  latent_space      = False, 
                                  three_dimensional = False):
        images = []
        for tensor in tensors:
            images.append(self.create_array_from_tensor(tensor))
        
        self.create_image_plot_tuple(images, 
                                     title, 
                                     latent_space       = latent_space,
                                     three_dimensional  = three_dimensional)


    def create_3d_plot(self, data_tensor):
        data = self.create_array_from_tensor(data_tensor)

        x       = np.arange(data.shape[1])
        y       = np.arange(data.shape[0])
        x, y    = np.meshgrid(x, y)

        light_source    = LightSource(azdeg     = 270, 
                                      altdeg    = 45)
        shaded = light_source.shade(
            data, 
            cmap       = plt.get_cmap("gray"), 
            vert_exag  = 1, 
            blend_mode = "soft")

        fig = plt.figure(figsize=(10, 10))
        ax  = fig.add_subplot(111, projection="3d", proj_type="persp")

        ax.plot_surface(x, 
                        y, 
                        data, 
                        facecolors  = shaded,
                        cstride     = 1, 
                        rstride     = 1, 
                        antialiased = False, 
                        shade       = False,
                        linewidth   = 0,
                        edgecolor   = "none")

        plt.title("Raster Image")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Height")
        ax.set_zlim(-1, +1)
        plt.show()

    def show_ensemble(self, 
                      save=False, 
                      save_only=False,
                      clear_afterwards=True,
                      save_dir = "default",
                      filename = "",
                      model = None):
        
        amount_rows     = len(self.plot_tuples)
        amount_columns  = max(len(plots) for plots in self.plot_tuples)

        fig, axs = plt.subplots(nrows = amount_rows, 
                                ncols = amount_columns, 
                                figsize=(5*amount_columns, 5*amount_rows))
        
        # Adjust shape for consistency
        if amount_rows == 1 and amount_columns == 1:
            axs = np.array([[axs]])
        elif amount_rows == 1:
            axs = np.array([axs])
        elif amount_columns == 1:
            axs = np.array(axs)[:, np.newaxis]
        else:
            axs = np.array(axs) 

        for column, plot_tuple in enumerate(self.plot_tuples):
            for row, plot in enumerate(plot_tuple):
                if plot.data is None:
                    continue
                
                ax = axs[column, row]

                ax.set_title(plot.title)
                ax.set_xlabel(plot.x_label)
                ax.set_ylabel(plot.y_label)

                match plot.plot_type:
                    case PlotType.GRAPH_2D:
                        ax.plot(plot.data)
                        
                    case PlotType.BAR:
                        ax.bar(range(len(plot.data)), 
                               plot.data)
                        
                    case PlotType.IMAGE:
                        cax = ax.imshow(plot.data, 
                                        cmap = plot.cmap, 
                                        vmin = plot.min_value, 
                                        vmax = plot.max_value)
                        fig.colorbar(cax, ax = ax)

                    case PlotType.PLANE:
                        data    = np.flip(plot.data, 0)

                        x       = np.arange(data.shape[1])
                        y       = np.arange(data.shape[0])
                        x, y    = np.meshgrid(x, y)

                        light_source    = LightSource(azdeg     = 270, 
                                                      altdeg    = 45)
                        shaded = light_source.shade(
                            data, 
                            cmap       = plt.get_cmap("gray"), 
                            vert_exag  = 1, 
                            blend_mode = "soft")
                        
                        fig.delaxes(ax)
                        ax = fig.add_subplot(amount_rows, 
                                             amount_columns, 
                                             column * amount_columns + row + 1, 
                                             projection = "3d",
                                             proj_type  = "persp")
                        axs[column, row] = ax

                        ax.plot_surface(x, 
                                        y, 
                                        data, 
                                        facecolors  = shaded,
                                        cstride     = 1, 
                                        rstride     = 1, 
                                        antialiased = False, 
                                        shade       = False,
                                        linewidth   = 0,
                                        edgecolor   = "none")

                        ax.set_xlabel("X")
                        ax.set_ylabel("Y")
                        ax.set_zlabel("Height")
                        ax.set_zlim(-1, +1)

                        ax.view_init(elev=30, azim=-60)

        for row in range(amount_rows):
            for column in range(len(self.plot_tuples[row]), amount_columns):
                axs[row, column].axis("off")

        plt.tight_layout()
        
        if save or save_only:
            if model is not None:
                save_path = os.path.join(constants.LOG_PATH, 
                                         model.model_family,
                                         model.name,
                                         constants.LOG_IMAGES_FOLDER)    
            else:
                save_path = os.path.join(constants.IMAGE_LOG, save_dir)
                os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, filename + ".png")
            plt.savefig(save_file)
        
        if not save_only:
            plt.show()

        if clear_afterwards:
            self.plot_tuples.clear()
            plt.close()


class PlotType(Enum):
    GRAPH_2D    = "Graph_2D"
    BAR         = "Bar"
    IMAGE       = "Image"
    PLANE       = "Plane"

class Plot():
    def __init__(self,
                 data       = None,
                 title      = None,
                 x_label    = "X",
                 y_label    = "Y",
                 z_label    = "Z",
                 cmap       = "gray",
                 min_value  = None, 
                 max_value  = None,
                 plot_type  = PlotType.GRAPH_2D):
        
        self.title      = title
        self.data       = data
        self.x_label    = x_label
        self.y_label    = y_label
        self.z_label    = z_label
        self.cmap       = cmap
        self.min_value  = min_value
        self.max_value  = max_value
        self.plot_type  = plot_type


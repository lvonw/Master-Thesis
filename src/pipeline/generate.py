import constants
import os
import torch
import util

import numpy        as np

from datetime       import datetime
from enum           import Enum
from data.data_util import GeoUtil, DataVisualizer, NormalizationMethod
from PIL            import Image
from pipeline.grid  import GenerationGrid
from tqdm           import tqdm

import matplotlib.pyplot as plt

import torch
from torch.profiler import profile, ProfilerActivity

class MaskInterpolation(Enum):
    NONE                = "None"
    LINEAR              = "Linear"
    LEFT_COSINE         = "Left Cosine"
    RIGHT_COSINE        = "Right Cosine"
    LEFT_EXPONENTIAL    = "Left Exponential"
    RIGHT_EXPONENTIAL   = "Right Exponential"

def generate(model,
             amount_samples     = 4,
             iterations         = 8,
             input_image_path   = None,
             weight             = 0.8,
             save_only          = False,
             perlin_generator   = None):
     
    with torch.no_grad():

        data_visualizer = DataVisualizer()
        model = model.to(util.get_device())
        model.eval()

        input_tensor = None

        if input_image_path is not None:
            input_image_path    = os.path.join(
                constants.RESOURCE_PATH_TEST_IMAGES,
                input_image_path)
            input_image         = Image.open(input_image_path)
            input_array         = np.array(input_image).transpose(2, 0, 1)[0]

            input_tensor        = GeoUtil.get_normalized_array(
                input_array,
                NormalizationMethod.CLIPPED_LINEAR,
                0,
                255).unsqueeze(dim=0).unsqueeze(dim=0).to(util.get_device())


        grid = GenerationGrid((1,) + model.get_output_shape(), 
                              perlin_generator.get_minimum_overlap())
        

        alpha = __get_alpha(perlin_generator.get_minimum_overlap(),
                            MaskInterpolation.RIGHT_COSINE)

                
        for i in tqdm(range(iterations),
                      total     = iterations,
                      desc      = "Generating Samples",
                      position  = 0,
                      leave     = True,
                      colour    = "magenta",
                      disable   = False):
            
            coordinate = (i%3, i//3)

            if perlin_generator is not None:
                input_image     = perlin_generator.generate_image(coordinate)
                input_tensor    = (torch.tensor(input_image, 
                                                dtype = torch.float32)
                                   .unsqueeze(dim=0)
                                   .unsqueeze(dim=0)
                                   .to(util.get_device()))

                input_tensor    /= i + 1

            # label = i  #((i+1)*2)-1 
            # label = [[5, 1],[5, 5],[5, 12],[5, 28]]
            label = [[i + 1, i * 3]] #,[5, 28]] #,[10, 12],[15, 12]]
            
            # weight = 0.0
            # weight = 0.500
            # weight = 0.600
            weight = 0.700
            # weight = 0.750
            # weight = 0.775
            # weight = 0.800
            # weight = 0.825
            # weight = 0.850
            # weight = 0.875
            # weight = 0.900
            weight = 0.950
            # weight = 0.999

            mask, masked_image  = grid.get_mask_for_coordinate(coordinate, 
                                                               alpha)
            
            samples = model.generate(label, 
                                     amount_samples, 
                                     input_tensor       = input_tensor,
                                     img2img_strength   = weight,
                                     mask               = mask,
                                     masked_input       = masked_image,
                                     dynamic_device     = False,
                                     fast_cfg           = True)
                            
            final_image = grid.create_final_image(samples, 
                                                masked_image,
                                                mask)

            grid.insert(final_image[0], coordinate) 
                

        stitched_image = grid.stitch_image()
        
        data_visualizer.create_image_tensor_tuple([stitched_image],
                                                   title=str(label)) 

        time = datetime.now().strftime("%m-%d_%H-%M-%S")
        data_visualizer.show_ensemble(
            save        = True,
            filename    = f"infinite_{weight}_{time}",
            model       = model,
            save_only   = save_only)

    # Transform back to image space

def __get_alpha(overlap_length, 
                interpolation_type = MaskInterpolation.NONE):

    match interpolation_type:
        case MaskInterpolation.NONE:
            alpha = 1
        case MaskInterpolation.LINEAR:
            alpha = torch.linspace(1, 
                                   0, 
                                   steps=overlap_length)
        case MaskInterpolation.LEFT_COSINE:
            alpha = torch.linspace(torch.pi / 2, 
                                   0, 
                                   steps=overlap_length)
            alpha = torch.sin(alpha)
        case MaskInterpolation.RIGHT_COSINE:
            alpha = torch.linspace(0, 
                                   torch.pi / 2,
                                   steps=overlap_length)
            
            alpha = -torch.sin(alpha) + 1

        case MaskInterpolation.RIGHT_EXPONENTIAL:
            alpha = torch.linspace(0, 
                                   7, 
                                   steps=overlap_length)
            alpha = torch.pow(2, -alpha)

    return alpha
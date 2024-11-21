import constants
import os
import torch
import util

import numpy        as np

from datetime       import datetime
from debug          import Printer
from enum           import Enum
from data.data_util import GeoUtil, DataVisualizer, NormalizationMethod
from PIL            import Image
from pipeline.grid  import GenerationGrid
from tqdm           import tqdm

import matplotlib.pyplot as plt

class MaskInterpolation(Enum):
    NONE                = "None"
    LINEAR              = "Linear"
    LEFT_COSINE         = "Left Cosine"
    RIGHT_COSINE        = "Right Cosine"
    LEFT_EXPONENTIAL    = "Left Exponential"
    RIGHT_EXPONENTIAL   = "Right Exponential"

def new_generate(model, 
                 configuration  = None, 
                 title          = "asd",
                 save_only      = False):
    data_visualizer     = DataVisualizer()
    generated_results   = []
    
    if configuration is None or configuration["Unguided"]["active"]:
        generated_results.append(__generate_unguided())

    if configuration is not None and configuration["Image2Image"]["active"]:
        generated_results.append(__generate_sketch_based())

    if configuration is not None and configuration["Grid"]["active"]:
        generated_results.append(__generate_grid())

    if configuration is not None and configuration["Inpainting"]["active"]:
        generated_results.append(__generate_inpainting())

    data_visualizer.create_image_tensor_tuple(generated_results,
                                              title=str(title)) 

    time = datetime.now().strftime("%m-%d_%H-%M-%S")
    data_visualizer.show_ensemble(
        save        = True,
        filename    = f"infinite_{title}_{time}",
        model       = model,
        save_only   = save_only)

def __generate_unguided():
    pass    

def __generate_sketch_based():
    pass

def __generate_grid():
    pass

def __generate_inpainting():
    pass

def generate(model,
             amount_samples         = 4,
             iterations             = 8,
             input_image_path       = None,
             weight                 = 0.8,
             save_only              = False,
             perlin_generator       = None,
             regenerate_first_chunk = True):
     
    with torch.no_grad():

        data_visualizer = DataVisualizer()
        model           = model.to(util.get_device())
        model.eval()

        # Load Sketch if necessary ============================================
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

        # Grid ================================================================
        grid = GenerationGrid((1,) + model.get_output_shape(), 
                              perlin_generator.get_minimum_overlap())
        
        # Generate Mask Alphas ================================================
        alpha = __get_alpha(perlin_generator.get_minimum_overlap(),
                            MaskInterpolation.RIGHT_COSINE)


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
        # weight = 0.950
        # weight = 0.999

        # Generate all Samples ================================================
        for i in tqdm(range(iterations),
                      total     = iterations,
                      desc      = "Generating Samples",
                      position  = 0,
                      leave     = False,
                      colour    = "magenta",
                      disable   = False):
            
            coordinate = (i%3, i//3)
            # label = [[1, i*3]]
            label = [[5, 28]]

            __generate_chunk(model,
                             grid,
                             perlin_generator,
                             coordinate,
                             weight,
                             alpha,
                             label,
                             amount_samples)
            
        for i in tqdm(range(iterations),
                      total     = iterations,
                      desc      = "Generating Samples",
                      position  = 0,
                      leave     = False,
                      colour    = "magenta",
                      disable   = False):
            
            coordinate = (i%3, i//3)
            # label = [[1, i*3]]
            label = [[5, 28]]

            __generate_chunk(model,
                             grid,
                             perlin_generator,
                             coordinate,
                             weight,
                             alpha,
                             label,
                             amount_samples)

        # First chunk behaviour ===============================================
        # saved_stitched_image, stitched_sketch = grid.stitch_image()
        # if regenerate_first_chunk:
        #     Printer().print_log("Regenerating first Chunk")
        #     __generate_chunk(model,
        #                      grid,
        #                      perlin_generator,
        #                      (0, 0),
        #                      weight,
        #                      alpha,
        #                      [[1, 1]],
        #                      amount_samples)
        
        # Stitch final image ==================================================
        stitched_image, stitched_sketch = grid.stitch_image()
        
        data_visualizer.create_image_tensor_tuple([stitched_image,
                                                   stitched_sketch],
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

def __generate_chunk(model,
                     grid,
                     perlin_generator, 
                     coordinate,
                     weight,
                     alpha,
                     label,
                     amount_samples = 1):
    
    if perlin_generator is not None:
        # t = (coordinate[0]*10, coordinate[1]*12)
        perlin_image    = perlin_generator.generate_image(coordinate)
        perlin_tensor   = (torch.tensor(perlin_image, 
                                        dtype = torch.float32)
                           .unsqueeze(dim=0)
                           .unsqueeze(dim=0)
                           .to(util.get_device()))

        perlin_tensor   = perlin_tensor / 2 - 0.3
    
    mask, masked_image = grid.get_mask_for_coordinate(coordinate, 
                                                      alpha,
                                                      sketch = perlin_tensor)
    
    samples = model.generate(label, 
                             amount_samples, 
                             input_tensor       = perlin_tensor,
                             img2img_strength   = weight,
                             mask               = mask,
                             masked_input       = masked_image,
                             dynamic_device     = False,
                             fast_cfg           = True)
                                
    final_image = grid.create_final_image(samples, 
                                          masked_image,
                                          mask)

    grid.insert(final_image[0], coordinate, perlin_tensor[0])
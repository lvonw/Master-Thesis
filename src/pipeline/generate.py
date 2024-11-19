import constants
import os
import torch
import util

import numpy        as np

from enum           import Enum
from data.data_util import GeoUtil, DataVisualizer, NormalizationMethod
from PIL            import Image
from pipeline.grid  import GenerationGrid
from datetime       import datetime

import matplotlib.pyplot as plt

OVERLAP_AREA = 128 #64

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
             img2img            = False,
             input_image_path   = None,
             weight             = 0.8,
             save_only          = False,
             perlin_generator   = None):
    
    data_visualizer = DataVisualizer()
    model = model.to(util.get_device())
    model.eval()

    input_tensor = None

    if input_image_path is not None:
        input_image_path    = os.path.join(constants.RESOURCE_PATH_TEST_IMAGES,
                                           input_image_path)
        input_image         = Image.open(input_image_path)
        input_array         = np.array(input_image).transpose(2, 0, 1)[0]

        input_tensor        = GeoUtil.get_normalized_array(
            input_array,
            NormalizationMethod.CLIPPED_LINEAR,
            0,
            255).unsqueeze(dim=0).unsqueeze(dim=0).to(util.get_device())
        
    previous = None
    with torch.no_grad():
        for i in range(iterations):
            grid = GenerationGrid(1, 
                                  perlin_generator.get_minimum_overlap())

            if perlin_generator is not None:
                input_image = perlin_generator.generate_image((i, 0))
                input_tensor = (torch.tensor(input_image, dtype=torch.float32)
                                .unsqueeze(dim=0)
                                .unsqueeze(dim=0)
                                .to(util.get_device()))

            #label = i  #((i+1)*2)-1 
            # label = [[5, 1],[5, 5],[5, 12],[5, 28]]
            label = [[1, 12],[5, 28]] #,[10, 12],[15, 12]]
            
            # weight = 0.0
            # weight = 0.500
            # weight = 0.600
            # weight = 0.700
            # weight = 0.750
            # weight = 0.775
            # weight = 0.800
            # weight = 0.825
            # weight = 0.850
            # weight = 0.875
            # weight = 0.900
            weight = 0.950
            # weight = 0.999

            samples = model.generate(label, 
                                     amount_samples, 
                                     input_tensor,
                                     weight)
            

            previous = samples

            shape = 1
            mask = torch.zeros(1, + shape, 
                    dtype=torch.float32, 
                    device=util.get_device())

            mask = torch.zeros((1, 1, 256, 256), 
                               dtype=torch.float32, 
                               device=util.get_device())
            #mask[:, :, :, :64] = 1 

            # mask[:, :, :, :OVERLAP_AREA] = torch.linspace(
            #     1, 0, steps=OVERLAP_AREA)

            # mask_value = torch.linspace(
            #     torch.pi/2, torch.pi, steps=OVERLAP_AREA)
            # mask_value = torch.cos(mask_value) + 1

            save_only = True

            mask_value = torch.linspace(
                0, 7, steps=OVERLAP_AREA)
            mask_value = torch.pow(2, -mask_value)


            mask[:, :, :, :OVERLAP_AREA] = mask_value
            # print (mask)
        

            perlin = perlin_generator.generate_image((i+10, 0))
            perlin = (torch.tensor(perlin, dtype=torch.float32)
                        .unsqueeze(dim=0)
                        .unsqueeze(dim=0)
                        .to(util.get_device()))
            
            perlin /= 10
            # perlin -= .2



            masked_image = torch.where(mask > 0, previous, perlin)

            masked_image[:, :, :, :OVERLAP_AREA] = previous[:,:,:, -OVERLAP_AREA:]
            label = [[15, 28],[15, 28]]

            samples_2 = model.generate(label, 
                                     amount_samples, 
                                     input_tensor=perlin,
                                     img2img_strength=weight,
                                     mask=mask,
                                     masked_input=masked_image)
            
            samples_left = samples[:, :, :, :-OVERLAP_AREA]
            
            samples_left_overlap = samples[:, :, :, -OVERLAP_AREA:]
            samples_right_overlap = samples_2[:, :, :, :OVERLAP_AREA]

            samples_right = samples_2[:, :, :, OVERLAP_AREA:]

            # alpha = torch.linspace(
            #     1, 0, steps=OVERLAP_AREA).view(1, 1, 1, -1).to(perlin.device)
            
            # alpha = torch.linspace(
            #     torch.pi/2, torch.pi, steps=OVERLAP_AREA).view(1, 1, 1, -1).to(perlin.device)
            # alpha = torch.cos(alpha) + 1

            alpha = torch.linspace(
                0, 7, steps=OVERLAP_AREA).view(1, 1, 1, -1).to(perlin.device)
            alpha = torch.pow(2, -alpha)
            # alpha /= 10
            
            # print (alpha)


            samples_overlap = (
                alpha * samples_left_overlap 
                + (1 - alpha) * samples_right_overlap)
            
            
            samples_3 = torch.cat((samples, samples_right), dim=-1)
            
            samples_31 = torch.cat((samples_left, samples_2), dim=-1)

            samples_32 = torch.cat(
                (samples_left, samples_overlap, samples_right), dim=-1)


            # if input_tensor is not None:
                # samples_4 = [perlin, 
                #            masked_image[0].unsqueeze(0), 
                #            samples_3[0],
                #            samples_3[1]] 

            samples_4   = [samples_3[0],
                           samples_3[1]]
                
            samples_41  = [samples_31[0],
                           samples_31[1]] 
            
            samples_42  = [samples_32[0],
                           samples_32[1]]

            data_visualizer.create_image_tensor_tuple(samples_4, 
                                                      title=str(label)) 
            
            data_visualizer.create_image_tensor_tuple(samples_41, 
                                                      title=str(label)) 
            
            data_visualizer.create_image_tensor_tuple(samples_42, 
                                                      title=str(label)) 

            # data_visualizer.create_image_tensor_tuple(samples, 
            #                                           title=str(label)) 
            
            # data_visualizer.create_image_tensor_tuple(samples_2, 
            #                                           title=str(label)) 
            
            time = datetime.now().strftime("%m-%d-%H-%M-%S")
            data_visualizer.show_ensemble(
                save = True,
                filename = f"infinite_{weight}_{time}",
                model = model,
                save_only = save_only)

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
            alpha = torch.linspace(torch.pi / 2, 
                                   0, 
                                   steps=overlap_length)
            alpha = -torch.sin(alpha) + 1
        case MaskInterpolation.RIGHT_EXPONENTIAL:
            alpha = torch.linspace(0, 
                                   7, 
                                   steps=overlap_length)
            alpha = torch.pow(2, -alpha)

    return alpha

def __stitch_chunks(overlap_length,
                    ):
    pass
import constants
import os
import torch
import util

import numpy        as np

from data.data_util import GeoUtil, DataVisualizer, NormalizationMethod
from PIL            import Image
from datetime       import datetime

import matplotlib.pyplot as plt

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

            if perlin_generator is not None:
                input_image = perlin_generator.generate_image((i, 0))
                input_tensor = (torch.tensor(input_image, dtype=torch.float32)
                                .unsqueeze(dim=0)
                                .unsqueeze(dim=0)
                                .to(util.get_device()))

            #label = i  #((i+1)*2)-1 
            # label = [[5, 1],[5, 5],[5, 12],[5, 28]]
            label = [[1, 12],[5, 28]] #,[10, 12],[15, 12]]
            
            # thing = 0
            thing = 700
            # thing = 750
            # thing = 775
            # thing = 800
            # thing = 825
            # thing = 850
            # thing = 875
            # thing = 900
            # thing = 950
            # thing = 999

            samples = model.generate(label, 
                                     amount_samples, 
                                     input_tensor,
                                     thing)
            

            # TODO use perlin to infer overlapping area, slice overlapping
            # area from previous sample, and use it as a mask for diffusion,
            # still use the new perlin noise as sketch, stitch both old, and new
            # samples at the overlapping area for presentation

            previous = samples

            mask = torch.zeros((1, 1, 256, 256), 
                               dtype=torch.float32, 
                               device=util.get_device())
            mask[:, :, :, :64] = 1 
        

            perlin = perlin_generator.generate_image((i+3, 0))
            perlin = (torch.tensor(perlin, dtype=torch.float32)
                        .unsqueeze(dim=0)
                        .unsqueeze(dim=0)
                        .to(util.get_device()))


            masked_image = torch.where(mask > 0, previous, perlin)

            masked_image[:, :, :, :64] = previous[:,:,:, -64:]
            label = [[5, 28],[1, 28]]

            samples_2 = model.generate(label, 
                                     amount_samples, 
                                     input_tensor=perlin,
                                     img2img_strength=thing,
                                     mask=mask,
                                     masked_input=masked_image)
            
            samples_left = samples[:, :, :, :192]
            
            samples_left_overlap = samples[:, :, :, 192:]
            samples_right_overlap = samples_2[:, :, :, :-192]

            samples_right = samples_2[:, :, :, -192:]

            alpha = torch.linspace(
                1, 0, steps=64).view(1, 1, 1, -1).to(perlin.device)
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

            samples_4 = [samples_3[0],
                        samples_3[1]]
                
            samples_41 = [samples_31[0],
                          samples_31[1]] 
            
            samples_42 = [samples_32[0],
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
                filename = f"infinite_{thing}_{time}",
                model = model,
                save_only = save_only)

    # Transform back to image space

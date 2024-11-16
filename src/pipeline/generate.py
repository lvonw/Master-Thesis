import constants
import os
import torch
import util

import numpy        as np

from data.data_util import GeoUtil, DataVisualizer, NormalizationMethod
from PIL            import Image
from datetime       import datetime

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
            label = [[1, 12],[5, 12],[10, 12],[15, 12]]
            
            # thing = 0
            thing = 700
            # thing = 750
            # thing = 775
            # thing = 800
            # thing = 825
            # thing = 850
            # thing = 875
            # thing = 900

            samples = model.generate(label, 
                                     amount_samples, 
                                     input_tensor,
                                     thing)

            if input_tensor is not None:
                samples = torch.cat((input_tensor, samples), dim=0)

            data_visualizer.create_image_tensor_tuple(samples, 
                                                      title=str(label)) 
            
            time = datetime.now().strftime("%m-%d-%H-%M-%S")
            data_visualizer.show_ensemble(
                save = True,
                filename = f"cfg_{thing}_{time}",
                model = model,
                save_only = save_only)

    # Transform back to image space

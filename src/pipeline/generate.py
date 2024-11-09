import constants
import os
import torch
import util

import numpy        as np

from data.data_util import GeoUtil, DataVisualizer, NormalizationMethod
from PIL            import Image

def generate(model,
             amount_samples     = 4,
             iterations         = 8,
             input_image_path   = None,
             weight             = 0.8):
    
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
            label = i#((i+1)*2)-1 
            #label = [[2, 1],[2, 5],[2, 12],[2, 28]]

            # thing = 700
            # thing = 750
            # thing = 775
            thing = 800
            # thing = 825
            # thing = 875
            # thing = 900

            samples = model.generate(label, 
                                     amount_samples, 
                                     input_tensor,
                                     thing)

            if input_tensor is not None:
                samples = torch.cat((input_tensor, samples), dim=0)

            data_visualizer.create_image_tensor_tuple(samples, title=str(label)) 
            
            data_visualizer.show_ensemble(
                save=True,
                filename=f"cfg_{thing}")

    # Transform back to image space

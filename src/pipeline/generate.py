import constants
import os
import torch
import util

import numpy        as np

from data.data_util import GeoUtil, DataVisualizer, NormalizationMethod
from PIL            import Image

def generate(model,
             amount_samples     = 4,
             iterations         = 10,
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
            label = i #% 2 + 1
            
            samples = model.generate(i, amount_samples, input_tensor)
            samples = torch.cat((input_tensor, samples), dim=0)

            data_visualizer.create_image_tensor_tuple(samples, title=str(label)) 
            
            data_visualizer.show_ensemble()

    # Transform back to image space

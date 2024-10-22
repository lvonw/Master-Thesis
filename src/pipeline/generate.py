import torch
import util

from data.data_util import DataVisualizer


def generate(model):
    data_visualizer = DataVisualizer()
    model = model.to(util.get_device())
    model.eval()
    with torch.no_grad():
        for i in range(10):
            label = i #% 2 + 1
            sample = model.generate(label)
            data_visualizer.create_image_tensor_tuple(sample, title=str(label)) 
            
            if (i+1) % 2 == 0:
                data_visualizer.show_ensemble()

    # Transform back to image space



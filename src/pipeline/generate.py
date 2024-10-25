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
            
            samples = model.generate(None, 4)

            data_visualizer.create_image_tensor_tuple(samples, title=str(label)) 
            
            data_visualizer.show_ensemble()

    # Transform back to image space



import torch
import util

from data.data_util import DataVisualizer


def generate(model):
    model = model.to(util.get_device())
    model.eval()
    with torch.no_grad():
        for i in range(10):
            label = i % 2 + 1
            sample = model.generate(label)
            DataVisualizer.create_image_tensor_figure(sample, str(label))

    # Transform back to image space



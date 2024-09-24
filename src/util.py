import torch
import torch.nn         as nn
import os

from torch.utils.data   import random_split


def get_device(should_print=False):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if should_print:
        print(f"Setting device to {device}")

    return torch.device(device)


def make_path(path_string):
    path_arr = path_string.split("/")
    return os.path.join(*path_arr)

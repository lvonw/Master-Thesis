import constants
import os
import torch
import torch.nn         as nn

# TODO Need to make it so we dont constantly check this
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)


def make_path(path_string):
    path_arr = path_string.split("/")
    return os.path.join(*path_arr)

# from torch.utils.data   import random_split
# from generation.models  import vae, ddpm

def get_model_family(model):
    # if isinstance(model, vae.VariationalAutoEncoder):
    #     return "vae"
    # elif isinstance(model, ddpm.DDPM):
    #     return "ddpm"
    
    return "diffusion"

def get_model_file_path(model):
    model_family = get_model_family(model)
    return os.path.join(constants.MODEL_PATH_MASTER,
                        model_family,
                        model.name + constants.MODEL_FILE_TYPE)
    
def save_model(model):
    torch.save(model.state_dict(), get_model_file_path(model))

def load_model(model):
    model_path = get_model_file_path(model)
    if not os.path.exists(model_path):
        return False

    model.load_state_dict(torch.load(model_path, 
                                     weights_only=False))
    return True

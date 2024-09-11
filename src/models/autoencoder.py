import constants
import torch
import torch.nn             as nn
import torch.nn.functional  as f

from configuration import Section

class AutoEncoderFactory():
    def __init__(self):
        pass

    def create_auto_encoder(data_configuration: Section):
        pass

class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 

class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 


class VQVAE(nn.Module):
    pass

class KLVAE(nn.Module):
    pass
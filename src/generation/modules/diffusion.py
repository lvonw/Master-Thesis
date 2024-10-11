import constants
import torch
import torch.nn             as nn
import torch.nn.functional  as f

from generation.modules.unet import UNET


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()

    def forward(self, x, control_signal, timestep):
        timestep = self.get_time_embedding(timestep)
        timestep = self.time_embedding(timestep)

        x = self.unet(x, control_signal, timestep)

        return x
    
    def get_time_embedding(timestep):
        freqs = torch.pow(10000, 
                            -torch.arange(start=0, end=160, dtype=torch.float32)/160)
        x  = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    
    
class TimeEmbedding(nn.Module):
    def __init__(self, num_embedding):
        super().__init__()

        self.linear_1(num_embedding, 4 * num_embedding)
        self.linear_1(4 * num_embedding, 4 * num_embedding)

    def forward(self, x):
        x = self.linear_1(x)
        x = f.silu(x)
        x = self.linear_2(x)

        return x


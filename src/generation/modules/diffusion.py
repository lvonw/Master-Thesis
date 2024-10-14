import constants
import torch
import util

import torch.nn             as nn
import torch.nn.functional  as f

from generation.modules.unet import UNETFactory


class Diffusion(nn.Module):
    def __init__(self, configuration):
        super().__init__()

        self.time_embedding_size = configuration["time_embedding_size"]
        self.time_embedding = TimeEmbedding(self.time_embedding_size)
        self.unet = UNETFactory.create_unet(configuration)

    def forward(self, x, control_signal, timestep):
        timestep = self.get_time_embedding(timestep, self.time_embedding_size)
        timestep = self.time_embedding(timestep)

        x = self.unet(x, control_signal, timestep)

        return x
    
    def get_time_embedding(self, timesteps, embedding_size):
        embedding_size = embedding_size // 2
        freqs = torch.pow(10000, 
                          -torch.arange(start=0, 
                                        end=embedding_size, 
                                        dtype=torch.float32) / embedding_size,
                                        ).to(util.get_device())
        
        x = timesteps.clone().detach().to(dtype=torch.float32).unsqueeze(-1)
        x = x * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    
    
class TimeEmbedding(nn.Module):
    def __init__(self, num_embedding):
        super().__init__()

        self.linear_1 = nn.Linear(num_embedding, 4 * num_embedding)
        self.linear_2 = nn.Linear(4 * num_embedding, 4 * num_embedding)

    def forward(self, x):
        x = self.linear_1(x)
        x = f.silu(x)
        x = self.linear_2(x)

        return x


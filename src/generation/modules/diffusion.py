import constants
import torch
import util

import torch.nn             as nn
import torch.nn.functional  as f

from enum                       import Enum
from generation.modules.unet    import UNETFactory

class ControlSignalIntegration(Enum):
      NONE              = "None"
      ADD_TO_TIME       = "Add to Time"
      CROSS_ATTENTION   = "Cross Attention"


class Diffusion(nn.Module):
    def __init__(self, configuration, amount_classes):
        super().__init__()

        self.time_embedding_size = configuration["time_embedding_size"]
        self.time_embedding = TimeEmbedding(self.time_embedding_size)
        self.unet = UNETFactory.create_unet(configuration)

        if amount_classes:
            self.control_signal_integration = ControlSignalIntegration.ADD_TO_TIME
        else:
            self.control_signal_integration = ControlSignalIntegration.NONE
            
        self.label_embedding = LabelEmbedding(amount_classes, 
                                              self.time_embedding.output_dim)

    def forward(self, x, control_signal, timestep):
        timestep = self.get_time_embedding(timestep, self.time_embedding_size)
        timestep = self.time_embedding(timestep)

        if control_signal is not None:
            match self.control_signal_integration:
                case ControlSignalIntegration.ADD_TO_TIME:
                    timestep += self.label_embedding(control_signal)
                case ControlSignalIntegration.CROSS_ATTENTION:
                    control_signal = self.label_embedding(control_signal)
                case ControlSignalIntegration.NONE:
                    control_signal = None

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
    

class LabelEmbedding(nn.Module):
    def __init__(self, amount_classes, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(amount_classes, embedding_dim)
    
    def forward(self, x):
        x = self.embedding(x)

        return x
    
    
class TimeEmbedding(nn.Module):
    def __init__(self, num_embedding):
        super().__init__()

        self.output_dim =  4 * num_embedding

        self.linear_1 = nn.Linear(num_embedding, self.output_dim)
        self.linear_2 = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = f.silu(x)
        x = self.linear_2(x)

        return x


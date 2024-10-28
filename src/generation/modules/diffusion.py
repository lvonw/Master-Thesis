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

class LabelCombinationType(Enum):
    CONCAT      = "Concat"
    ADD         = "Add"
    MULTIPLY    = "Multiply"

class LabelEmbedding(nn.Module):
    """
    Expecting the labels to be in shape (batch, classes)
    """
    def __init__(self, 
                 amounts_classes, 
                 embedding_dim, 
                 combination_type = LabelCombinationType.ADD):
        super().__init__()

        self.embeddings         = nn.ModuleList()
        self.combination_type   = combination_type

        for amount in amounts_classes:
            self.embeddings.append(nn.Embedding(amount, embedding_dim))

    
    def forward(self, x):
        embeddings = []
        if len(x.shape) > 1: 
            for idx, embedding in enumerate(self.embeddings):
                embeddings.append(embedding(x[:, idx]))
        else: 
            embeddings.append(self.embeddings[0](x))

        match self.combination_type:
            case LabelCombinationType.CONCAT:
                # x = torch.cat(embeddings, dim=-1)
                pass
            case LabelCombinationType.ADD:
                x = torch.stack(embeddings).sum(dim=0)
            case LabelCombinationType.MULTIPLY:
                x = torch.stack(embeddings).prod(dim=0)

        return x
    
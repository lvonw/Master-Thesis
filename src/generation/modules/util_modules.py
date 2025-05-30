import constants
import math
import torch
import torch.nn             as nn
import torch.nn.functional  as f


class NonLinearity(nn.Module):
    def __init__(self):
        super().__init__()
        self.non_linearity = nn.SiLU()

    def forward(self, x):
        return self.non_linearity(x)

class Normalize(nn.Module):
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, 
                                 num_channels=num_channels, 
                                 eps=1e-6, 
                                 affine=True)

    def forward(self, x, *args, **kwargs):
        return self.norm(x)

class Downsample(nn.Module):
    def __init__(self, channels, asymmetric_padding=True):
        super().__init__()
        self.asymmetric_padding = asymmetric_padding

        if asymmetric_padding:    
            self.conv = nn.Conv2d(channels,
                                  channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=0)
        else:
            self.conv = nn.Conv2d(channels,
                                  channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=1)

    def forward(self, x):
        if self.asymmetric_padding:
            # Asymmetric padding so that we exactly half the dimensions
            x = f.pad(x, (0,1,0,1))

        x = self.conv(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, channels, with_conv = False, scale_factor=2):
        super().__init__()
        self.with_conv      = with_conv
        self.scale_factor   = scale_factor

        if self.with_conv:
            self.conv = torch.nn.Conv2d(channels,
                                        channels,
                                        kernel_size=3,
                                        padding=1)

    def forward(self, x):
        x = f.interpolate(x, scale_factor=self.scale_factor, mode="nearest")

        if self.with_conv:
            x = self.conv(x)

        return x

class ResNetBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 use_adagn = False, 
                 time_embedding_channels=1280):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, 
                                out_channels, 
                                kernel_size=3, 
                                padding=1)
        self.conv_2 = nn.Conv2d(out_channels, 
                                out_channels, 
                                kernel_size=3, 
                                padding=1)

        self.norm_1 = Normalize(in_channels, 32)
        
        self.use_adagn = use_adagn
        if use_adagn:
            self.norm_2 = AdaGN(out_channels, 
                               time_embedding_channels,
                               32)
        else:     
            self.norm_2 = Normalize(out_channels, 32)
        
            if time_embedding_channels > 0:
                self.time_linear = nn.Linear(time_embedding_channels,
                                            out_channels)
                   
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else: 
            self.residual = nn.Conv2d(in_channels, 
                                      out_channels, 
                                      kernel_size=1, 
                                      padding=0)
    
    def forward(self, x, time_embedding=None):
        residual_x = x
        
        x = self.norm_1(x)
        x = f.silu(x)
        x = self.conv_1(x)

        if not self.use_adagn and time_embedding is not None: 
            time_embedding = f.silu(time_embedding)
            time_embedding = self.time_linear(time_embedding)
            x = x + time_embedding[:, :, None, None]
            
        x = self.norm_2(x, time_embedding)
        x = f.silu(x)
        x = self.conv_2(x)

        return x + self.residual(residual_x) 

class AdaGN(nn.Module):
    """From Diffusion Models beat GANs"""
    def __init__(self, 
                 amount_channels,
                 embedding_channels, 
                 amount_groups=32):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups=amount_groups, 
                                 num_channels=amount_channels, 
                                 eps=1e-6, 
                                 affine=True)
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_channels, 2 * amount_channels))

    def forward(self, x, control_signal):
        gamma, beta = self.mlp(control_signal)[:, :, None, None].chunk(2, 
                                                                       dim=1)
        x = self.norm(x)
        return x * (1 + gamma) + beta
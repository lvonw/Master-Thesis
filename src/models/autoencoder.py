import constants
import torch
import torch.nn             as nn
import torch.nn.functional  as f

from configuration  import Section
from util_modules   import (ResNetBlock, 
                            Downsample, 
                            Upsample, 
                            AttentionBlock,
                            Normalize)

class AutoEncoderFactory():
    def __init__(self):
        pass

    def create_auto_encoder(data_configuration: Section):
        pass

class VAEEncoder(nn.Module):
    def __init__(self,
                 size,
                 amount):
        super().__init__()
        
        self.encoder = nn.ModuleList()

        # TODO Generalize
        # TODO somehow figure out how to configure generalizations
        #ch start 128
        self.encoder.append(nn.Conv2d(1, 128, kernel_size=3, padding=1))
        self.encoder.append(ResNetBlock(128, 128))
        self.encoder.append(ResNetBlock(128, 128))
        self.encoder.append(Downsample(128))

        #ch mult 2
        self.encoder.append(ResNetBlock(128, 256))
        self.encoder.append(ResNetBlock(256, 256))
        self.encoder.append(Downsample(256))

        #ch mult 2
        self.encoder.append(ResNetBlock(256, 512))
        self.encoder.append(ResNetBlock(512, 512))
        self.encoder.append(Downsample(512))

        # Bottleneck
        self.encoder.append(ResNetBlock(512, 512))
        self.encoder.append(ResNetBlock(512, 512))
        self.encoder.append(ResNetBlock(512, 512))

        self.encoder.append(AttentionBlock(512))
        
        self.encoder.append(ResNetBlock(512, 512))
        
        # ? why specifically 32
        self.encoder.append(Normalize(512, 32))

        # SD uses sigmoid, maybe i should too?
        self.encoder.append(nn.SiLU())

        # outchannels 8
        self.encoder.append(nn.Conv2d(512, 8, kernel_size=3, padding=1))
        self.encoder.append(nn.Conv2d(8, 8, kernel_size=1, padding=0))


    def forward(self, x, noise):
        for module in self.encoder:
            x = module(x)

        mu, log_variance = torch.chunk(x, 2, dim=1)
        
        sigma = torch.clamp(log_variance, -30, 20).exp().sqrt()

        # Reparameterization where noise ~ N(0,I)
        x = mu + sigma * noise
        # SD constant, idk why this is here, can try removing it
        x = x * 0.18215

class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.ModuleList()

        self.decoder.append(nn.Conv2d(4, 4, kernel_size=1, padding=1))

        self.decoder.append(nn.Conv2d(4, 512, kernel_size=3, padding=1))
        self.decoder.append(ResNetBlock(512, 512))
        self.decoder.append(AttentionBlock(512))

        self.decoder.append(ResNetBlock(512, 512))
        self.decoder.append(ResNetBlock(512, 512))
        self.decoder.append(ResNetBlock(512, 512))
        self.decoder.append(ResNetBlock(512, 512))
        
        self.decoder.append(Upsample(512))
        self.decoder.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.decoder.append(ResNetBlock(512, 512))
        self.decoder.append(ResNetBlock(512, 512))
        self.decoder.append(ResNetBlock(512, 512))
        
        self.decoder.append(Upsample(512))
        self.decoder.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.decoder.append(ResNetBlock(512, 256))
        self.decoder.append(ResNetBlock(256, 256))
        self.decoder.append(ResNetBlock(256, 256))

        self.decoder.append(Upsample(256))
        self.decoder.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.decoder.append(ResNetBlock(256, 128))
        self.decoder.append(ResNetBlock(128, 128))
        self.decoder.append(ResNetBlock(128, 128))
        
        self.decoder.append(Normalize(128, 32))
        self.decoder.append(nn.SiLU())

        self.decoder.append(nn.Conv2d(128, 1, kernel_size=3, padding=1))



    def forward(self, x):
        x = x / 0.18215

        for module in self.decoder:
            x = module(x)

        return x


class VQVAE(nn.Module):
    pass

class KLVAE(nn.Module):
    pass


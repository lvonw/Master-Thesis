import constants
import torch
import torch.nn             as nn
import torch.nn.functional  as f

from configuration  import Section
from util_modules   import (ResNetBlock, 
                            Downsample, 
                            Upsample, 
                            Normalize)
from attention      import AttentionBlock, ContextualAttentionBlock


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_Outputlayer(320, 4)

    def forward(self, x, control_signal, time):
        time = self.time_embedding(time)

        x = self.unet(x, control_signal, time)
        x = self.final(x)

        return x
    
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

class SwitchSequential(nn.Sequential):
    def forward(self, x, control_signal, time):
        for layer in self:
            if isinstance(layer, ContextualAttentionBlock):
                x = layer(x, control_signal)
            elif isinstance(layer, ResNetBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNET(nn.Module):
    def __init__(self, num_embedding):
        super().__init__()

        self.encoder = nn.ModuleList()

        #input conv
        self.encoder.append(SwitchSequential(
            nn.Conv2d(4, 320, kernel_size=3, padding=1)))

        # 2 res 2 attention
        self.encoder.append(SwitchSequential(ResNetBlock(320, 320), 
                                             ContextualAttentionBlock(8, 40)))
        self.encoder.append(SwitchSequential(ResNetBlock(320, 320), 
                                             ContextualAttentionBlock(8, 40)))

        # Downsample 1
        self.encoder.append(SwitchSequential(
            nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)))

        # 2 res 2 attention
        self.encoder.append(SwitchSequential(ResNetBlock(320, 640), 
                                             ContextualAttentionBlock(8, 80)))
        self.encoder.append(SwitchSequential(ResNetBlock(640, 640), 
                                             ContextualAttentionBlock(8, 80)))

        # Downsample 1
        self.encoder.append(SwitchSequential(
            nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)))

        # 3 res 3 attention
        self.encoder.append(SwitchSequential(ResNetBlock(640, 1280), 
                                             ContextualAttentionBlock(8, 160)))
        self.encoder.append(SwitchSequential(ResNetBlock(1280, 1280), 
                                             ContextualAttentionBlock(8, 160)))
        self.encoder.append(SwitchSequential(ResNetBlock(1280, 1280), 
                                             ContextualAttentionBlock(8, 160)))

        # Downsample 1
        self.encoder.append(SwitchSequential(
            nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)))
        
        # 2 Res
        self.encoder.append(SwitchSequential(ResNetBlock(1280, 1280)))
        self.encoder.append(SwitchSequential(ResNetBlock(1280, 1280)))


        # Res Cross Res
        self.bottleneck = SwitchSequential(
            ResNetBlock(1280, 1280),
            ContextualAttentionBlock(8, 160),
            ResNetBlock(1280, 1280)
        )

        self.decoder = nn.ModuleList()
        
        # 2 Res
        self.decoder.append(SwitchSequential(ResNetBlock(2560, 1280)))
        self.decoder.append(SwitchSequential(ResNetBlock(2560, 1280)))

        # upsample 1
        self.decoder.append(SwitchSequential(ResNetBlock(2560), 
                                             Upsample(1280)))

        # 3 res 3 attention 
        # upsample 1
        self.decoder.append(SwitchSequential(ResNetBlock(2560, 1280), 
                                             ContextualAttentionBlock(8, 160)))
        self.decoder.append(SwitchSequential(ResNetBlock(2560, 1280), 
                                             ContextualAttentionBlock(8, 160)))
        self.decoder.append(SwitchSequential(ResNetBlock(1920, 1280), 
                                             ContextualAttentionBlock(8, 160), 
                                             Upsample(1280, True)))

        # 3 Res 3 attention
        # upsample 1
        self.decoder.append(SwitchSequential(ResNetBlock(1920, 640), 
                                             ContextualAttentionBlock(8, 80)))
        self.decoder.append(SwitchSequential(ResNetBlock(1280, 640), 
                                             ContextualAttentionBlock(8, 80)))
        
        self.decoder.append(SwitchSequential(ResNetBlock(960, 640), 
                                             ContextualAttentionBlock(8, 80), 
                                             Upsample(640, True)))
        
        # 3 Res 3 attention
        self.decoder.append(SwitchSequential(ResNetBlock(960, 320), 
                                             ContextualAttentionBlock(8, 40)))
        self.decoder.append(SwitchSequential(ResNetBlock(640, 320), 
                                             ContextualAttentionBlock(8, 40)))
        self.decoder.append(SwitchSequential(ResNetBlock(640, 320), 
                                             AttentionBlock(8, 40)))

    def forward(self, x, context, time):
        skip_connections = []
        
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x
    

class UNET_Outputlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.norm = Normalize(in_channels, 32)                           
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = f.silu(x)
        x = self.conv(x)

        return x
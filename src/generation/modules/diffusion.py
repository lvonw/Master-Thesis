import constants
import torch
import torch.nn             as nn
import torch.nn.functional  as f

from configuration                      import Section
from generation.modules.util_modules    import (ResNetBlock, 
                                                Downsample, 
                                                Upsample, 
                                                Normalize)
from generation.modules.attention       import ContextualAttentionBlock


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()

    def forward(self, x, control_signal, time):
        time = self.time_embedding(time)

        x = self.unet(x, control_signal, time)

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
        self.name = "UNET"
        latent_channel_amount = 4
        starting_channels = 320
        num_heads = 8
        channel_multipliers_encoder = [1,2,4,4]
        channel_multipliers_decoder = channel_multipliers_encoder[::-1]
        amount_blocks = len(channel_multipliers_encoder)
        resNet_per_level_encoder = 2
        resNet_per_level_decoder = resNet_per_level_encoder + 1
        embedding_channels = 40
        skip_channels = [] 
        attention_levels = [0,1,2]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        # TODO Do this for the autoencoder too
        self.input_conv = nn.Conv2d(latent_channel_amount, 
                                    starting_channels, 
                                    kernel_size=3,
                                    stride=1, 
                                    padding=1)
        skip_channels.append(starting_channels)
        
        previous_channel_amount = starting_channels

        for level, multiplier in enumerate(channel_multipliers_encoder):
            current_channel_amount  = starting_channels * multiplier
            current_embed           = embedding_channels * multiplier
            
            for layer in range(resNet_per_level_encoder):
                current_block           = []

                current_block.append(
                    ResNetBlock(previous_channel_amount, 
                                current_channel_amount)
                )

                if level in attention_levels:
                    current_block.append(
                        ContextualAttentionBlock(num_heads, current_embed)  
                    )
                
                self.encoder.append(SwitchSequential(*current_block))
                skip_channels.append(current_channel_amount)
                previous_channel_amount = current_channel_amount

            if level + 1 < len(channel_multipliers_encoder):
                self.encoder.append(Downsample(current_channel_amount,
                                               asymmetric_padding=False))
                skip_channels.append(current_channel_amount)
        
        # Bottleneck
        self.bottleneck = SwitchSequential(
            ResNetBlock(current_channel_amount, current_channel_amount),
            ContextualAttentionBlock(num_heads, current_embed),
            ResNetBlock(current_channel_amount, current_channel_amount)
        )

        # Decoder 
        for level_idx, multiplier in enumerate(channel_multipliers_decoder):
            # Reverse levels, because we are ascending
            level = len(channel_multipliers_decoder) - (level_idx + 1)

            current_channel_amount  = starting_channels * multiplier
            current_embed           = embedding_channels * multiplier

            for layer in range(resNet_per_level_decoder):
                current_block           = []
                current_skip_channel    = skip_channels.pop()
                
                # ResNet
                current_block.append(
                    ResNetBlock(current_skip_channel + previous_channel_amount, 
                                current_channel_amount)
                )
                # Attention
                if level in attention_levels:
                    current_block.append(
                        ContextualAttentionBlock(num_heads, current_embed)  
                    )
                # Upsampling
                if (layer + 1 == resNet_per_level_decoder and level):
                    current_block.append(
                        Upsample(current_channel_amount, True)
                    )

                self.decoder.append(SwitchSequential(*current_block))
                previous_channel_amount = current_channel_amount
        
        # Output
        self.norm           = Normalize(current_channel_amount, 32)                           
        self.output_conv    = nn.Conv2d(current_channel_amount, 
                                        latent_channel_amount, 
                                        kernel_size=3, 
                                        padding=1)

    def forward(self, x, context, time):
        skip_connections = []

        x = self.input_conv(x)
        skip_connections.append(x)
        
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        x = self.norm(x)
        x = f.silu(x)
        x = self.output_conv(x)

        return x
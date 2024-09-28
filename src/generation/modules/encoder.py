import torch.nn             as nn
import torch.nn.functional  as f

from generation.modules.util_modules    import (ResNetBlock, 
                                                Downsample, 
                                                Upsample, 
                                                Normalize)
from generation.modules.attention       import AttentionBlock


class Encoder(nn.Module):
    def __init__(self,
                 data_shape,
                 latent_shape,
                 starting_channels,
                 amount_blocks,
                 channel_multipliers,
                 resNet_per_layer,
                 attention_blocks):
        super().__init__()
        self.data_shape = data_shape
        self.latent_shape = latent_shape
        
        data_channel_amount = data_shape[0]
        latent_channel_amount = latent_shape[0]

        self.input_conv = nn.Conv2d(data_channel_amount, 
                                    starting_channels, 
                                    kernel_size=3,
                                    stride=1, 
                                    padding=1)

        self.encoder = nn.ModuleList()
        previous_channel_amount = starting_channels
    
        for block in range(amount_blocks):
            current_channel_amount  = (starting_channels 
                                       * channel_multipliers[block])
            
            for _ in range(resNet_per_layer):
                self.encoder.append(ResNetBlock(previous_channel_amount,
                                                current_channel_amount))
                previous_channel_amount = current_channel_amount

            if block in attention_blocks:
                self.encoder.append(AttentionBlock(current_channel_amount))
            
            self.encoder.append(Downsample(current_channel_amount))


        # Bottleneck
        for _ in range(resNet_per_layer):
                self.encoder.append(ResNetBlock(current_channel_amount,
                                                current_channel_amount))

        self.bottleneck_resNet_1    = ResNetBlock(current_channel_amount, 
                                                  current_channel_amount)
        self.bottleneck_attention   = AttentionBlock(current_channel_amount)
        self.bottleneck_resNet_2    = ResNetBlock(current_channel_amount, 
                                                  current_channel_amount)

        self.norm = Normalize(current_channel_amount, 32)
        self.non_linearity = nn.SiLU()

        # TODO this probably makes more sense in the VAE 
        self.output_conv_1 = nn.Conv2d(current_channel_amount, 
                                       latent_channel_amount * 2, 
                                       kernel_size=3, 
                                       stride=1,
                                       padding=1)

        self.output_conv_2 = nn.Conv2d(latent_channel_amount * 2, 
                                       latent_channel_amount * 2, 
                                       kernel_size=1, 
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        x = self.input_conv(x)

        for module in self.encoder:
            x = module(x)

        x = self.bottleneck_resNet_1(x)
        x = self.bottleneck_attention(x)
        x = self.bottleneck_resNet_2(x)

        x = self.norm(x)
        x = self.non_linearity(x)

        x = self.output_conv_1(x)
        x = self.output_conv_2(x)

        return x

class Decoder(nn.Module):
    def __init__(self,
                 latent_shape,
                 data_shape,
                 starting_channels,
                 amount_blocks,
                 channel_multipliers,
                 resNet_per_layer,
                 attention_blocks):
        super().__init__()

        self.latent_shape = latent_shape
        self.data_shape = data_shape
        
        latent_channel_amount = latent_shape[0]
        data_channel_amount = data_shape[0]

        current_channel_amount = starting_channels * channel_multipliers[0]

        self.input_conv_1 = nn.Conv2d(latent_channel_amount, 
                                      latent_channel_amount, 
                                      kernel_size=1, 
                                      stride=1,
                                      padding=0)
 
        self.input_conv_2 = nn.Conv2d(latent_channel_amount,
                                      current_channel_amount, 
                                      kernel_size=3, 
                                      stride=1,
                                      padding=1)

        self.bottleneck_resNet_1    = ResNetBlock(current_channel_amount, 
                                                  current_channel_amount)
        self.bottleneck_attention   = AttentionBlock(current_channel_amount)
        self.bottleneck_resNet_2    = ResNetBlock(current_channel_amount, 
                                                  current_channel_amount)

        self.decoder = nn.ModuleList()
        for _ in range(resNet_per_layer):
                self.decoder.append(ResNetBlock(current_channel_amount,
                                                current_channel_amount))

        previous_channel_amount = current_channel_amount
        for block in range(amount_blocks):
            current_channel_amount  = (starting_channels 
                                       * channel_multipliers[block])

            self.decoder.append(Upsample(previous_channel_amount))
            
            for _ in range(resNet_per_layer):
                self.decoder.append(nn.Conv2d(previous_channel_amount,
                                              previous_channel_amount,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1))
                self.decoder.append(ResNetBlock(previous_channel_amount,
                                                current_channel_amount))
                previous_channel_amount = current_channel_amount

            if block in attention_blocks:
                self.decoder.append(AttentionBlock(current_channel_amount))

        self.norm = Normalize(current_channel_amount, 32)
        self.non_linearity = nn.SiLU()

        self.output_conv = nn.Conv2d(current_channel_amount, 
                                     data_channel_amount, 
                                     kernel_size=3, 
                                     stride=1,
                                     padding=1)

    def forward(self, z):
        z = self.input_conv_1(z)
        z = self.input_conv_2(z)

        z = self.bottleneck_resNet_1(z)
        z = self.bottleneck_attention(z)
        z = self.bottleneck_resNet_2(z)

        for module in self.decoder:
            z = module(z)

        z = self.norm(z)
        z = self.non_linearity(z)
        z = self.output_conv(z)

        return z
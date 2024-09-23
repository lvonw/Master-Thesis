import constants
import torch
import torch.nn             as nn
import torch.nn.functional  as f

from configuration  import Section
from util_modules   import (ResNetBlock, 
                            Downsample, 
                            Upsample, 
                            Normalize)
from attention      import AttentionBlock

class AutoEncoderFactory():
    def __init__(self):
        pass

    def create_auto_encoder(vae_configuration: Section):
        starting_channels   = vae_configuration["starting_channels"]
        
        data_shape      = (vae_configuration["data_num_channels"], 
                           vae_configuration["data_resolution_x"],
                           vae_configuration["data_resolution_y"])
        
        latent_shape    = (vae_configuration["latent_num_channels"], 
                           vae_configuration["latent_resolution_x"],
                           vae_configuration["latent_resolution_y"]) 
        
        
        channel_multipliers_encoder = vae_configuration["channel_multipliers"]
        channel_multipliers_decoder = reversed(channel_multipliers_encoder)
        
        amount_resolutions  = len(channel_multipliers_encoder)
        
        attention_resolutions_encoder = vae_configuration[
            "attention_resolutions"]
        
        attention_resolutions_decoder = attention_resolutions_encoder

        resNet_per_layer_encoder = vae_configuration["ResNet_blocks_per_layer"]
        resNet_per_layer_decoder = resNet_per_layer_encoder + 1

        encoder = VAEEncoder(data_shape,
                             latent_shape,
                             starting_channels,
                             amount_resolutions,
                             channel_multipliers_encoder,
                             resNet_per_layer_encoder,
                             attention_resolutions_encoder)

        decoder = VAEDecoder(latent_shape,
                             data_shape,
                             starting_channels,
                             amount_resolutions,
                             channel_multipliers_decoder,
                             resNet_per_layer_decoder,
                             attention_resolutions_decoder)

        return VariationalAutoEncoder(encoder, decoder)

class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

class VAEEncoder(nn.Module):
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

    def forward(self, x, noise):
        for module in self.encoder:
            x = module(x)

        x = self.bottleneck_resNet_1(x)
        x = self.bottleneck_attention(x)
        x = self.bottleneck_resNet_2(x)

        x = self.norm(x)
        x = self.non_linearity(x)

        x = self.output_conv_1(x)
        x = self.output_conv_2(x)

        mu, log_variance = torch.chunk(x, 2, dim=1) 
        sigma = torch.clamp(log_variance, -30, 20).exp().sqrt()

        # Reparameterization where noise ~ N(0,I)
        x = mu + sigma * noise
        
        # SD constant, idk why this is here, can try removing it
        x = x * 0.18215

        return x

class VAEDecoder(nn.Module):
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
        z = z / 0.18215

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
import constants
import torch
import torch.nn                 as nn
import torch.nn.functional      as f

from configuration              import Section
from generation.modules.encoder import Encoder, Decoder

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

        encoder = Encoder(data_shape,
                          latent_shape,
                          starting_channels,
                          amount_resolutions,
                          channel_multipliers_encoder,
                          resNet_per_layer_encoder,
                          attention_resolutions_encoder)

        decoder = Decoder(latent_shape,
                          data_shape,
                          starting_channels,
                          amount_resolutions,
                          channel_multipliers_decoder,
                          resNet_per_layer_decoder,
                          attention_resolutions_decoder)

        return VariationalAutoEncoder(encoder, 
                                      decoder,
                                      data_shape,
                                      latent_shape)

class VariationalAutoEncoder(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 data_shape, 
                 latent_shape):
        super().__init__()

        self.encoder        = encoder
        self.decoder        = decoder
        self.data_shape     = data_shape
        self.latent_shape   = latent_shape

    def training_step(self):
        pass

    def encode(self, x):
        x = self.encoder(x)
        mu, log_variance = torch.chunk(x, 2, dim=1) 
        
        sigma   = torch.clamp(log_variance, -30, 20).exp().sqrt()
        # Reparameterization where noise ~ N(0,I)
        noise   = torch.randn(mu.shape).to(mu.device)
        x       = mu + sigma * noise
        # SD constant, idk why this is here, can try removing it
        x       = x * 0.18215
        
        return (mu, x)

    def decode(self, z):
        z = z / 0.18215
        x = self.decoder(z)
        return x
    
    def forward(self, x, decode_x=True):
        z = self.encode(x)

        if decode_x:
            z = z[1]
        else: 
            z = z[0]
            
        x = self.decode(z)

        return x



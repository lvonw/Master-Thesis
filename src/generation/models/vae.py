import constants
import torch
import torch.nn                 as nn
import torch.nn.functional      as f

from configuration              import Section
from generation.modules.encoder import Encoder, Decoder
from util                       import get_device

class AutoEncoderFactory():
    def __init__(self):
        pass

    def create_auto_encoder(vae_configuration: Section):
        architecture = vae_configuration["architecture"]

        data_shape      = (vae_configuration["data_num_channels"], 
                           vae_configuration["data_resolution_x"],
                           vae_configuration["data_resolution_y"])
        
        latent_shape    = (vae_configuration["latent_num_channels"], 
                           vae_configuration["latent_resolution_x"],
                           vae_configuration["latent_resolution_y"]) 
        
        starting_channels   = architecture["starting_channels"]
        
        channel_multipliers_encoder = architecture["channel_multipliers"]
        channel_multipliers_decoder = channel_multipliers_encoder[::-1]
        
        amount_resolutions  = len(channel_multipliers_encoder)
        
        attention_resolutions_encoder = architecture[
            "attention_resolutions"]
        
        attention_resolutions_decoder = attention_resolutions_encoder

        resNet_per_layer_encoder = architecture["ResNet_blocks_per_layer"]
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


    def ELBO_loss(a, b, c):
        return a


    def training_step(self, inputs, labels):
        reconstructions, posteriours = self(inputs)

        log_var, mean = posteriours[2], posteriours[0]

        # Reconstruction loss ensures that we learn meaningful latents, and
        # good reconstructions
        reconstruction_loss = f.binary_cross_entropy(input  = reconstructions, 
                                                     target = inputs,
                                                     reduction="sum")
        
        # KL-Divergence between a standard gaussian and our posterior
        # ensures that our latent space is as close as possible to a gaussian
        kl_divergence = torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/-2

        return reconstruction_loss + kl_divergence

    def encode(self, x):
        x = self.encoder(x)
        mu, log_variance = torch.chunk(x, 2, dim=1) 
        
        sigma   = torch.clamp(log_variance, -30, 20).exp().sqrt()
        # Reparameterization where noise ~ N(0,I)
        noise   = torch.randn(mu.shape).to(mu.device)
        x       = mu + sigma * noise
        # SD constant, idk why this is here, can try removing it
        x       = x * 0.18215
        
        return (mu, x, log_variance)

    def decode(self, z):
        z = z / 0.18215
        x = self.decoder(z)
        # make sure that the values are in image space
        x = f.sigmoid(x)
        return x
    

    def generate(self):
        noise   = torch.randn((1,) + self.latent_shape)
        return self.decode(noise)
    
    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)

        if sample_posterior:
            z = posterior[1]
        else: 
            z = posterior[0]
            
        x = self.decode(z)


        return x, posterior



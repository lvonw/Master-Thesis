import constants
import torch
import torch.nn                 as nn
import torch.nn.functional      as f

from configuration              import Section
from debug                      import Printer, LogLevel
from enum                       import Enum
from generation.modules.encoder import Encoder, Decoder

class LossMethod(Enum):
    BINARY_CROSS_ENTROPY    = "bianry cross entropy"
    MEAN_SQUARED_ERROR      = "mean squared error"
    
class AutoEncoderFactory():
    def __init__(self):
        pass

    def create_auto_encoder(vae_configuration: Section):
        architecture    = vae_configuration["architecture"]

        beta            = vae_configuration["beta"]

        data_shape      = (vae_configuration["data_num_channels"], 
                           vae_configuration["data_resolution_x"],
                           vae_configuration["data_resolution_y"])
        
        latent_shape    = (vae_configuration["latent_num_channels"], 
                           vae_configuration["latent_resolution_x"],
                           vae_configuration["latent_resolution_y"]) 
        
        starting_channels   = architecture["starting_channels"]

        loss_method         = LossMethod.MEAN_SQUARED_ERROR
        
        channel_multipliers_encoder = architecture["channel_multipliers"]
        channel_multipliers_decoder = channel_multipliers_encoder[::-1]
        
        amount_resolutions  = len(channel_multipliers_encoder)
        
        use_attention = architecture["use_attention"]
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
                          attention_resolutions_encoder,
                          use_attention)

        decoder = Decoder(latent_shape,
                          data_shape,
                          starting_channels,
                          amount_resolutions,
                          channel_multipliers_decoder,
                          resNet_per_layer_decoder,
                          attention_resolutions_decoder,
                          use_attention)

        return VariationalAutoEncoder(encoder, 
                                      decoder,
                                      data_shape,
                                      latent_shape,
                                      vae_configuration["name"],
                                      beta=beta,
                                      loss_method=loss_method)

class VariationalAutoEncoder(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 data_shape, 
                 latent_shape,
                 name,
                 beta,
                 loss_method):
        super().__init__()

        self.model_family   = "vae"
        self.name           = name
        
        self.encoder        = encoder
        self.decoder        = decoder
        self.data_shape     = data_shape
        self.latent_shape   = latent_shape
        self.beta           = beta
        self.loss_method    = loss_method


    def training_step(self, inputs, labels, loss_weights):
        reconstructions, latent_encoding = self(inputs)
    
        # Reconstruction Loss
        match self.loss_method:
            case LossMethod.BINARY_CROSS_ENTROPY: 
                reconstruction_loss = f.binary_cross_entropy(
                    input  = reconstructions, 
                    target = inputs,
                    reduction="none")
            case LossMethod.MEAN_SQUARED_ERROR:
                reconstruction_loss = f.mse_loss(
                    input  = reconstructions, 
                    target = inputs,
                    reduction="none")   
                       
        reconstruction_loss = torch.sum(reconstruction_loss, dim=(1, 2, 3))
        
        # KL-Divergence between Posterior and Gaussian
        mean            = latent_encoding.mean     
        log_var         = latent_encoding.log_variance
        kl_divergence   = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        kl_divergence   = torch.sum(kl_divergence, dim=(1, 2, 3))
        
        # Combine the losses
        individual_losses   = reconstruction_loss + self.beta * kl_divergence

        if labels is not None and loss_weights is not None:
            for idx in range(len(individual_losses)):
                individual_losses[idx] *= loss_weights[labels[idx].item()]

        reduced_loss        = torch.sum(individual_losses)        

        return reduced_loss, individual_losses
    
    def on_training_step_completed(self):
        return 

    def encode(self, x):
        x = self.encoder(x)
        mu, log_variance = torch.chunk(x, 2, dim=1) 
        
        sigma   = torch.clamp(log_variance, -30, 20).exp().sqrt()
        # Reparameterization where noise ~ N(0,I)
        noise   = torch.randn(mu.shape).to(mu.device)
        x       = mu + sigma * noise
        
        return LatentEncoding(x, mu, log_variance)

    def decode(self, z):
        x = self.decoder(z)

        if self.loss_method == LossMethod.BINARY_CROSS_ENTROPY:
            x = f.sigmoid(x)
        
        return x

    def generate(self):
        noise   = torch.randn((1,) + self.latent_shape)
        return self.decode(noise)
    
    def forward(self, x, sample_posterior=True):
        if x.shape[1:] != self.data_shape:
            Printer().print_log("Data shape does not match specified shape!",
                                LogLevel.WARNING)
            return x, None
        
        latent_encoding = self.encode(x)

        if sample_posterior:
            z = latent_encoding.latents
        else: 
            z = latent_encoding.mean
            
        x = self.decode(z)

        return x, latent_encoding

class LatentEncoding():
    def __init__(self, latents, mean, log_variance):
        self.latents        = latents
        self.mean           = mean
        self.log_variance   = log_variance
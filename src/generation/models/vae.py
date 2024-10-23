import torch
import torch.nn                 as nn
import torch.nn.functional      as f
import torch.optim              as optim

from configuration                      import Section
from debug                              import Printer, LogLevel, LossLog
from enum                               import Enum
from generation.models.discriminator    import Discriminator
from generation.modules.ema             import EMA
from generation.modules.encoder         import Encoder, Decoder
from torchmetrics.image.lpip            import LearnedPerceptualImagePatchSimilarity
from lpips                              import LPIPS

class LossMethod(Enum):
    BINARY_CROSS_ENTROPY    = "binary cross entropy"
    MEAN_SQUARED_ERROR      = "mean squared error"
    ABSOLUTE_DIFFERENCE     = "absolute difference"

    
class AutoEncoderFactory():
    def __init__(self):
        pass

    def create_auto_encoder(vae_configuration: Section,
                            pre_trained = False):
        printer         = Printer()

        architecture    = vae_configuration["Architecture"]
        loss            = vae_configuration["Loss"]

        beta                    = loss["kl_beta"]
        use_class_weights       = loss["use_class_weights"]

        data_shape      = (vae_configuration["data_num_channels"], 
                           vae_configuration["data_resolution_x"],
                           vae_configuration["data_resolution_y"])
        
        latent_shape    = (vae_configuration["latent_num_channels"], 
                           vae_configuration["latent_resolution_x"],
                           vae_configuration["latent_resolution_y"]) 
        
        starting_channels   = architecture["starting_channels"]

        loss_method         = LossMethod.ABSOLUTE_DIFFERENCE
        
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
        
        # Vae Gan =============================================================
        use_discriminator       = (loss["use_discriminator"] 
                                   and not pre_trained)
        discriminator_weight    = loss["discriminator_weight"]
        discriminator_warmup    = loss["discriminator_warm_up"]
        discriminator           = None
        if use_discriminator:   
            printer.print_log("Using Discriminator")
            discriminator = Discriminator(starting_channels=data_shape[0])

        # Perceptual Metric ===================================================
        use_perceptual_loss     = (loss["use_perceptual_loss"]
                                   and not pre_trained)
        perceptual_weight       = loss["perceptual_weight"]
        perceptual_loss         = None
        if use_perceptual_loss:
            printer.print_log("Using Perceptual Loss")
            # perceptual_loss = LearnedPerceptualImagePatchSimilarity(
            #     net_type    = loss["perceptual_net"], 
            #     reduction   = "sum").eval()
            
            perceptual_loss = LPIPS().eval()
            
        # EMA =================================================================
        # SDXL mentions that they use EMA
        use_ema                 = vae_configuration["EMA"]["active"]
        encoder_ema_model       = None
        decoder_ema_model       = None
        if use_ema:
            encoder_ema_model   = EMA(encoder, vae_configuration["EMA"])
            decoder_ema_model   = EMA(decoder, vae_configuration["EMA"])

        return VariationalAutoEncoder(
            encoder                 = encoder, 
            decoder                 = decoder,
            data_shape              = data_shape,
            latent_shape            = latent_shape,
            name                    = vae_configuration["name"],
            beta                    = beta,
            loss_method             = loss_method,
            use_discriminator       = use_discriminator,
            use_class_weights       = use_class_weights,
            discriminator           = discriminator,
            discriminator_weight    = discriminator_weight,
            discriminator_warmup    = discriminator_warmup,
            use_perceptual_loss     = use_perceptual_loss,
            perceptual_loss         = perceptual_loss,
            perceptual_weight       = perceptual_weight,
            use_ema                 = use_ema,
            encoder_ema_model       = encoder_ema_model,
            decoder_ema_model       = decoder_ema_model)

class VariationalAutoEncoder(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 data_shape, 
                 latent_shape,
                 name,
                 beta,
                 loss_method,
                 use_discriminator,
                 use_class_weights,
                 discriminator,
                 discriminator_weight,
                 discriminator_warmup,
                 use_perceptual_loss,
                 perceptual_loss,
                 perceptual_weight,
                 use_ema,
                 encoder_ema_model,
                 decoder_ema_model):
        super().__init__()

        self.model_family   = "vae"
        self.name           = name
        self.loss_log       = LossLog()
        
        self.encoder        = encoder
        self.decoder        = decoder
        self.data_shape     = data_shape
        self.latent_shape   = latent_shape
        self.beta           = beta
        self.loss_method    = loss_method

        self.use_class_weights = use_class_weights
        self.use_discriminator = use_discriminator

        self.discriminator          = discriminator
        self.discriminator_weight   = discriminator_weight
        self.discriminator_warmup   = discriminator_warmup
        
        self.use_perceptual_loss    = use_perceptual_loss
        self.perceptual_weight      = perceptual_weight
        self.perceptual_loss        = perceptual_loss

        self.use_ema                = use_ema
        self.encoder_ema_model      = encoder_ema_model
        self.decoder_ema_model      = decoder_ema_model

        self.optimizers             = self.__get_optimizers()

        

    # =========================================================================
    # Training
    # =========================================================================
    def training_step(self, 
                      inputs, 
                      labels, 
                      loss_weights,
                      epoch_idx,
                      training_step_idx, 
                      optimizer_idx):
        """ Inspired by Stable Diffusion autoencoder_kl and Taming LPIPS"""
        
        reconstructions, latent_encoding    = self(inputs)
        train_discriminator                 = optimizer_idx == 1

        # Discriminator Training ==============================================
        if self.use_discriminator and train_discriminator:
            if epoch_idx < self.discriminator_warmup:
                # Zero loss
                return torch.tensor(0.0), None
            
            input_logits            = self.discriminator(
                inputs.contiguous().detach())
            
            reconstruction_logits   = self.discriminator(
                reconstructions.contiguous().detach())
            
            loss = self.hinge_loss(input_logits, reconstruction_logits)
            
            self.loss_log.add_entry("Discriminator", loss)
            return loss, None

        # VAE Training ========================================================
        # Reconstruction Loss -------------------------------------------------
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
            # Places higher focus on less blurry reconstructions than MSE
            case LossMethod.ABSOLUTE_DIFFERENCE:
                reconstruction_loss = torch.abs(reconstructions - inputs)   

        individual_reconstruction_loss = torch.sum(reconstruction_loss, 
                                                   dim=(1, 2, 3))
        
        # Perceptual Loss -----------------------------------------------------
        # assumes we only have one channel, at some point it will have to be 
        # more dynamic
        if self.use_perceptual_loss:
            lpips_inputs            = (inputs
                                       .repeat(1, 3, 1, 1)
                                       .contiguous())
            lpips_reconstructions   = (reconstructions
                                       .repeat(1, 3, 1, 1)
                                       .contiguous()) 
            
            # Currently this is the mean, so adding to individual doesnt make 
            # too much semantic sense
            perceptual_loss = self.perceptual_loss(lpips_inputs, 
                                                   lpips_reconstructions)
            
            perceptual_loss = perceptual_loss.squeeze()
            
            individual_reconstruction_loss += (self.perceptual_weight 
                                               * perceptual_loss)
        
        # Class weights if necessary ------------------------------------------
        # Not entirely sure what makes more sense, to have this here or at the 
        # end. Also might not use this afterall
        if (self.use_class_weights 
            and labels          is not None 
            and loss_weights    is not None):
            
            for idx in range(len(individual_reconstruction_loss)):
                individual_reconstruction_loss[idx] *= loss_weights[labels[idx]
                                                                    .item()]

        # Average losses
        reconstruction_loss = torch.mean(individual_reconstruction_loss)
        
        self.loss_log.add_entry("Reconstruction", reconstruction_loss)


        # KL-Divergence between Posterior and Gaussian ------------------------
        mean            = latent_encoding.mean     
        log_var         = latent_encoding.log_variance
        kl_divergence   = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        
        individual_kl_divergence    = torch.sum(kl_divergence, dim=(1, 2, 3))
        
        kl_divergence = torch.mean(individual_kl_divergence)
        kl_divergence *= self.beta
        
        self.loss_log.add_entry("KL-Divergence", kl_divergence)
    
        # Discriminator Loss --------------------------------------------------
        discriminator_loss = 0.
        if self.use_discriminator and epoch_idx >= self.discriminator_warmup:

            # Use discriminator as metric
            logits = self.discriminator(reconstructions.contiguous())
            
            # We want the logits as large as possible 
            # -> Positive classification
            discriminator_loss = -torch.mean(logits)

            reconstruction_gradients = torch.autograd.grad(
                reconstruction_loss, 
                self.decoder.output_conv.weight, 
                retain_graph=True)[0]
            
            discriminator_gradients  = torch.autograd.grad(
                discriminator_loss, 
                self.decoder.output_conv.weight, 
                retain_graph=True)[0]

            # Larger weight when the gradient for the reconstruction is larger
            # than for the descrimination loss 
            # -> we likely have a good discriminator
            weight = (torch.norm(reconstruction_gradients) 
                      / (torch.norm(discriminator_gradients) + 1e-4))
            
            weight = torch.clamp(weight, 0.0, 1e4).detach()
            print (weight)
            print (discriminator_loss)

            discriminator_loss *= weight * self.discriminator_weight

    

            self.loss_log.add_entry("Discrimination", discriminator_loss)

        # TODO remove
        discriminator_loss = 0.
            
        # Combine the losses --------------------------------------------------
        loss  = (reconstruction_loss 
                 + kl_divergence
                 + discriminator_loss)    

        self.loss_log.add_entry("Total", loss)

        return loss, None
    
    def on_training_step_completed(self):
        if self.use_ema:
            self.encoder_ema_model.ema_step(self.encoder)
            self.decoder_ema_model.ema_step(self.decoder)
    
    def hinge_loss(self, input_logits, reconstruction_logits):
        """ Taming hinge_d_loss """
        loss_real = torch.mean(f.relu(1. - input_logits))
        loss_fake = torch.mean(f.relu(1. + reconstruction_logits))
        
        return 0.5 * (loss_real + loss_fake)
    
    def __get_optimizers(self):
        # Need to make the distinction here, so we dont train the discriminator
        # when training the vae and vice versa
        vae_parameter_list      = (list(self.encoder.parameters())
                                   + list(self.decoder.parameters()))
        vae_optimizer           = optim.Adam(vae_parameter_list, lr=4.5e-6)    
        if not self.use_discriminator:
            return [vae_optimizer]

        discriminator_optimizer = optim.Adam(self.discriminator.parameters(), 
                                             lr=4.5e-6)
        return [vae_optimizer, discriminator_optimizer]


    # =========================================================================
    # Sampling
    # =========================================================================
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
        return x
    
    def generate(self):
        noise   = torch.randn((1,) + self.latent_shape)
        return self.decode(noise)
    
    def forward(self, x):
        if x.shape[1:] != self.data_shape:
            Printer().print_log("Data shape does not match specified shape!",
                                LogLevel.ERROR)
            return x, None
        
        latent_encoding = self.encode(x)

        z = latent_encoding.latents
        
        x = self.decode(z)

        return x, latent_encoding
    
    # =========================================================================
    # General
    # =========================================================================
    def on_loaded_as_pretrained(self):
        self.__apply_ema()
    
    def __apply_ema(self):
        # Be sure to not save this as long as we are not a submodel
        self.encoder_ema_model.apply_to_model(self.encoder)
        self.decoder_ema_model.apply_to_model(self.decoder)

        self.encoder_ema_model = None
        self.decoder_ema_model = None


class LatentEncoding():
    def __init__(self, latents, mean, log_variance):
        self.latents        = latents
        self.mean           = mean
        self.log_variance   = log_variance
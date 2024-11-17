"""
Implementation guided by:
https://github.com/openai/improved-diffusion
https://github.com/hkproj/pytorch-stable-diffusion
https://github.com/Stability-AI/stablediffusion
https://github.com/dome272/Diffusion-Models-pytorch
https://github.com/gmongaras/Diffusion_models_from_scratch
"""

import constants
import enum
import torch
import util

import numpy                        as np
import torch.nn                     as nn
import torch.nn.functional          as f
import torch.optim                  as optim

from debug                          import Printer, LogLevel, LossLog
from generation.models.vae          import AutoEncoderFactory
from generation.modules.diffusion   import Diffusion
from generation.modules.ema         import EMA
from tqdm                           import tqdm

class BetaSchedules(enum.Enum):
    LINEAR  = "Linear"
    COSINE  = "Cosine"

class PredictionType(enum.Enum):
    L_HYBRID        = "Mean_Variance"
    L_SIMPLE        = "Epsilon_Simple"

class DDPM(nn.Module):
    def __init__(self, 
                 configuration,
                 generator=torch.Generator(device=util.get_device()), 
                 beta_start=0.00085,
                 beta_end = 0.0120, 
                 amount_classes=0):
        super().__init__()

        self.model_family           = "diffusion"
        self.name                   = configuration["name"]
        self.loss_log               = LossLog()
        self.can_save               = True

        self.learn_variance         = configuration["learn_variance"]
        if self.learn_variance:
            Printer().print_log("Learning Variance")
            self.prediction_type    = PredictionType.L_HYBRID
        else:
            Printer().print_log("Learning Epsilon")
            self.prediction_type    = PredictionType.L_SIMPLE

        if configuration["beta_schedule"] == BetaSchedules.COSINE.value:
            Printer().print_log("Using Cosine Beta Schedule")
            self.beta_schedule      = BetaSchedules.COSINE
        else:
            Printer().print_log("Using Linear Beta Schedule")
            self.beta_schedule      = BetaSchedules.LINEAR

        self.device                 = util.get_device()
        self.generator              = generator
        self.amount_classes         = amount_classes

        self.amount_training_steps  = configuration["training_steps"]
        self.amount_sample_steps    = configuration["sample_steps"]
        self.sample_steps           = self.__get_sampling_timesteps(
            self.amount_sample_steps)
        
        self.input_shape            = (configuration["input_num_channels"], 
                                       configuration["input_resolution_x"],
                                       configuration["input_resolution_y"])
        
        # LATENT ==============================================================
        self.is_latent          = configuration["Latent"]["active"]
        if self.is_latent:
            self.latent_model   = AutoEncoderFactory.create_auto_encoder(
                configuration["Latent"]["model"],
                configuration["Latent"]["pre_trained"])
            
            # Latent diffusion paper
            self.inverse_latent_std = 0.3576
            
            Printer().print_log("Loading Latent model")
            if (configuration["Latent"]["pre_trained"] 
                and not util.load_checkpoint(self.latent_model, strict=False)):
                
                Printer().print_log(
                    f"Model {self.latent_model.name} could not be loaded",
                    LogLevel.WARNING)
            else:
                self.latent_model.on_loaded_as_pretrained()  
                Printer().print_log("Finished.")


        # DIFFUSION MODEL =====================================================               
        self.model          = Diffusion(configuration, 
                                        amount_classes,
                                        self.learn_variance)

        # EMA =================================================================
        self.use_ema        = configuration["EMA"]["active"]
        if self.use_ema:
            self.ema_model  = EMA(self.model, configuration["EMA"])

        # CLASSIFIER FREE GUIDANCE ============================================
        self.use_classifier_free_guidance = configuration[
            "Classifier_free_guidance"]["active"]
        self.no_class_probability = configuration[
            "Classifier_free_guidance"]["no_class_probability"]
        self.cfg_weight = configuration[
            "Classifier_free_guidance"]["weight"]

        # DDPM PARAMETERS =====================================================
        self.register_buffer("betas", self.__create_beta_schedule(
            self.amount_training_steps, 
            beta_start, 
            beta_end, 
            self.beta_schedule))
        self.register_buffer("log_betas", torch.log(self.betas))

        self.register_buffer("alphas",          1.0 - self.betas)
        self.register_buffer("alpha_bars",      torch.cumprod(self.alphas, 0))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))

        vlb_coefficients = self.__compute_vlb_coefficients()
        self.register_buffer("x_zero_coefs",            vlb_coefficients[0])
        self.register_buffer("x_t_coefs",               vlb_coefficients[1])
        self.register_buffer("variance_t",              vlb_coefficients[2])
        self.register_buffer("log_variance_t",          vlb_coefficients[3])
        self.register_buffer("x_zero_x_t_coefs",        vlb_coefficients[4])
        self.register_buffer("x_zero_epsilon_coefs",    vlb_coefficients[5])

        epsilon_coefficients = self.__compute_epsilon_coefficients()
        self.register_buffer("one_over_sqrt_alphas",  epsilon_coefficients[0])
        self.register_buffer("epsilon_coefficient",   epsilon_coefficients[1])

        # TRAINING ============================================================
        self.optimizers     = self.__get_optimizers()

    # =========================================================================
    # Training
    # =========================================================================
    def training_step(self, 
                      inputs, 
                      labels, 
                      loss_weights,
                      epoch_idx,
                      total_training_step_idx, 
                      relative_training_step_idx,
                      optimizer_idx,
                      global_rank,
                      local_rank):
        
        # Latent Encoding =====================================================
        if self.is_latent:
            self.latent_model.eval()
            with torch.no_grad():
                inputs = self.latent_model.encode(inputs).latents
                inputs *= self.inverse_latent_std

        # Forward Process =====================================================
        timesteps = self.__sample_from_timesteps(
            self.amount_training_steps, inputs.shape[0]).to(inputs.device)
        
        noised_images, noise = self.__add_noise(inputs, timesteps)

        # Classifier Free Guidance ============================================
        if self.use_classifier_free_guidance: 
            # Perhaps we could try setting each entry independantly
            mask = torch.rand(
                labels.shape[0],
                device=labels.device) < self.no_class_probability
            
            if len(self.amount_classes) > 1:
                mask = mask.unsqueeze(1).repeat(1, labels.shape[1])

            labels = torch.where(mask, constants.NULL_LABEL, labels)   

        # Noise Prediction ====================================================
        model_output = self.model(noised_images, labels, timesteps)

        match self.prediction_type:
            # As proposed by algorithm 2 DDPM
            case PredictionType.L_SIMPLE:
                loss = self.__l_simple(noise, 
                                       model_output)
            # As proposed by Improved DDPMs
            case PredictionType.L_HYBRID:
                loss = self.__l_hybrid(noise,
                                       inputs,
                                       noised_images, 
                                       model_output,
                                       timesteps)

        return loss, None
    
    def on_training_step_completed(self):
        if self.use_ema:
            self.ema_model.ema_step(self.model)
    
    def __l_hybrid(self, noise, x_zero, x_t, model_output, timesteps):
        predicted_noise, predicted_log_variance = model_output.chunk(2, dim=1)

        vlb_weight  = 0.001
        l_simple    = self.__l_simple(noise, predicted_noise)
        l_vlb       = self.__l_vlb(x_zero,
                                   x_t,
                                   predicted_noise,
                                   predicted_log_variance,
                                   timesteps)
        
        loss        = l_simple + vlb_weight * l_vlb
        
        self.loss_log.add_entry("L_hybrid", loss)
        return loss
    
    def __l_simple(self, noise, predicted_noise):
        loss = f.mse_loss(noise, predicted_noise)

        self.loss_log.add_entry("L_simple", loss)
        return loss

    def __l_vlb(self, 
                x_zero, 
                x_t, 
                predicted_noise, 
                predicted_log_variance, 
                timesteps):
        noise_parameters = self.__get_parameters_from_x_t(
            x_zero, 
            x_t, 
            timesteps)
        
        predicted_noise_parameters = self.__get_parameters_from_noise(
            x_t,
            predicted_noise,
            predicted_log_variance,
            timesteps)
        
        kl_divergence = self.__gaussian_kl_divergence(
            noise_parameters.mean,
            noise_parameters.log_variance,
            predicted_noise_parameters.mean,
            predicted_noise_parameters.log_variance)

        loss = kl_divergence

        self.loss_log.add_entry("L_vlb", loss)
        return loss
    
    def __batchify(self, tensor):
        return tensor[:, None, None, None]
    
    def __get_parameters_from_x_t(self, x_zero, x_t, timesteps):        
        mean = (x_zero  * self.__batchify(self.x_zero_coefs[timesteps]) 
                + x_t   * self.__batchify(self.x_t_coefs[timesteps]))
                
        variance        = self.__batchify(self.variance_t[timesteps])
        log_variance    = self.__batchify(self.log_variance_t[timesteps])
        
        return NoiseParameters(mean, variance, log_variance)
    
    def __get_parameters_from_noise(self, x_t, noise, log_variance, timesteps):
        x_zero = (
            x_t     * self.__batchify(self.x_zero_x_t_coefs[timesteps])  
            - noise * self.__batchify(self.x_zero_epsilon_coefs[timesteps]))

        mean = (
            x_zero  * self.__batchify(self.x_zero_coefs[timesteps]) 
            + x_t   * self.__batchify(self.x_t_coefs[timesteps]))
        
        # Improved DDPM formula 15
        log_beta        = self.__batchify(self.log_betas[timesteps])
        log_beta_tilde  = self.__batchify(self.log_variance_t[timesteps])

        # Latent space is N(0, I) so to use this as a gate we translate it
        log_variance    = (log_variance + 1) / 2
        log_variance    = (log_variance         * log_beta 
                           + (1 - log_variance) * log_beta_tilde)
        
        variance = torch.exp(log_variance)
        
        return NoiseParameters(mean, variance, log_variance)
    
    def __gaussian_kl_divergence(self,
                                 p_mean, 
                                 p_log_variance, 
                                 q_mean, 
                                 q_log_variance):
        
        kl_divergence = -0.5 * (
            1.0 
            + p_log_variance - q_log_variance
            - torch.exp(p_log_variance - q_log_variance)
            - ((p_mean - q_mean) ** 2) * torch.exp(-q_log_variance))
        
        kl_divergence = kl_divergence.mean()
        return kl_divergence

    def __get_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        # optimizer = optim.Adam(self.model.parameters(), lr=4.5e-6)
        return [optimizer]

    # =========================================================================
    # Sampling
    # =========================================================================
    def forward(self, x, training=False, *args, **kwargs):
        if training:
            return self.training_step(*args, **kwargs)

    def generate(self,
                 labels=None, 
                 amount=1, 
                 input_tensor=None, 
                 img2img_strength=800,
                 mask = None,
                 masked_input = None):

        if labels is None:
            labels = []
        elif issubclass(int, type(labels)):
            labels = [labels] * amount

        for idx in range(len(labels)):
            if labels[idx] is None:
                labels[idx] = constants.NULL_LABEL
        
        for idx in range(len(labels), amount):
            labels.append(constants.NULL_LABEL)

        labels = torch.tensor(labels, device=self.device)

        return self.sample(amount, 
                           labels, 
                           input_tensor=input_tensor, 
                           img2img_strength=img2img_strength,
                           mask=mask,
                           masked_input=masked_input)

    @torch.no_grad()
    def sample(self, 
               amount_samples, 
               control_signals,
               input_tensor=None,
               img2img_strength=800,
               mask = None,
               masked_input = None):
        """ Algorithm 2 DDPM """

        # Sketch guidance =====================================================
        starting_offset = 0
        if input_tensor is None:
            # Start with pure noise -------------------------------------------
            x = torch.randn((amount_samples,) + self.input_shape, 
                            device=control_signals.device)
        else:
            # Encoding input if sketch guided --------------------------------- 
            starting_offset = img2img_strength

            timesteps = torch.tensor([self.sample_steps[starting_offset]] 
                                     * amount_samples).to(input_tensor.device)
            
            x   = self.latent_model.encode(input_tensor).latents
            x   *= self.inverse_latent_std
            x   = x.repeat(amount_samples, 1, 1, 1)
            x, _= self.__add_noise(x, timesteps)
            

        # Masking =============================================================
        if mask is not None:
            if self.is_latent:
                self.latent_model.eval()

                masked_input = self.latent_model.encode(masked_input).latents
                masked_input *= self.inverse_latent_std

                # mask = self.latent_model.encode(mask).latents
                # mask *= mask
                mask = f.interpolate(
                    mask, 
                    size=(32, 32), 
                    mode="bilinear",
                    align_corners=False  
                )

            
            inverted_mask = 1 - mask

        # Main Loop ===========================================================
        for _, timestep in tqdm(
            enumerate(self.sample_steps[starting_offset:], starting_offset),
            total   = self.amount_sample_steps,
            desc    = "Generating Image",
            initial = starting_offset,
            leave   = False):
        
            timesteps = torch.full((amount_samples,),
                                    timestep, 
                                    device=control_signals.device)
                    
            model_output = self.model(x, control_signals, timesteps)

            # Classifier Free Guidance ========================================
            # Theoretically it's slower to have 2 passes through the model, 
            # however this saves on VRAM, therefore making it faster on lower
            # end GPUs
            if (self.use_classifier_free_guidance and control_signals != None):  
                
                null_labels = torch.full_like(control_signals, 
                                              constants.NULL_LABEL)
                unconditional_model_output = self.model(x, 
                                                        null_labels, 
                                                        timesteps)

                model_output = torch.lerp(unconditional_model_output, 
                                          model_output,
                                          self.cfg_weight)


            # Denoising  ======================================================
            match self.prediction_type:
                # As proposed by algorithm 2 DDPM
                case PredictionType.L_SIMPLE:
                    x = self.__predict_epsilon(timestep, 
                                            x, 
                                            model_output)
                # As proposes by Improved DDPMs
                case PredictionType.L_HYBRID:
                    x = self.__predict_mean_variance(timesteps, 
                                                     x, 
                                                     model_output)
                    
            # Apply Mask ======================================================
            if mask is not None:
                noised_masked_input, _ = self.__add_noise(masked_input, 
                                                          timesteps) 
                # if self.is_latent:
                #     x /= self.inverse_latent_std
                #     x = self.latent_model.decode(x)

                #     noised_masked_input /= self.inverse_latent_std  
                #     noised_masked_input = self.latent_model.decode(
                #         noised_masked_input)


                # print (mask.shape)
                x = noised_masked_input * mask + x * inverted_mask
            
                # if self.is_latent:
                #     x = self.latent_model.encode(x).latents
                #     x *= self.inverse_latent_std
    
            x = torch.clamp(x, -1.0, 1.0)

        # Decoding ============================================================
        if self.is_latent:
            self.latent_model.eval()
            x /= self.inverse_latent_std
            x = self.latent_model.decode(x)

        return x
    
    # =========================================================================
    # Predict Epsilon
    # =========================================================================
    def __predict_epsilon(self, timestep, x_t, model_output):
        """ Algorithm 2 DDPM """
        if timestep > 0:
            noise = torch.randn(model_output.shape, 
                                generator=self.generator, 
                                device=model_output.device, 
                                dtype=model_output.dtype)
        else: 
            noise = 0
                        
        x_t_minus_one = x_t - self.epsilon_coefficient[timestep] * model_output
        x_t_minus_one *= self.one_over_sqrt_alphas[timestep]
        x_t_minus_one += torch.sqrt(self.betas[timestep]) * noise

        return x_t_minus_one
    

    def __compute_epsilon_coefficients(self):
        one_over_sqrt_alphas    = 1. / torch.sqrt(self.alphas)
        epsilon_coefficient     = ((1. - self.alphas) 
                                    / torch.sqrt(1. - self.alpha_bars))
        
        return one_over_sqrt_alphas, epsilon_coefficient

    # =========================================================================
    # Predict Mean and Variance
    # =========================================================================
    def __predict_mean_variance(self, timesteps, x_t, model_output):
        if timesteps[0].item() > 0:
            noise = torch.randn_like(x_t, 
                                     device = x_t.device, 
                                     dtype  = x_t.dtype)
        else: 
            noise = torch.full_like(x_t, 
                                    0, 
                                    device = x_t.device, 
                                    dtype  = x_t.dtype)

        predicted_noise, predicted_log_variance = model_output.chunk(2, dim=1)
        predicted_noise_parameters = self.__get_parameters_from_noise(
            x_t,
            predicted_noise,
            predicted_log_variance,
            timesteps)
        
        mean            = predicted_noise_parameters.mean
        std_deviation   = torch.sqrt(predicted_noise_parameters.variance)
        
        x_t_minus_one   = mean + noise * std_deviation

        return x_t_minus_one


    def __compute_vlb_coefficients(self):
        """ Formula 7 DDPM """
        # Shift alpha_bar to create previous timestep
        alpha_bar_prev  = torch.cat([torch.tensor([1.]), self.alpha_bars[:-1]])
        beta_prod       = 1 - self.alpha_bars
        beta_prod_prev  = 1 - alpha_bar_prev

        x_zero_coefs    = ((torch.sqrt(alpha_bar_prev) * self.betas) 
                            / beta_prod)
        
        x_t_coefs       = ((torch.sqrt(self.alphas) * beta_prod_prev) 
                            / beta_prod) 

        variance_t      = (beta_prod_prev / beta_prod) * self.betas
        variance_t      = torch.clamp(variance_t, min=1e-20) 
        log_variance_t  = torch.log(variance_t) 

        mean_x_zero_coefs   = torch.sqrt(1.0 / self.alpha_bars)
        mean_x_t_coefs      = torch.sqrt(1.0 / self.alpha_bars - 1)

        return (x_zero_coefs, 
                x_t_coefs, 
                variance_t,
                log_variance_t,
                mean_x_zero_coefs, 
                mean_x_t_coefs)
    

    # =========================================================================
    # Noise and Noise schedules
    # =========================================================================
    def __add_noise(self, original_samples, timesteps):
        """ Formula 12 DDPM """
        sqrt_alpha_bars             = torch.sqrt(self.alpha_bars[timesteps]
                                                 ).flatten()
        sqrt_one_minus_alpha_bars   = torch.sqrt(1-self.alpha_bars[timesteps]
                                                 ).flatten()

        while len(sqrt_alpha_bars.shape) < len(original_samples.shape):
            sqrt_alpha_bars = sqrt_alpha_bars.unsqueeze(-1)

        while len(sqrt_one_minus_alpha_bars.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.unsqueeze(-1)

        noise = torch.randn(original_samples.shape,
                            generator=self.generator, 
                            device=original_samples.device)
    
        noisy_samples = (sqrt_alpha_bars             * original_samples 
                         + sqrt_one_minus_alpha_bars * noise)

        return noisy_samples, noise
    

    def __create_beta_schedule(self,
                             amount_timesteps,
                             beta_start,
                             beta_end, 
                             schedule = BetaSchedules.COSINE):
        match schedule:
            case BetaSchedules.LINEAR:
                return torch.linspace(beta_start ** 0.5, 
                                      beta_end ** 0.5,
                                      amount_timesteps, 
                                      dtype=torch.float32) ** 2
            # Only use cosine when learning the variance
            case BetaSchedules.COSINE:
                # Improved DDPM formula 17
                s = 0.01
                pi_over_two         = np.pi / 2
                timesteps           = torch.arange(end=amount_timesteps + 1,
                                                   dtype=torch.float32)
                timestep_over_time  = timesteps / amount_timesteps
                time_over_offset    = (timestep_over_time + s) / (1 + s) 

                f_t = torch.pow(torch.cos(time_over_offset * pi_over_two), 2)

                alpha_bar_t = f_t / f_t[0]
                betas       = 1 - (alpha_bar_t[1:] / alpha_bar_t[:-1])

                return torch.clip(betas, min=0, max=0.999)

    
    # =========================================================================
    # Timesteps schedule
    # =========================================================================
    def __sample_from_timesteps(self, amount_steps, amount_samples):
        return torch.randint(low=1, 
                             high=amount_steps,
                             size=(amount_samples,))
    
    def __get_sampling_timesteps(self, amount_sample_steps):
        step_size   = self.amount_training_steps // amount_sample_steps
        timesteps   = np.arange(start  = 0,
                                stop   = self.amount_training_steps,
                                step   = step_size)[::-1]
        return timesteps
    
    # =========================================================================
    # Util
    # =========================================================================
    def get_state_dict_to_save(self):
        if self.can_save:
            state_dict = self.state_dict()
            state_dict.pop("latent_model")
            return state_dict
        return None
    
    def apply_ema(self):
        self.ema_model.apply_to_model(self.model) 
        self.ema_model  = None
        self.can_save   = False

class NoiseParameters():
    def __init__(self, mean, variance, log_variance):
        self.mean           = mean
        self.variance       = variance
        self.log_variance   = log_variance

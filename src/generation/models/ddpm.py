import constants
import enum
import torch
import util

import numpy                        as np
import torch.nn                     as nn
import torch.nn.functional          as f
import torch.optim                  as optim

from debug                          import Printer, LogLevel
from generation.models.vae          import AutoEncoderFactory
from generation.modules.diffusion   import Diffusion
from generation.modules.ema         import EMA
from tqdm                           import tqdm


class BetaSchedules(enum.Enum):
    LINEAR  = "Linear"
    COSINE  = "Cosine"

class PredictionType(enum.Enum):
    MEAN_VARIANCE   = "Mean_Variance"
    EPSILON_SIMPLE  = "Epsilon_Simple"

class DDPM(nn.Module):
    def __init__(self, 
                 configuration,
                 generator=torch.Generator(device=util.get_device()), 
                 beta_start=0.00085,
                 beta_end = 0.0120, 
                 amount_classes=0):
        
        super().__init__()
        self.model_family   = "diffusion"
        self.name           = configuration["name"]
        self.device         = util.get_device()
        self.generator      = generator
        self.amount_classes = amount_classes

        self.optimizers     = self.__get_optimizers()

        self.amount_training_steps     = configuration["training_steps"]
        self.amount_sample_steps       = configuration["sample_steps"]
        self.sample_steps = self.__get_sampling_timesteps(
            self.amount_sample_steps)
        
        self.input_shape    = (configuration["input_num_channels"], 
                               configuration["input_resolution_x"],
                               configuration["input_resolution_y"])
        
        # LATENT ==============================================================
        self.is_latent          = configuration["Latent"]["active"]
        if self.is_latent:
            self.latent_model   = AutoEncoderFactory.create_auto_encoder(
                configuration["Latent"]["model"])
            
            if (configuration["Latent"]["pre_trained"] 
                and not util.load_model(self.latent_model)):
                
                Printer().print_log(
                    f"Model {self.latent_model.name} could not be loaded",
                    LogLevel.WARNING)    

        # DIFFUSION MODEL =====================================================               
        self.model          = Diffusion(configuration, amount_classes)

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
                                                BetaSchedules.LINEAR))

        self.register_buffer("alphas",          1.0 - self.betas)
        self.register_buffer("alpha_bars",      torch.cumprod(self.alphas, 0))

        mean_variance_coefficients = self.__compute_mean_variance_coefficients()
        self.register_buffer("x_zero_coefs",    mean_variance_coefficients[0])
        self.register_buffer("x_t_coefs",       mean_variance_coefficients[1])
        self.register_buffer("variance_t",      mean_variance_coefficients[2])

        epsilon_coefficients = self.__compute_epsilon_coefficients()
        self.register_buffer("one_over_sqrt_alphas",  epsilon_coefficients[0])
        self.register_buffer("epsilon_coefficient",   epsilon_coefficients[1])

    # =========================================================================
    # Sampling
    # =========================================================================
    def generate(self, label=None):
        if label is not None:
            label = torch.tensor([label], device=self.device)
        
        return self.sample(1, label)

    # TODO prediction Type and other enums as config param
    @torch.no_grad()
    def sample(self, 
               amount_samples, 
               control_signals, 
               prediction_type= PredictionType.EPSILON_SIMPLE):
        """ Algorithm 2 DDPM """

        if self.use_ema:
            self.ema_model.apply_to_model(self.model) 

        x = torch.randn((amount_samples,) + self.input_shape, 
                        device=self.device)

        for _, timestep in tqdm(enumerate(self.sample_steps),
                             total = self.amount_sample_steps,
                             desc = "Generating Image"):
            
            
            timesteps = torch.tensor([timestep] * amount_samples, 
                                    device=self.device)
            
            model_output = self.model(x, control_signals, timesteps)
            
            if (self.use_classifier_free_guidance and control_signals != None):  
                unconditional_model_output = self.model(x, None, timesteps)
                model_output = torch.lerp(unconditional_model_output, 
                                          model_output,
                                          self.cfg_weight)

            match prediction_type:
                # As proposed by algorithm 2
                case PredictionType.EPSILON_SIMPLE:
                    x = self.__predict_epsilon_simple(timesteps, 
                                                      x, 
                                                      model_output)
                case PredictionType.MEAN_VARIANCE:
                    x = self.__predict_mean_variance(timesteps, 
                                                     x, 
                                                     model_output)

        if self.is_latent:
            self.latent_model.eval()
            x = self.latent_model.decode(x)

        return x
    
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
        
        if self.is_latent:
            self.latent_model.eval()
            with torch.no_grad():
                inputs = self.latent_model.encode(inputs).latents

        inputs_shape    = inputs.shape
        timesteps       = self.__sample_from_timesteps(
            self.amount_training_steps, inputs_shape[0])
        
        noised_images, noise = self.__add_noise(inputs, timesteps)

        if (self.use_classifier_free_guidance 
            and np.random.random() < self.no_class_probability):
            labels = None         

        predicted_noise = self.model(noised_images, labels, timesteps)

        loss = f.mse_loss(noise, predicted_noise)

        return loss, None
    
    def on_training_step_completed(self):
        if self.use_ema:
            self.ema_model.ema_step(self.model)

    def __get_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=4.5e-6)
        return [optimizer]

    # =========================================================================
    # Predict Epsilon
    # =========================================================================
    def __predict_epsilon_simple(self, timestep, x_t, model_output):
        """ Algorithm 2 DDPM """
        noise = 0
        if timestep > 0:
            noise = torch.randn(model_output.shape, 
                                generator=self.generator, 
                                device=model_output.device, 
                                dtype=model_output.dtype)
        
        return (self.one_over_sqrt_alphas[timestep] 
                * (x_t - self.epsilon_coefficient[timestep] * model_output) 
                + torch.sqrt(self.betas[timestep]) * noise)


    # =========================================================================
    # Predict Mean and Variance
    # =========================================================================
    def __predict_mean_variance(self, timestep, x_t, model_output):
        alpha_bar_t = self.alpha_bars[timestep]
        beta_prod_t = 1 - alpha_bar_t 

        # Formula 15
        pred_x_zero = ((x_t - torch.sqrt(beta_prod_t) * model_output) 
                       / torch.sqrt(alpha_bar_t))    

        pred_mean_t = (pred_x_zero * self.pred_x_zero_coef 
                       + x_t       * self.pred_x_t_coef)
         
        variance = 0
        if timestep > 0: 
            noise = torch.randn(model_output.shape, 
                                generator=self.generator, 
                                device=model_output.device, 
                                dtype=model_output.device)
            
            # can extract this
            pred_variance_t = torch.clamp(self.pred_variance_t, min=1e-20) 
            variance        = torch.sqrt(pred_variance_t) * noise
            

        x_t_prev = pred_mean_t + variance
        return x_t_prev
    
    def __compute_epsilon_coefficients(self):
        one_over_sqrt_alphas    = 1. / torch.sqrt(self.alphas)
        epsilon_coefficient     = ((1. - self.alphas) 
                                   / torch.sqrt(1. - self.alpha_bars))
        
        return one_over_sqrt_alphas, epsilon_coefficient
            
    def __compute_mean_variance_coefficients(self):
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

        return x_zero_coefs, x_t_coefs, variance_t
    


    # =========================================================================
    # Noise and Noise schedules
    # =========================================================================
    def __add_noise(self, original_samples, timesteps):
        """ Formula 12 DDPM """
        alpha_bars   = self.alpha_bars.to(device  = original_samples.device,
                                          dtype   = original_samples.dtype)
        
        timesteps   = timesteps.to(device=original_samples.device)    

        sqrt_alpha_bars             = torch.sqrt(alpha_bars[timesteps]
                                                 ).flatten()
        sqrt_one_minus_alpha_bars   = torch.sqrt(1-alpha_bars[timesteps]
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

                # 0.999 causes the magnitude of the image to be completely
                # wrong, why???
                # Some github issues suggest clipping during reverse is needed
                # Seems to be what SD does
                # Some say this doesnt work with predicting epsilon
                return torch.clip(betas, min=0, max=0.999)

    
    # =========================================================================
    # Timesteps schedule
    # =========================================================================
    def __sample_from_timesteps(self, amount_steps, amount_samples):
        return torch.randint(low=1, 
                             high=amount_steps,
                             size=(amount_samples,),
                             device=self.device)
    
    def __get_sampling_timesteps(self, amount_sample_steps):
        step_size   = self.amount_training_steps // amount_sample_steps
        timesteps   = np.arange(start  = 0,
                                stop   = self.amount_training_steps,
                                step   = step_size)[::-1]
        return timesteps
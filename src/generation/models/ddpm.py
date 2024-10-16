import enum
import torch
import util

import numpy                        as np
import torch.nn                     as nn
import torch.nn.functional          as f

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
                 num_training_steps=1000, 
                 beta_start=0.00085,
                 beta_end = 0.0120, 
                 amount_classes=0):
        super().__init__()
        self.name           = configuration["name"]
        self.use_ema        = configuration["use_ema"]
        self.device         = util.get_device()
        self.generator      = generator
        self.amount_classes = amount_classes
        
        self.input_shape    = (configuration["input_num_channels"], 
                               configuration["input_resolution_x"],
                               configuration["input_resolution_y"])
        self.latent         = configuration["latent"]

        if self.latent:
            self.latent_model   = AutoEncoderFactory.create_auto_encoder(
                configuration["latent_model"])
        self.model          = Diffusion(configuration, amount_classes)

        self.ema_model      = None
        if self.use_ema:
            self.ema_model  = EMA(self.model)

        self.betas = self.__create_beta_schedule(num_training_steps, 
                                                 beta_start,
                                                 beta_end,
                                                 BetaSchedules.COSINE)
                
        self.alphas             = 1.0 - self.betas
        self.alpha_bars         = torch.cumprod(self.alphas, 0)
        self.num_training_steps = num_training_steps


        mean_variance_coefficients = self.__compute_mean_variance_coefficients()
        self.x_zero_coefs   = mean_variance_coefficients[0]
        self.x_t_coefs      = mean_variance_coefficients[1] 
        self.variance_t     = mean_variance_coefficients[2]

        epsilon_coefficients        = self.__compute_epsilon_coefficients()
        self.one_over_sqrt_alphas   = epsilon_coefficients[0]
        self.epsilon_coefficient    = epsilon_coefficients[1]


    
    def to_device(self, device):
        return self.to(device=device)

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.device                 = device

            self.betas                  = self.betas.to(device) 
            self.alphas                 = self.alphas.to(device) 
            self.alpha_bars             = self.alpha_bars.to(device) 
            self.x_zero_coefs           = self.x_zero_coefs.to(device)
            self.x_t_coefs              = self.x_t_coefs.to(device)
            self.variance_t             = self.variance_t.to(device)
            self.one_over_sqrt_alphas   = self.one_over_sqrt_alphas.to(device)
            self.epsilon_coefficient    = self.epsilon_coefficient.to(device)
                    
        return super().to(device=device, *args, **kwargs)

    def set_inference_timesteps():
        pass

    def generate(self, label=None):
        if label is not None:
            label = torch.tensor([label], device=self.device)
        return self.sample(1, label)

    @torch.no_grad()
    def sample(self, amount_samples, control_signals):
        """ Algorithm 2 DDPM """
        classifier_free_guidance_weight = 3.0

        x = torch.randn((amount_samples,) + self.input_shape,
                        device=self.device)
        
        prediction_type = PredictionType.EPSILON_SIMPLE

        for timestep in tqdm(reversed(range(self.num_training_steps)),
                             total = self.num_training_steps,
                             desc = "Generating Image"):
            timesteps = torch.tensor([timestep] * amount_samples, 
                                    device=self.device)
            
            model_output = self.model(x, control_signals, timesteps)
            
            if classifier_free_guidance_weight and control_signals: 
                unconditional_model_output = self.model(x, None, timesteps)
                model_output = torch.lerp(
                    unconditional_model_output, 
                    model_output,
                    classifier_free_guidance_weight
                )

            match prediction_type:
                # As proposed by algorithm 2
                case PredictionType.EPSILON_SIMPLE:
                    x = self.__predict_epsilon_simple(timesteps, 
                                                      x, 
                                                      model_output)
                        

                # Without the simplification, this is what SD does 
                case PredictionType.MEAN_VARIANCE:
                    x = self.__predict_mean_variance(timesteps, x, model_output)
            
        return x
    
    def training_step(self, inputs, labels):
        inputs_shape        = inputs.shape
        timesteps           = self.__sample_timesteps(self.num_training_steps, 
                                                      inputs_shape[0])
        noised_images, noise = self.__add_noise(inputs, timesteps)

        # Classifier Free Guidance
        if np.random.random() < 0.1:
            labels = None         

        predicted_noise = self.model(noised_images, labels, timesteps)

        loss = f.mse_loss(noise, predicted_noise, reduction="sum")

        return loss, None
    
    def on_training_step_completed(self):
        if self.use_ema:
            self.ema_model.ema_step(self.model)
    
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
                return torch.clip(betas, min=0, max=0.999)
    
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
    
    
    def __add_noise(self, original_samples, timesteps):
        """ Formula 4 DDPM """
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

    def __sample_timesteps(self, amount_steps, amount_samples):
        return torch.randint(low=1, 
                             high=amount_steps,
                             size=(amount_samples,),
                             device=self.device)

    def compute_loss():
        pass
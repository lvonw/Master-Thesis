import torch
import numpy as np

class DDPM():
    def __init__(self, 
                 generator=torch.Generator(), 
                 num_training_steps=1000, 
                 beta_start=0.00085,
                 beta_end = 0.0120):
        
        # TODO different schedules
        self.betas = torch.linspace(beta_start ** 0.5, 
                                    beta_end ** 0.5,
                                    num_training_steps, 
                                    dtype=torch.float32) ** 2
        
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch(self.alphas, 0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps

    def set_inference_timesteps():
        pass

    def add_noise(self, original_samples, alpha_bar, timesteps):
        alpha_cumprod = self.alpha_cumprod.to(device=original_samples.device,
                                              dtype=original_samples.dtype)
        
        timesteps  = timesteps.to(device=original_samples.device)    

        sqrt_alpha_cumprod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()

        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)

        sqrt_one_minus_alpha_cumprod = (1-alpha_cumprod[timesteps]) ** 0.5 # sigma
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()

        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, 
                            generator=self.generator, 
                            device=original_samples.device)
        
        noisy_samples = (sqrt_alpha_cumprod * original_samples) + (sqrt_one_minus_alpha_cumprod * noise)

        return noisy_samples
    
    def __get_previous_timestep(self, timestep):
        #something something noise schedule
        return 
    
    def denoising_step(self, timestep, pred_x_t, model_output, predicted_noise):
        previous_timestep = self.__get_previous_timestep(timestep)

        alpha_t = self.alphas(timestep)

        alpha_bar_t = self.alpha_bar[timestep]
        alpha_bar_t_minus_1 = self.alpha_bar[previous_timestep]

        beta_t = 1 - alpha_bar_t / alpha_bar_t_minus_1

        beta_prod_t = 1 - alpha_bar_t 
        beta_prod_t_prev = 1 - alpha_bar_t_minus_1

        current_alpha_t = alpha_bar_t / alpha_bar_t_minus_1
        current_beta_t = 1 - current_alpha_t

        pred_x_zero = (pred_x_t - torch.sqrt(beta_prod_t) * predicted_noise) / torch.sqrt(alpha_bar_t)    

        pred_x_zero_coef = (torch.sqrt(alpha_bar_t_minus_1)*beta_prod_t)/(beta_prod_t)
        pred_x_t_coef = (torch.sqrt(alpha_t) * (beta_prod_t_prev))/(beta_prod_t)  
        pred_mean_t = pred_x_zero * pred_x_zero_coef + pred_x_t * pred_x_t_coef

        pred_variance_t = ((beta_prod_t_prev)/(beta_prod_t))*beta_t
        pred_variance_t = torch.clamp(pred_variance_t, min=1e-20) 
        pred_sigma_t    = torch.sqrt(pred_variance_t)

        if timestep > 0: 
            device = model_output.device 
            noise = torch.randn(model_output.shape, 
                                generator=self.generator, 
                                device=device, 
                                dtype=model_output.device)

        pred_x_t_minus_1 = pred_mean_t + pred_sigma_t * noise

    def train():
        pass

    def compute_loss():
        pass
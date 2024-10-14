import enum
import torch
import torch.nn                 as nn
import torch.nn.functional      as f
import util

from generation.models.vae          import AutoEncoderFactory
from generation.modules.unet        import UNETFactory
from generation.modules.diffusion   import Diffusion
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
                 beta_end = 0.0120):
        super().__init__()
        self.name           = configuration["name"]
        self.device         = util.get_device()
        self.generator      = generator
        
        self.input_shape    = (configuration["input_num_channels"], 
                               configuration["input_resolution_x"],
                               configuration["input_resolution_y"])
        self.latent         = configuration["latent"]

        if self.latent:
            self.latent_model   = AutoEncoderFactory.create_auto_encoder(
                configuration["latent_model"])
        self.model          = Diffusion(configuration)
            
        self.betas = self.__create_beta_schedule(num_training_steps, 
                                                 beta_start,
                                                 beta_end,
                                                 BetaSchedules.LINEAR)
                
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

    def generate(self):
        return self.sample(1)

    @torch.no_grad()
    def sample(self, amount_samples):
        """ Algorithm 2 DDPM """
        x = torch.randn((amount_samples,) + self.input_shape,
                        device=self.device)
        control_signal = None
        prediction_type = PredictionType.EPSILON_SIMPLE

        for timestep in tqdm(reversed(range(self.num_training_steps)),
                             total = self.num_training_steps,
                             desc = "Generating Image"):
            timesteps = torch.tensor([timestep] * amount_samples, 
                                    device=self.device)
            
            model_output = self.model(x, control_signal, timesteps)

            match prediction_type:
                # As proposed by algorithm 2
                case PredictionType.EPSILON_SIMPLE:
                    x = self.__predict_epsilon_simple(timesteps, x, model_output)
                # Without the simplification, this is what SD does 
                case PredictionType.MEAN_VARIANCE:
                    x = self.__predict_mean_variance(timesteps, x, model_output)
            
        return x
    
    def training_step(self, inputs, labels):
        inputs_shape = inputs.shape
        timesteps = self.__sample_timesteps(self.num_training_steps, 
                                            inputs_shape[0])
        noised_images, noise = self.__add_noise(inputs, timesteps)

        predicted_noise = self.model(noised_images, None, timesteps)

        loss = f.mse_loss(noise, predicted_noise, reduction="sum")

        return loss, None
    
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
                             schedule = BetaSchedules.LINEAR):
        match schedule:
            case BetaSchedules.LINEAR:
                return torch.linspace(beta_start ** 0.5, 
                                      beta_end ** 0.5,
                                      amount_timesteps, 
                                      dtype=torch.float32) ** 2
            # TODO
            case BetaSchedules.COSINE:
                return None
            
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
    
    def __compute_epsilon_coefficients(self):
        one_over_sqrt_alphas    = 1. / torch.sqrt(self.alphas)
        epsilon_coefficient     = ((1. - self.alphas) 
                                   / torch.sqrt(1. - self.alpha_bars))
        
        return one_over_sqrt_alphas, epsilon_coefficient


    
    # Do i really need to noise all images at the same step
    def __add_noise(self, original_samples, timesteps):
        """ Formula 4 DDPM """
        alpha_bars   = self.alpha_bars.to(device  = original_samples.device,
                                          dtype   = original_samples.dtype)
        
        timesteps   = timesteps.to(device=original_samples.device)    

        sqrt_alpha_bars             = torch.sqrt(alpha_bars[timesteps]
                                                 ).flatten()
        sqrt_one_minus_alpha_bars   = torch.sqrt(1-alpha_bars[timesteps]
                                                 ).flatten()

        # can probably just set this to len 4?
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
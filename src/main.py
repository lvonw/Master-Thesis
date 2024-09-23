import torch

from cli.cli                import CLI
from configuration          import Configuration
from data.dataset           import DatasetFactory
from torch.utils.data       import DataLoader
from tqdm                   import tqdm

def generate(condition, 
             input_image, 
             strength,  
             do_cfg, 
             cfg_scale, 
             device="cuda",
             sampler="ddpm",
             n_inference_steps=50,
             models={},
             seed=None):
    
    generator = torch.Generator()
    if seed:
        generator.manual_seed(seed)
    else:
        generator.seed() 

    latents_shape = (1, 4, 64, 64)

    if input_image:
        # get encoder
        # encoder = nn.Module

        # encode image properly
        # schedule noise
        pass
    else:
        latents = torch.randn(latents_shape, generator=generator, device=device)

    for i in range(123):
        # time embedding    
        # model_output = diffusion(model_input, context, time_embedding)

        # latents = sampler.step(timestep, latents, model_output) 
        pass

    # image = decoder(latents) 

    # transform prediciont back into image


def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32)/160)
    x  = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def main():
    config = Configuration()
    config.load_defaults()

    cli = CLI(config);

    config, should_quit = cli.cli_loop();

    if should_quit:
        quit()

    if config["Main"]["train"]:
        training_set, test_set = DatasetFactory.create_dataset(config["Data"])

        dataloader  = DataLoader(training_set, 
                                    batch_size=1, 
                                    shuffle=False,
                                    )
        
        for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
            #for i, data in enumerate(dataloader, 0):
            break
        

    # if config["Main"]["generate"]:
    #    generate(1,2,3,4,5)

if __name__ == "__main__":
    main()
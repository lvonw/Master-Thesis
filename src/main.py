import argparse
import constants
import torch
import os

from cli.cli                import CLI
from configuration          import Configuration
from data.dataset           import DatasetFactory
from torch.utils.data       import DataLoader
from tqdm                   import tqdm

from pipeline               import generate, training
from debug                  import Printer
from generation.models.vae  import AutoEncoderFactory, VariationalAutoEncoder
from util                   import get_device

import matplotlib.pyplot    as plt
import numpy                as np
import torch.nn             as nn

from generation.modules.encoder import Encoder, Decoder
from generation.modules.util_modules import (ResNetBlock, 
                                             Upsample, 
                                             Downsample,
                                             Normalize)
from generation.modules.attention   import AttentionBlock, SelfAttention


def prepare_arg_parser():
    parser = argparse.ArgumentParser(prog="Diffusion",
                                     description="Description")
    parser.add_argument("-cfg",
                        "--config",
                        dest="config", 
                        nargs=1,
                        type=str,
                        help="")
    parser.add_argument("-cli",
                        "--cli",
                        dest="cli", 
                        action="store_true",
                        help="")
    
    return parser


def main():
    parser      = prepare_arg_parser()
    arguments   = parser.parse_args()
    printer     = Printer()

    printer.print_log("Loading Configuration...")
    config = Configuration()
    if arguments.config:
        config.load(arguments.config)
    else:
        config.load_defaults()
    printer.print_log("Finished.")

    if arguments.cli:
        config.load_usages()
        cli = CLI(config);
        config, should_quit = cli.cli_loop();
        if should_quit:
            quit()

    printer.print_log("Creating VAE...")
    VAE = AutoEncoderFactory.create_auto_encoder(
        config["Variational_Auto_Encoder"])
    printer.print_log("Finished.")
    total_params = sum(p.numel() for p in VAE.parameters())
    printer.print_log(f"Total amount of parameters: {total_params}")
    printer.print_log(f"Using device: {get_device()}")


    cpu_core_num = os.cpu_count()
    printer.print_log(f"Core count: {cpu_core_num}")


    # printer.print_log("Loading state dict...")

    # torch.serialization.add_safe_globals([getattr,
    #                                       VariationalAutoEncoder,
    #                                       set,
    #                                       Encoder,
    #                                       Decoder,
    #                                       nn.Conv2d,
    #                                       nn.ModuleList,
    #                                       ResNetBlock, 
    #                                       Upsample, 
    #                                       Downsample,
    #                                       Normalize,
    #                                       nn.GroupNorm,
    #                                       nn.Linear,
    #                                       nn.Identity,
    #                                       AttentionBlock,
    #                                       SelfAttention,
    #                                       nn.SiLU])
    # asd = torch.load(constants.MODEL_PATH_TEST, 
    #                                weights_only=False)
    
    # VAE.load_state_dict(asd)
    # printer.print_log("Finished.")
    
    
    # TODO just for testing 
    if config["Main"]["generate"]:
        VAE.eval()
        with torch.no_grad():
            sample = VAE.generate()

            
    if config["Main"]["train"]:
        printer.print_log("Creating Dataset...")
        training_set, test_set = DatasetFactory.create_dataset(config["Data"])
        printer.print_log("Finished.")

        data_loader_generator = torch.Generator()
        data_loader_generator.manual_seed(constants.DATALOADER_SEED)

        training_dataloader  = DataLoader(training_set, 
                                    batch_size=32, 
                                    shuffle=True,
                                    generator=data_loader_generator,
                                    num_workers=cpu_core_num // 2,
                                    pin_memory=True, 
                                    pin_memory_device=str(get_device()))
        
        training.train(VAE, training_dataloader, config["Training"])




if __name__ == "__main__":
    main()
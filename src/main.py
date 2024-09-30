import argparse
import constants
import torch
import os

from cli.cli                import CLI
from configuration          import Configuration
from data.dataset           import DatasetFactory
from tqdm                   import tqdm

from pipeline               import generate, training
from debug                  import Printer
from generation.models.vae  import AutoEncoderFactory, VariationalAutoEncoder
from util                   import get_device

import matplotlib.pyplot    as plt
import numpy                as np
import torch.nn             as nn

from torchvision.datasets   import MNIST
from torchvision            import transforms



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

    printer.print_log("Creating Model...")
    model = AutoEncoderFactory.create_auto_encoder(
        config["Model"])
    printer.print_log("Finished.")

    total_params = sum(p.numel() for p in model.parameters())
    printer.print_log(f"Total amount of parameters: {total_params}")
    printer.print_log(f"Using device: {get_device()}")
    cpu_core_num = os.cpu_count()
    printer.print_log(f"Core count: {cpu_core_num}")


    printer.print_log("Loading state dict...")
    model.load_state_dict(torch.load(constants.MODEL_PATH_TEST, 
                                   weights_only=False))
    printer.print_log("Finished.")
    
    # TODO just for testing 
    if config["Main"]["generate"]:
        model.eval()
        with torch.no_grad():
            sample = model.generate()

            
    if config["Main"]["train"]:
        # DATA_PATH_CLIMATE   = os.path.join(constants.DATA_PATH_MASTER)
        mnist_path = constants.DATA_PATH_MASTER
        training_set = MNIST(root=mnist_path, 
                              train=True, 
                              download=False, 
                              transform=transforms.ToTensor())

        validation_set = MNIST(root=mnist_path, 
                             train=False, 
                             download=False, 
                             transform=transforms.ToTensor())



        # printer.print_log("Creating Dataset...")
        # training_set, validation_set = DatasetFactory.create_dataset(
        #     config["Data"])
        # printer.print_log("Finished.")
        
        training.train(model, training_set, validation_set, config["Training"])




if __name__ == "__main__":
    main()
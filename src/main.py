import argparse
import constants
import os
import torch
import util                   

from cli.cli                import CLI
from configuration          import Configuration
from data.dataset           import DatasetFactory

from pipeline               import generate, training
from debug                  import Printer
from generation.models.vae  import AutoEncoderFactory
from generation.models.ddpm import DDPM
from data.data_util         import DataVisualizer

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

# Helper to see if my models work
def get_mnist():
    training_set = MNIST(root=constants.DATA_PATH_MASTER, 
                         train=True, 
                         download=False, 
                         transform=transforms.ToTensor())

    validation_set = MNIST(root=constants.DATA_PATH_MASTER, 
                           train=False, 
                           download=False, 
                           transform=transforms.ToTensor())
    

    return training_set, validation_set


def main():
    parser      = prepare_arg_parser()
    arguments   = parser.parse_args()
    printer     = Printer()

    # =========================================================================
    # Configuration
    # =========================================================================
    printer.print_log("Loading Configuration...")
    config = Configuration()
    if arguments.config:
        config.load(arguments.config)
    else:
        config.load_defaults()
    printer.print_log("Finished.")

    # =========================================================================
    # CLI
    # =========================================================================
    if arguments.cli:
        config.load_usages()
        cli = CLI(config);
        config, should_quit = cli.cli_loop();
        if should_quit:
            quit()

    # =========================================================================
    # Dataset
    # =========================================================================
    needs_dataset = config["Main"]["train"] or config["Main"]["test"]
    amount_classes = 16
    if needs_dataset:
        printer.print_log("Creating Dataset...")
        if config["Main"]["use_MNIST"]:
            printer.print_log("Using MNIST")
            training_set, validation_set = get_mnist()
            amount_classes = 10
        else:
            printer.print_log("Using DEMs")
            complete_dataset, amount_classes = DatasetFactory.create_dataset(
                config["Data"])
        printer.print_log("Finished.")

    # =========================================================================
    # Model
    # =========================================================================
    model_name = config["Model"]["name"]
    printer.print_log(f"Creating Model {model_name}...")
    # model = DDPM(config["Model"], amount_classes=amount_classes)
    model = AutoEncoderFactory.create_auto_encoder(config["Model"])
    printer.print_log("Finished.")

    if config["Main"]["load_model"]:
        printer.print_log("Loading state dict...")
        starting_epoch_idx = util.load_checkpoint(model)
        if not starting_epoch_idx:
            printer.print_log(f"Model {model.name} could not be loaded",
                              constants.LogLevel.WARNING)
        else:    
            printer.print_log("Finished.")

    # =========================================================================
    # Stats
    # =========================================================================
    total_params = sum(p.numel() for p in model.parameters())
    printer.print_log(f"Total amount of parameters: {total_params:,}")
    printer.print_log(f"Using device: {util.get_device()}")
    printer.print_log(f"Core count: {os.cpu_count()}")
    printer.print_log(f"Amount Classes: {amount_classes}")

    # =========================================================================
    # Training
    # =========================================================================
    if config["Main"]["train"]:        
        training.train(model, 
                       complete_dataset, 
                       config["Training"], 
                       starting_epoch=starting_epoch_idx)

    # =========================================================================
    # Generation
    # =========================================================================
    if config["Main"]["generate"]:
        generate.generate(model)

if __name__ == "__main__":
    main()
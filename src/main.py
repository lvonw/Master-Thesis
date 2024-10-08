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
from data.data_util         import DataVisualizer

from torchvision.datasets   import MNIST
from torchvision            import transforms

from generation.modules.diffusion    import UNET



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
    # model = AutoEncoderFactory.create_auto_encoder(config["Model"])
    model = UNET(2)
    printer.print_log("Finished.")


    total_params = sum(p.numel() for p in model.parameters())
    printer.print_log(f"Total amount of parameters: {total_params:,}")
    printer.print_log(f"Using device: {util.get_device()}")
    printer.print_log(f"Core count: { os.cpu_count()}")

    if config["Main"]["load_model"]:
        printer.print_log("Loading state dict...")
        if not util.load_model(model):
            printer.print_log(f"Model {model.name} could not be loaded",
                              constants.LogLevel.WARNING)
        else:    
            printer.print_log("Finished.")
    
    if config["Main"]["train"]:
        printer.print_log("Creating Dataset...")
        if config["Main"]["use_MNIST"]:
            printer.print_log("Using MNIST")
            training_set, validation_set = get_mnist()
        else:
            printer.print_log("Using DEMs")
            training_set, validation_set = DatasetFactory.create_dataset(
                config["Data"])
        printer.print_log("Finished.")
        
        training.train(model, training_set, validation_set, config["Training"])
    
    # TODO just for testing 
    if config["Main"]["generate"]:
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                sample = model.generate()
                DataVisualizer.create_image_tensor_figure(sample)




if __name__ == "__main__":
    main()
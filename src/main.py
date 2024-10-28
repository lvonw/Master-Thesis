import argparse
import constants
import os
import torch
import util                   

from cli.cli                import CLI
from configuration          import Configuration
from data.dataset           import DatasetFactory

from pipeline               import generate, training
from debug                  import Printer, LogLevel
from generation.models.vae  import AutoEncoderFactory
from generation.models.ddpm import DDPM
from data.data_util         import DataVisualizer

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

    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

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
    needs_dataset   = config["Main"]["train"] or config["Main"]["test"]
    amount_classes  = [16]
    if needs_dataset:
        printer.print_log("Loading Dataset...")
        dataset_wrapper = DatasetFactory.create_dataset(config["Data"])
        amount_classes = dataset_wrapper.amount_classes
        printer.print_log("Finished.")

    # =========================================================================
    # Model
    # =========================================================================
    model_name = config["Model"]["name"]
    
    # Initialize Model ========================================================
    printer.print_log(f"Creating Model {model_name}...")
    try:
        model = DDPM(config["Model"], amount_classes=amount_classes)
    except TypeError as e:
       model = AutoEncoderFactory.create_auto_encoder(config["Model"])

    printer.print_log("Finished.")
    
    # Load Checkpoint =========================================================
    starting_epoch = 0
    if config["Main"]["load_model"]:
        printer.print_log("Loading state dict...")
        starting_epoch = util.load_checkpoint(model)
        
        if not starting_epoch:
            printer.print_log(f"Model {model.name} could not be loaded",
                              LogLevel.WARNING)
            
        printer.print_log(f"Finished, starting at epoch: {starting_epoch}.")

    # =========================================================================
    # Stats
    # =========================================================================
    total_params = sum(p.numel() for p in model.parameters())
    printer.print_log(f"Total amount of parameters: {total_params:,}")
    estimated_vram = 4. * total_params / 1000000000.
    printer.print_log(f"Estimated VRAM: {estimated_vram:,.2f} GB")
    printer.print_log(f"Using device: {util.get_device()}")
    printer.print_log(f"Core count: {os.cpu_count()}")
    printer.print_log(f"Amount Classes: {amount_classes}")

    # =========================================================================
    # Training
    # =========================================================================
    if config["Main"]["train"]:        
        training.train(model, 
                       dataset_wrapper, 
                       config["Training"], 
                       starting_epoch=starting_epoch)

    # =========================================================================
    # Generation
    # =========================================================================
    if config["Main"]["generate"]:
        generate.generate(model)

if __name__ == "__main__":
    main()
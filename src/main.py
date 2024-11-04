import argparse
import constants
import os
import platform
import torch
import util                   

import torch.distributed            as distributed 

from cli.cli                        import CLI
from configuration                  import Configuration
from data.dataset                   import DatasetFactory
from debug                          import Printer, LogLevel
from generation.models.vae          import AutoEncoderFactory
from generation.models.ddpm         import DDPM
from pipeline                       import generate, training
from torch.nn.parallel              import DistributedDataParallel  as DDP

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
    
    parser.add_argument("--distributed", 
                        dest="distributed", 
                        action="store_true")

    return parser

def __get_backend():
    system = platform.system()
    if system == "Windows":
        return "GLOO"
    elif system == "Linux":
        return "nccl"


def main():
    parser      = prepare_arg_parser()
    arguments   = parser.parse_args()
    printer     = Printer()
    # TODO possibly try to get this to work
    share_data  = False

    # Distribution ============================================================    
    is_distributed  = arguments.distributed and torch.cuda.device_count() > 1
    global_rank     = 0
    local_rank      = 0
    
    if is_distributed: 
        global_rank  = int(os.environ["RANK"])
        local_rank   = int(os.environ["LOCAL_RANK"])    

        printer.rank = local_rank
        distributed.scatter_object_list

        backend = __get_backend()
        distributed.init_process_group(backend)
        torch.cuda.set_device(local_rank)

        printer.print_log(f"Global Rank: {global_rank}")
        printer.print_log(f"Local Rank: {local_rank}")
        printer.print_log(f"Using backend: {backend}")
    else: 
        printer.print_log("Running on single Machine")

    # =========================================================================
    # Configuration
    # =========================================================================
    printer.print_log("Loading Configuration...")
    config = Configuration()
    if arguments.config:
        config.load(arguments.config)
    else:
        config.load_defaults()

    printer.print_only_rank_0 = config["Debug"]["print_only_rank_zero"]
    printer.print_log("Finished.")

    # =========================================================================
    # CLI
    # =========================================================================
    if arguments.cli and global_rank == 0:
        config.load_usages()
        cli = CLI(config);
        config, should_quit = cli.cli_loop()
        if should_quit:
            quit()

    # =========================================================================
    # Dataset
    # =========================================================================
    needs_dataset   = config["Main"]["train"] or config["Main"]["test"]
    if needs_dataset:
        printer.print_log("Loading Dataset...")
        if local_rank == 0 or not share_data:
            dataset_wrapper = DatasetFactory.create_dataset(config["Data"], 
                                                            prepare=True)
            if is_distributed and share_data: 
                distributed.barrier()
                printer.print_log(f"Lifting barrier")
        else:
            printer.print_log(f"Waiting for barrier to be lifted") 
            distributed.barrier()
            printer.print_log(f"Barrier was lifted, continuing") 
            dataset_wrapper = DatasetFactory.create_dataset(config["Data"])

        amount_classes = dataset_wrapper.amount_classes
        printer.print_log("Finished.")
    else:
        # TODO this should really be done differently, probably another static
        # function in datasetfactory that takes the config and determines the
        # necessary constants
        amount_classes  = [constants.LABEL_AMOUNT_GTC,
                       constants.LABEL_AMOUNT_CLIMATE]

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

    model.to(util.get_device())
    if is_distributed:
        model = DDP(model, 
                    device_ids=[local_rank], 
                    find_unused_parameters=True)        
    
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
        if is_distributed:
            training.train(model,
                           model.module, 
                           dataset_wrapper, 
                           config["Training"], 
                           starting_epoch,
                           is_distributed       = True,
                           global_rank          = global_rank,
                           local_rank           = local_rank,
                           local_amount         = torch.cuda.device_count(),
                           global_amount        = torch.cuda.device_count())
        else: 
            training.train(model,
                           model, 
                           dataset_wrapper, 
                           config["Training"], 
                           starting_epoch)
    
    # =========================================================================
    # Generation
    # =========================================================================
    if config["Main"]["generate"]:
        do_img2img = config["Main"]["img2img"]
        if do_img2img:
            generate.generate(model, 4, 10, config["Main"]["test_image"])
        else:
            generate.generate(model, 4, 10)

    if is_distributed:
        distributed.destroy_process_group()

if __name__ == "__main__":
    main()

import argparse

from cli.cli                import CLI
from configuration          import Configuration
from data.dataset           import DatasetFactory
from torch.utils.data       import DataLoader
from tqdm                   import tqdm

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

    config = Configuration()
    if arguments.config:
        config.load(arguments.config)
    else:
        config.load_defaults()

    if arguments.cli:
        cli = CLI(config);
        config, should_quit = cli.cli_loop();
        if should_quit:
            quit()

    if config["Main"]["train"]:
        training_set, test_set = DatasetFactory.create_dataset(config["Data"])

        print(len(training_set))

        dataloader  = DataLoader(training_set, 
                                    batch_size=1, 
                                    shuffle=False)
        
        for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
            if i == 25: break



if __name__ == "__main__":
    main()
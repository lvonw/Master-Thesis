from cli                import CLI
from configuration      import Configuration
from dataset            import DatasetFactory
from torch.utils.data   import DataLoader
from tqdm               import tqdm

def main():
    config = Configuration()
    config.load_defaults()

    cli = CLI(config);
    config, should_quit = cli.cli_loop();

    if should_quit:
        quit()

    training_set, test_set = DatasetFactory.create_dataset(config["Data"])

    dataloader  = DataLoader(training_set, 
                                 batch_size=1, 
                                 shuffle=False,
                                 )
    


    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
    #for i, data in enumerate(dataloader, 0):
        continue
    

if __name__ == "__main__":
    main()
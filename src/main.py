from cli import CLI
from configuration import Configuration

def main():
    config = Configuration()
    config.load_defaults()
    cli = CLI(config);
    cli.cli_loop();
    

if __name__ == "__main__":
    main()
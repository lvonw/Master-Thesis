import json
import os
import parsing
import constants

class _MenuAction(parsing.Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "current_menu" not in kwargs:
            self.current_menu = None
        else:
            self.current_menu = kwargs["current_menu"]

    def __call__(self, argument, values):
        key = argument.name
        
        if issubclass(argument.argument_type, MasterMenu): 
            self.current_menu.sub_menu = values[0]
            pass     
        elif argument.argument_type is bool:
            self.current_menu.config_data[key] = not self.current_menu.config_data[key]
        else:
            self.current_menu.config_data[key] = values[0]

class MasterMenu():
    def __init__(self, config_dir, selected_config=constants.CONFIG_DEFAULT_FILE):
        self.config_dir = config_dir
        self.sub_menu = None
        
        self.config_data = self.load_config(selected_config)
        self.parser = self.prepare_parser()

    def get_load_options(self):
        files = []

        for file in os.listdir(self.config_dir):
            if (os.path.isfile(os.path.join(self.config_dir, file)) 
                and file.endswith(".json")):
                
                files.append(file)
        
        return files

    def load_config(self, config_file):
        """Loads configuration data from the specified file."""
        config_path = os.path.join(self.config_dir, config_file)
    
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return json.load(file)
        elif config_file is not constants.CONFIG_DEFAULT_FILE:
            print(f"No config file found at {config_path}, loading defaults.")
            return self.load_defaults()
        else:
            print("Default file could not be loaded.")
            return {}

    def save_config(self, config_file):
        """Saves the current configuration data to the specified file."""
        with open(config_file, 'w') as file:
            json.dump(self.config_data, file, indent=4)

    def print(self):
        for key, value in self.config_data.items():
            print(f"{key}: {value}")   
        print()

    def get_config_entry(self, key):
        return self.config_data[key]

    def get_sub_menu(self):
        return self.sub_menu
    
    def load_defaults(self):
        return self.load_config(constants.CONFIG_DEFAULT_FILE)

    def prepare_parser(self):
        parser = parsing.Parser()

        for argument, value in self.config_data.items():
            arg_type = type(value)

            parser.add_argument(
                name    = argument,
                action  = _MenuAction,
                argument_type = arg_type,
                nargs   = 0 if arg_type is bool else 1,
                current_menu = self)
        
        return parser
        


class MainMenu(MasterMenu):
    def __init__(self, config_dir=constants.CONFIG_PATH_MAIN):
        super().__init__(config_dir)
        # self.config_data = {
        #     "print_this":    False,
        #     "print_that":    False,
        # }

class DebugMenu(MasterMenu):
    def __init__(self, config_dir=constants.CONFIG_PATH_DEBUG):
        super().__init__(config_dir)
        self.config_data = {
            "print_this":    False,
            "print_that":    False,
        }

class PTGModelMenu(MasterMenu):
    def __init__(self, config_dir=constants.CONFIG_PATH_HYPER_PARAMETERS):
        super().__init__(config_dir)
        self.config_data = {
            "batch_size":       False,
            "amount_epochs":    64,
        }

class HyperParameterMenu(MasterMenu):
    def __init__(self, config_dir=constants.CONFIG_PATH_HYPER_PARAMETERS):
        super().__init__(config_dir)
        self.config_data = {
            "batch_size":       False,
            "amount_epochs":    64,
        }
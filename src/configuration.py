import json
import os
import argparse
import constants

class _UpdateValueAction(argparse.Action):
    def __init__(self, 
                 option_strings, 
                 dest, 
                 type,
                 data = None, 
                 nargs = None, 
                 **kwargs):
        super().__init__(option_strings, dest, type=type, nargs=nargs, **kwargs)
        
        self.data = data

    def __call__(self, parser, namespace, values, option_string=None):
        key = self.dest
        
        if self.type is bool: 
           self.data[key] = not self.map_ref[key]     
        else:
            self.data[key] = values
    
class MasterConfig():
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.config = {}

    def get_load_options(self):
        files = []

        for file in os.listdir(self.config_dir):
            if (os.path.isfile(os.path.join(self.config_dir, file)) 
                and file.endswith(".json")):
                
                files.append(file)
        
        return files

    def load(self, config_file):
        """Loads configuration data from the specified file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as file:
                self.config_data = json.load(file)
        else:
            print(f"No config file found at {self.config_file}, loading defaults.")
            self.set_defaults()

    def save(self, config_file):
        """Saves the current configuration data to the specified file."""
        with open(config_file, 'w') as file:
            json.dump(self.config_data, file, indent=4)

    def print(self):
        for key, value in self.config_data.items():
            print(f"{key}: {value}")   
        print()

    def prepare_parser(self):
        parser = argparse.ArgumentParser(prog=constants.USAGE_PROGRAM_NAME,
                                     description=constants.USAGE_PROGRAM_DESC)

        for argument, value in self.config_data.items():
            arg_type = type(value)

            parser.add_argument(
                argument,
                dest    = argument, 
                nargs   = 0 if arg_type is bool else 1,
                help    = "",
                type    = arg_type,
                action  = _UpdateValueAction,
                data    = self.config_data)



class HyperParameterConfig(MasterConfig):
    def __init__(self, config_dir=constants.CONFIG_PATH_HYPER_PARAMETERS):
        super().__init__(config_dir)
        self.config_data = {
        "batch_size":       False,
        "amount_epochs":    64,
        }
    

class DebugConfig(MasterConfig):
    def __init__(self, config_dir=constants.CONFIG_PATH_DEBUG):
        self.config_data = {
            "print_this":    False,
            "print_that":    False,
        }
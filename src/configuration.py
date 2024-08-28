import json
import os
import constants

class Configuration():
    def __init__(self):
        self.name = "Default"
        self.sections = {}
        self.directory = constants.CONFIG_PATH

    def add_section(self, section_name, section_configuration):
        self.sections[section_name] = Section(section_name, section_configuration)

    def get_configuration(self, section_name, configuration_name):
        return self.sections[section_name][configuration_name]
    
    def __getitem__(self, section_name):
        return self.sections[section_name]
    
    def get_section(self, section_name):
        return self[section_name]

    def find_configuration(self, configuration_name):
        for _, section in self.sections.items():
            for name, configuration in section.items():
                if name == configuration_name:
                    return configuration

    def print_load_options(self):
        files = []

        for file in os.listdir(self.directory):
            if (os.path.isfile(os.path.join(self.directory, file)) 
                and file.endswith(".json")):
                
                files.append(file)
        
        print (files)

    def load_defaults(self):
        self.load(constants.CONFIG_DEFAULT_FILE)

    def load(self, config_file):
        """Loads configuration data from the specified file."""
        config_path = os.path.join(self.directory, config_file)
    
        if os.path.exists(config_path):
            loaded_data = {}
            with open(config_path, 'r') as file:
                loaded_data = json.load(file)
            
            for section_name, section_configuration in loaded_data.items():  
                self.add_section(section_name, section_configuration)

        elif config_file is not constants.CONFIG_DEFAULT_FILE:
            print(f"No config file found at {config_path}, loading defaults.")
            loaded_data = self.load_defaults()
        else:
            print("Default file could not be loaded.")
            self.print_load_options()
            return

    def save(self, config_file=constants.CONFIG_DEFAULT_FILE):
        """Saves the current configuration data to the specified file."""
        config_path = os.path.join(self.directory, config_file)

        with open(config_path, 'w') as file:
            json.dump(self.config_data, file, indent=4)
                
class Section():
    def __init__(self, section_name, configuration):
        self.configurations = configuration
        self.name = section_name

    def __getitem__(self, configuration_name):
        return self.configurations[configuration_name]
    
    def __setitem__(self, configuration_name, configuration):
        self.configurations[configuration_name] = configuration

    def __contains__(self, item):
        return item in self.configurations

    def items(self):
        return self.configurations.items()
    


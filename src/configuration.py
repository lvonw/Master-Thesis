import constants
import json
import os
import yaml

class Configuration():
    def __init__(self):
        self.name = "Default"
        self.sections = {}
        self.directory = constants.CONFIG_PATH

    def __getitem__(self, section_name):
        return self.sections[section_name]
    
    def add_section(self, section_name, section):
        self.sections[section_name] = Section(section_name, section)

    def get_configuration(self, section_name, configuration_name):
        return self.sections[section_name][configuration_name]
    
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
                # loaded_data = json.load(file)
                loaded_data = yaml.safe_load(file)

            for section_name, section in loaded_data.items():  
                self.add_section(section_name, section)

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
            # json.dump(self.config_data, file, indent=4)
            yaml.dump(self.config_data, file, indent=4)   

class Section():
    def __init__(self, section_name, section):
        self.name           = section_name
        self.description    = section["description"]
        self.configurations = self.__prepare_items(section["Configurations"])

    def __getitem__(self, configuration_name):
        if configuration_name in self.configurations:
            return self.configurations[configuration_name].value
        return None
    
    def __setitem__(self, configuration_name, configuration):
        self.configurations[configuration_name].value = configuration

    def __contains__(self, configuration_name):
        return configuration_name in self.configurations
    
    def __prepare_items(self, configuration_items):
        configurations = {}
        for name, item in configuration_items.items():
            configurations[name] = Item(item["value"], item["usage"])
        return configurations

    def get_usage(self, configuration_name):
        if configuration_name in self.configurations:
            return self.configurations[configuration_name].usage
        return ""
    
    def get_description(self):
        return self.description
        
    def items(self):
        return self.configurations.items()
    
    def values(self):
        values = []
        for name, config in self.configurations.items():
            values.append((name, config.value))
        return values
    
class Item():
    def __init__(self, value, usage):
        self.value  = value
        self.usage  = usage   
        self.type   = type(value) 
    


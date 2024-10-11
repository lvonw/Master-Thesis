import constants
import json
import os
import yaml
import util

from debug  import Printer

class Configuration():
    def __init__(self):
        self.name = "Default"
        self.directory = constants.CONFIG_PATH
        
        self.sections       = {}
        self.section_paths  = {}

    def __getitem__(self, section_name):
        if section_name in self.sections:
            return self.sections[section_name]
        else:
            Printer().print_log(f"{section_name} is not a valid section",
                                log_level=constants.LogLevel.ERROR) 
            return _InvalidSection()
        
    def __str__(self):
        string = "\n"

        for key, value in self.sections.items():
            string += key + "\n"
            string += str(value)

        return string
    
    def __prepare_sections(self, sections, config_path):
        for section_name, section in sections.items(): 
            if "external" in section:
                self.__add_external_section(section_name, section)
            else:
                self.__add_section(section_name, config_path, section)

    def get_external_section(section):
        external_path = util.make_path(section["external"])
        external_section = Configuration.__load_config_file(external_path)
        if not external_section:
            return None
        return external_section, external_path

    def __add_external_section(self, section_name, section):
            external_section, external_path = (
                Configuration.get_external_section(section))
            
            if (external_section 
                and "only_load_selection" in section 
                and "selection" in section
                and section["selection"] in external_section):

                external_section = external_section[section["selection"]]

                self.__add_section(section_name, 
                                   external_path, 
                                   external_section,
                                   read_only=True)
                return
            
            #self.__add_section(section_name, external_path, external_sections)
            
    def __add_section(self, 
                      section_name, 
                      section_path, 
                      section, 
                      read_only = False):

        self.sections[section_name] = Section(section_name, 
                                              section, 
                                              read_only=read_only)
        
        self.add_section_path(section_path, section_name)

    def __load_config_file(config_path):
        loaded_data = {}
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                loaded_data = yaml.safe_load(file)
        
        return loaded_data
            
    def add_section_path(self, section_path, section_name):
        self.section_paths.setdefault(section_path, []).append(section_name) 

    def print_load_options(self):
        files = []

        for file in os.listdir(self.directory):
            if (os.path.isfile(os.path.join(self.directory, file)) 
                and file.endswith(".json")):
                
                files.append(file)
        
        Printer().print(files)

    def load_defaults(self):
        self.load_configuration(constants.CONFIG_DEFAULT_FILE)

    def load_configuration(self, config_file):
        """Loads configuration data from the specified file."""
        config_path = os.path.join(self.directory, config_file)
        loaded_data = Configuration.__load_config_file(config_path)

        if loaded_data:
            self.__prepare_sections(loaded_data, config_path)
        elif config_file is not constants.CONFIG_DEFAULT_FILE:
            Printer().print(
                f"No config file found at {config_path}, loading defaults.")
            self.load_defaults()
        else:
            Printer().print("Default file could not be loaded.")
            self.print_load_options()
     
    def load_usages(self):
        usages = self.__load_config_file(constants.USAGES_FILE)

        if not usages:
            return 

        # distribute usage to items
        
    def save(self, config_file=constants.CONFIG_DEFAULT_FILE):
        """Saves the current configuration data to the specified file."""
        config_path = os.path.join(self.directory, config_file)

        with open(config_path, 'w') as file:
            yaml.dump(self.config_data, file, indent=4)   

class Section():
    def __init__(self, 
                 section_name, 
                 section,
                 read_only=False):
        
        self.name           = section_name
        self.description    = "Description not set yet"
        self.configurations = self.__prepare_items(section)
        self.read_only      = read_only

    def __getitem__(self, configuration_name):
        if configuration_name in self.configurations:
            if isinstance(self.configurations[configuration_name], Section):
                return self.configurations[configuration_name]
            return self.configurations[configuration_name].value
        else:
            Printer().print_log(f"{configuration_name} is not a valid item",
                                log_level=constants.LogLevel.ERROR) 
        return None
    
    def __setitem__(self, configuration_name, configuration):
        if self.read_only:
            Printer().print_log(f"{self.name} is read-only",
                                log_level=constants.LogLevel.WARNING) 
            return

        if configuration_name in self.configurations:
            self.configurations[configuration_name].value = configuration
        else:
            Printer().print_log(f"{configuration_name} is not a valid item",
                                log_level=constants.LogLevel.ERROR) 

    def __contains__(self, configuration_name):
        return configuration_name in self.configurations
    
    def __str__(self, level=1):
        string = ""

        for key, value in self.configurations.items():
            string += "\t" * level
            string += key + ":\n"
            string += value.__str__(level + 1)

        return string
    
    def __prepare_items(self, section):
        configurations = {}
        for item_name, item in section.items():
            if not isinstance(item, dict):
                configurations[item_name] = Item(item)
            elif "value" in item:
                configurations[item_name] = Item(item["value"])
            elif "external" in item:
                external_section, _ = (
                    Configuration.get_external_section(item))
                if not external_section:
                    continue
                configurations[item_name] = Section(item_name, 
                                                    external_section[
                                                        item["selection"]])
            else:
                configurations[item_name] = Section(item_name, item)

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
    def __init__(self, value):
        self.value  = value
        self.usage  = "No usage set yet"
        self.type   = type(value) 

    def __str__(self, level=2):
        return "\t" * level + str(self.value) + "\n"

    def set_usage(self, usage):
        self.usage  = usage   



class _InvalidSection(Section):
    _singleton = None
    
    def __new__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton
    
    def __init__(self):
        self.name           = "Invalid Section"
        self.description    = "Invalid Section, add this section to the config"
        self.configurations = {}


import json
import os
import parsing
import constants

from configuration import Section

class _MenuAction(parsing.Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "current_menu" not in kwargs:
            self.current_menu = None
        else:
            self.current_menu = kwargs["current_menu"]

class _MenuValueAction(_MenuAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, argument, values):
        key = argument.name
        
        if not (key in self.current_menu.config_data):
            return

        if argument.argument_type is bool:
            self.current_menu.config_data[key] = not self.current_menu.config_data[key]
        else:
            self.current_menu.config_data[key] = values[0]

class _SubMenuAction(_MenuAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "sub_menu" not in kwargs:
            self.sub_menu = None
        else:
            self.sub_menu = kwargs["sub_menu"]

    def __call__(self, argument, _):
        self.current_menu.chosen_sub_menu = self.sub_menu

class _MasterMenu():
    def __init__(self, title, configuration, sub_menues=[]):
        self.title = title
        self.chosen_sub_menu = None
        self.sub_menues = sub_menues
        
        self.config_data = configuration[self.title]
        self.parser = self.prepare_parser()

    def print(self):
        line_counter = 0
        for key, value in self.config_data.items():
            print(f"{constants.COLOUR_GRAY if line_counter % 2 == 1 else constants.STYLE_RESET}{key}: {value}{constants.STYLE_RESET}")
            line_counter += 1
        
        print("---")

        for sub_menu in self.sub_menues: 
            print (sub_menu.title)

        print("---")
        
        return line_counter
    
    def get_config_entry(self, key):
        return self.config_data[key]

    def get_and_reset_sub_menu(self):
        chosen_menu = self.chosen_sub_menu
        self.chosen_sub_menu = None
        return chosen_menu
    
    def load_defaults(self):
        return self.load_config(constants.CONFIG_DEFAULT_FILE)

    def prepare_parser(self):
        parser = parsing.Parser()

        for argument, value in self.config_data.items():
            arg_type = type(value)

            parser.add_argument(
                name    = argument,
                action  = _MenuValueAction,
                argument_type = arg_type,
                nargs   = 0 if arg_type is bool else 1,
                current_menu = self)
            
        for sub_menu in self.sub_menues:
            parser.add_argument(
                name    = sub_menu.title,
                action  = _SubMenuAction,
                argument_type = type(sub_menu),
                nargs   = 0,
                sub_menu = sub_menu,
                current_menu = self)

        return parser
        
class MainMenu(_MasterMenu):
    def __init__(self, configuration):
        super().__init__(constants.CONFIGURATION_MAIN, 
                         configuration,
                         sub_menues = [
                            DebugMenu(configuration),
                            PTGModelMenu(configuration),
                         ])

class DebugMenu(_MasterMenu):
    def __init__(self, configuration):
        super().__init__(constants.CONFIGURATION_DEBUG, 
                         configuration,
                         sub_menues = [
                             
                         ])
        
class PTGModelMenu(_MasterMenu):
    def __init__(self, configuration):
        super().__init__(constants.CONFIGURATION_PTG_MODEL, 
                         configuration,
                         sub_menues=[
                            HyperParameterMenu(configuration) 
                         ])

class HyperParameterMenu(_MasterMenu):
    def __init__(self, configuration):
        super().__init__(constants.CONFIGURATION_HYPER_PARAMETERS,
                         configuration,
                         sub_menues = [
                             
                         ])
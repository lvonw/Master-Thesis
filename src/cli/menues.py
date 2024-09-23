import json
import os
import constants

from cli            import parsing
from configuration  import Configuration, Section
from debug          import Printer

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
    def __init__(self, title, configuration: Configuration, sub_menues=[]):
        self.title = title
        self.chosen_sub_menu = None
        self.sub_menues = sub_menues
        self.config_data: Section = configuration[self.title]
        self.description = self.config_data.get_description()
        
        self.parser = self.prepare_parser()

    def print(self):
        printer = Printer()

        printer.print("===================")
        printer.print(self.description)
        printer.print("===================")
        self.__print_values(printer)
        printer.print("===================")


        printer.print_headline("Sub Menues")
        for sub_menu in self.sub_menues: 
            printer.print (f"{constants.FIXED_INDENT}{sub_menu.title}")
        printer.print("===================")
    
    def __print_values(self, printer):
        printer.print_headline("Values")

        value_line      = []
        line_counter    = 0

        for key, item in self.config_data.items():
            value_line.append(constants.FIXED_INDENT)

            if line_counter % 2 == 1:
                value_line.append(constants.COLOUR_GRAY)
            else:
                value_line.append(constants.STYLE_RESET)

            value_line.append(f"{key}: ")

            if item.type is bool:
                if item.value:
                    # value_line.append(constants.COLOUR_GREEN)
                    value_line.append("✅")
                else:
                    #value_line.append(constants.COLOUR_RED)
                    value_line.append("❌")
            else:
                value_line.append(str(item.value))
            
            value_line.append(constants.STYLE_RESET)
            printer.print("".join(value_line))

            line_counter += 1
            value_line.clear()

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

        for name, item in self.config_data.items():

            parser.add_argument(
                name            = name,
                action          = _MenuValueAction,
                argument_type   = item.type,
                nargs           = 0 if item.type is bool else 1,
                usage           = item.usage,
                current_menu    = self)
            
        for sub_menu in self.sub_menues:
            parser.add_argument(
                name            = sub_menu.title,
                action          = _SubMenuAction,
                argument_type   = type(sub_menu),
                nargs           = 0,
                sub_menu        = sub_menu,
                current_menu    = self)

        return parser
        
class MainMenu(_MasterMenu):
    def __init__(self, configuration):
        super().__init__(constants.CONFIGURATION_MAIN, 
                         configuration,
                         sub_menues = [
                            DebugMenu(configuration),
                            PTGModelMenu(configuration),
                            DataMenu(configuration),
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
        
class DataMenu(_MasterMenu):
    def __init__(self, configuration):
        super().__init__(constants.CONFIGURATION_DATA,
                         configuration,
                         sub_menues = [
                             
                         ])
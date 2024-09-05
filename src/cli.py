import menues
import parsing
import configuration

from debug import Printer

class CLI():
    def __init__(self, 
                 configuration: configuration.Configuration):
        
        self.configuration  = configuration

        self.nav_root   = menues.MainMenu(configuration)
        self.nav_stack  = [self.nav_root]

        self.cli_parser = parsing.Parser()
        self.cli_parser.add_argument(
                name    = "exit")
        self.cli_parser.add_argument(
                name    = "start")
        self.cli_parser.add_argument(
                name    = "back")
        self.cli_parser.add_argument(
                name    = "save",
                nargs   = 1)
        self.cli_parser.add_argument(
                name    = "load",
                nargs   = 1)
        self.cli_parser.add_argument(
                name    = "options")
        
        self.__add_cli_controls(self.nav_root.parser)

    def __add_cli_controls(self, parser):
        parser.decorate(self.cli_parser)

    def __print_nav_stack(self, printer):
        nav_stack_string = ""
        for i in range(len(self.nav_stack)-1):
            nav_stack_string += self.nav_stack[i].title
            nav_stack_string += ">"
        nav_stack_string += self.nav_stack[-1].title

        printer.print(nav_stack_string)
        printer.print("---")

    def print(self, printer):
        self.__print_nav_stack(printer)

    def cli_loop(self):
        printer = Printer()

        should_quit = False
        quit_loop = False
        
        while not quit_loop:
            
            self.nav_stack[-1].print()
            self.print(printer)
            user_input = printer.input(">>> ")
            printer.clear_all()

            parser = self.nav_stack[-1].parser
            parser.parse(user_input)

            chosen_sub_menu = self.nav_stack[-1].get_and_reset_sub_menu()
            
            if chosen_sub_menu is not None:
                self.__add_cli_controls(chosen_sub_menu.parser)
                self.nav_stack.append(chosen_sub_menu)
            elif parser.get_and_set_false("back") and len(self.nav_stack) > 1 :
                self.nav_stack.pop()
            elif parser["load"]:
                self.configuration.load(parser.get_and_set_false("load"))
            elif parser["save"]:
                self.configuration.save(parser.get_and_set_false("save"))
            elif parser.get_and_set_false("options"):
                self.configuration.print_load_options()
            elif parser.get_and_set_false("exit"):
                quit_loop = True
                should_quit = True
            elif parser.get_and_set_false("start"):
                quit_loop = True
        
        return self.configuration, should_quit


            
            





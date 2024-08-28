import menues
import parsing


class _CLIAction(parsing.Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "cli" not in kwargs:
            self.cli = None
        else:
            self.cli = kwargs["cli"]

    def __call__(self, argument, values):
        key = argument.name
        
        if key == "exit":
            self.cli.quit_loop = True

class _FlowControlAction(_CLIAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, argument, values):
        key = argument.name
        
        if key == "exit":
            self.cli.quit_loop = True
            
class CLI():
    def __init__(self, configuration, flow_control={}):
        self.configuration = configuration

        self.nav_root = menues.MainMenu(configuration)
        self.nav_stack = [self.nav_root]

        self.quit_loop = False

        self.cli_parser = parsing.Parser()
        self.cli_parser.add_argument(
                name    = "exit")
        self.cli_parser.add_argument(
                name    = "back")
        self.cli_parser.add_argument(
                name    = "save",
                nargs   = 1)
        self.cli_parser.add_argument(
                name    = "load",
                nargs   = 1)
        
        self.__add_cli_controls(self.nav_root.parser)

    def __add_cli_controls(self, parser):
        parser.decorate(self.cli_parser)

    def __print_nav_stack(self):
        nav_stack_string = ""
        for i in range(len(self.nav_stack)-1):
            nav_stack_string += self.nav_stack[i].title
            nav_stack_string += ">"
        nav_stack_string += self.nav_stack[-1].title

        print (nav_stack_string)
        print ("---")

    def cli_loop(self):
        while not self.quit_loop:
            self.nav_stack[-1].print()
            # print cli things
            self.__print_nav_stack()
            user_input = input(">>> ")

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
            elif parser.get_and_set_false("exit"):
                self.self.quit_loop = True


            
            





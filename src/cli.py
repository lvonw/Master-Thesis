import argparse
import enum
import configuration
import parsing

class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum._value2member_map_[values[0]]
        setattr(namespace, self.dest, value)

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
            

class CLI():
    def __init__(self):
        self.nav_root = configuration.MainMenu()
        
        self.nav_stack = [self.nav_root]
        self.quit_loop = False

        self.cli_parser = parsing.Parser()
        self.cli_parser.add_argument(
                name    = "exit",
                action  = _CLIAction,
                nargs   = 0,
                cli = self)
        

        self.__add_cli_controls(self.nav_root.parser)

    def __add_cli_controls(self, parser):
        parser.decorate(self.cli_parser)

    def cli_loop(self):
        while not self.quit_loop:
            self.nav_stack[-1].print()
            # print cli things

            user_input = input(">>> ")

            self.nav_stack[-1].parser.parse(user_input)

            sub_menu = self.nav_stack[-1].get_sub_menu()
            if sub_menu is not None:
                self.__add_cli_controls(sub_menu)
                self.nav_stack.append(sub_menu)





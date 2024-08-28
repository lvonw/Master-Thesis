import enum
import debug

class Action():
    def __init__(self, **kwargs):
        pass

    def __call__(self, argument, values):
        pass

class StoreTrueAction(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, argument, values):
        argument.parser[argument.name] = True

class StoreValueAction(Action):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, argument, values):
        argument.parser[argument.name] = values[0]

class EnumAction(Action):
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

class Argument():
    def __init__(self, name, action, parser, argument_type=str, nargs=0, **kwargs):
        self.name = name
        self.argument_type = argument_type
        self.action = action(**kwargs)
        self.nargs = nargs
        self.parser = parser

class Parser():
    def __init__(self):
        self.argument_map = {}
        self.value_map = {}
        self.super_parser = None

    def __getitem__(self, key):
        if key in self.value_map:
            return self.value_map[key]
        elif self.is_decorated() and key in self.super_parser.value_map:
            return self.super_parser.value_map[key]

        return None
    
    def __setitem__(self, key, value):
        self.value_map[key] = value

    def get_and_set_false(self, key):
        value = False

        if key in self.value_map:
            value = self.value_map[key]
            self.value_map[key] = False
        elif self.is_decorated() and key in self.super_parser.value_map:
            value = self.super_parser.value_map[key]
            self.super_parser.value_map[key] = False

        return value
        
    def add_argument(self, 
                     name, 
                     action=None, 
                     argument_type=str, 
                     nargs=0, 
                     **kwargs):

        if action is None:
            if nargs == 0:
                action = StoreTrueAction
            elif nargs == 1:
                action = StoreValueAction

        self.argument_map[name.lower()] = Argument(name, 
                                           action, 
                                           self, 
                                           argument_type=argument_type, 
                                           nargs=nargs, 
                                           **kwargs)

    def contains_argument(self, argument):
        return argument in self.argument_map
    
    def decorate(self, parser):
        self.super_parser = parser

    def is_decorated(self):
        return not (self.super_parser is None)

    def parse(self, user_input):
        user_input = user_input.lower()
        split_input = user_input.split(" ")

        if len(split_input) == 0 or user_input == "":
            print("no argument provided")
            return
        
        argument = None
        if self.contains_argument(split_input[0]):
            argument = self.argument_map[split_input[0]]
        elif self.is_decorated() and self.super_parser.contains_argument(split_input[0]):
            argument = self.super_parser.argument_map[split_input[0]]
        else: 
            print("argument could not be found")
            return

        values = split_input[1:]

        if len(values) is not argument.nargs:
            print(f"Expected {argument.nargs} arguments, got {len(values)}")
            return

        casted_values = []
        for value in values:
            try:
                casted_values.append(argument.argument_type(value))
            except (ValueError, TypeError) as e:
                print(f"Conversion to {argument.argument_type.__name__} failed")
                return

        argument.action.__call__(argument, casted_values)
        



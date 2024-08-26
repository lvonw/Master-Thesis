class Action():
    def __init__(self, **kwargs):
        pass

    def __call__(self, argument, values):
        pass

class StoreTrueAction(Action):
    def __init__(self, **kwargs):
        super.__init__(kwargs)

    def __call__(self, argument, values):
        argument.parser[argument] = True


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
        return self.value_map[key] if key in self.value_map else False
    
    def __setitem__(self, key, value):
        self.value_map[key] = value

    def add_argument(self, name, action=StoreTrueAction, argument_type=str, nargs=0, **kwargs):
        self.argument_map[name] = Argument(name, action, self, argument_type=argument_type, nargs=nargs, **kwargs)

    def contains_argument(self, argument):
        return argument in self.argument_map
    
    def decorate(self, parser):
        self.super_parser = parser

    def parse(self, user_input):
        user_input = user_input.lower()
        split_input = user_input.split(" ")

        if len(split_input) is 0 or user_input is "":
            print("no argument provided")
            return
        
        argument = None
        if self.contains_argument(split_input[0]):
            argument = self.argument_map[split_input[0]]
        elif self.super_parser is not None and self.super_parser.contains_argument(split_input[0]):
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
        



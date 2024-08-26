import argparse
import enum

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

def prepare_arg_parser():
    pass
    # parser = argparse.ArgumentParser(prog=constants.USAGE_PROGRAM_NAME,
    #                                  description=constants.USAGE_PROGRAM_DESC)
    # parser.add_argument("-a",
    #                     "--ask",
    #                     dest="question", 
    #                     nargs=1,
    #                     help=constants.USAGE_ASK,
    #                     type=str)
    # parser.add_argument("-db",
    #                     "--database",
    #                     dest="database",
    #                     default=constants.DEFAULT_DATABASE,
    #                     nargs=1,
    #                     help=constants.USAGE_DATABASE,
    #                     type=constants.LoaderMethod,
    #                     action=EnumAction) 
    # parser.add_argument("-m",
    #                     "--model",
    #                     dest="model",
    #                     default=constants.DEFAULT_MODEL,
    #                     nargs=1,
    #                     help=constants.USAGE_MODEL,
    #                     type=constants.ModelMethod,
    #                     action=EnumAction) 
    # parser.add_argument("-g",
    #                     "--gpt",
    #                     dest="gpt", 
    #                     nargs=1,
    #                     default=constants.DEFAULT_GPT,
    #                     choices=['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo', 'gpt-4-32k-0613', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4', 'gpt-4-1106-preview'],
    #                     help=constants.USAGE_GPT) 
    # parser.add_argument("-i", 
    #                     "--init", 
    #                     dest="init",
    #                     action="store_true",
    #                     help=constants.USAGE_INIT)
    # parser.add_argument("-v", 
    #                     "--validate", 
    #                     dest="validate",
    #                     action="store_true",
    #                     help=constants.USAGE_INIT)
    # parser.add_argument("-cv",
    #                     "--closest-vectors",
    #                     dest="cv",
    #                     action="store_true",
    #                     help=constants.USAGE_CLOSEST_V)
    # parser.add_argument("-c",
    #                     "--cli",
    #                     dest="cli",
    #                     action="store_true",
    #                     help=constants.USAGE_CLI)
    # return parser

import constants
import sys
import os

from datetime   import datetime
from enum       import Enum

class LogLevel(Enum):
    TEMP    = ("TEMP",      constants.COLOUR_GRAY)
    INFO    = ("INFO",      constants.COLOUR_BLUE)
    WARNING = ("WARNING",   constants.COLOUR_YELLOW)
    ERROR   = ("ERROR",     constants.COLOUR_RED) 

def print_to_log_file(data, filename):
    file_path = os.path.join(constants.LOG_PATH, filename)
    with open(file_path, 'a') as file:
        file.write(f"{data}\n")
    
class Printer():
    _singleton = None

    def __new__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(self):
        self.line_count = 0
    
    def __count_lines(self, message):
        return len(message.splitlines())
    
    def __increase_counter(self, message):
        self.line_count += self.__count_lines(message)
        
    def print(self, message):
        print(message)
        self.__increase_counter(message)

    def input(self, prompt):
        value = input(prompt)
        self.__increase_counter(prompt)
        return value
    
    def clear_lines(self, amount):
        for _ in range(amount):
            sys.stdout.write("\x1b[1A")
            sys.stdout.write("\x1b[2K")
            sys.stdout.flush()

        self.line_count -= amount

    def clear_all(self):
        self.clear_lines(self.line_count)

    def print_log(self, 
                  message, 
                  log_level=LogLevel.INFO):
        
        log_line = []
        log_line.append(str(log_level.value[1]))
        log_line.append(str(log_level.value[0]))
        log_line.append(" ")
        log_line.append(str(datetime.now()))
        log_line.append(" | ")
        log_line.append(constants.STYLE_RESET)
        log_line.append(message)

        self.print("".join(log_line))


    def print_headline(self, message):
        self.print(f"{constants.STYLE_BOLD}{message}{constants.STYLE_RESET}")





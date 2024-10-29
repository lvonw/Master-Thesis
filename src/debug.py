import constants
import sys
import os

import numpy    as np

from datetime   import datetime
from enum       import Enum

class LogLevel(Enum):
    TEMP    = ("TEMP",      constants.COLOUR_GRAY)
    INFO    = ("INFO",      constants.COLOUR_BLUE)
    WARNING = ("WARNING",   constants.COLOUR_YELLOW)
    ERROR   = ("ERROR",     constants.COLOUR_RED) 

def print_to_log_file(data, filename):
    os.makedirs(constants.LOG_PATH, exist_ok=True)
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
        self.ping_counter = 0

        self.rank = 0
    
    def __count_lines(self, message):
        return len(message.splitlines())
    
    def __increase_counter(self, message):
        self.line_count += self.__count_lines(message)
        
    def print(self, message):
        if self.rank == 0:
            print(message)
            self.__increase_counter(message)

    def input(self, prompt):
        if self.rank == 0:
            value = input(prompt)
            self.__increase_counter(prompt)
            return value
    
    def ping(self):
        self.print_log(self.ping_counter)
        self.ping_counter += 1

    
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
        log_line.append(str(datetime.
                            now()))
        log_line.append(" | ")
        log_line.append(constants.STYLE_RESET)
        log_line.append(str(message))

        self.print("".join(log_line))


    def print_headline(self, message):
        self.print(f"{constants.STYLE_BOLD}{message}{constants.STYLE_RESET}")

class LossLog():
    def __init__(self):
        self.loss_log = {}
        self.longest_category = 0

    def add_entry(self, category, value):
        if category not in self.loss_log:
            self.loss_log[category] = []
            if len(category) > self.longest_category:
                self.longest_category = len(category)

        self.loss_log[category].append(value.item())

    def __str__(self, index=-1, average_over=200):
        string = ""
        for category, values in self.loss_log.items():
            latest_losses = values[-average_over:]

            string += category + ":"
            string += " " * (1 + self.longest_category - len(category))
            string += str(np.mean(latest_losses)) 
            string += "\n"
        
        return string


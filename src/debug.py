import constants
import sys

from datetime import datetime

def print_log(message, log_level=constants.LogLevel.INFO):
    print(f"[LOG] {datetime.now()} | {log_level[1]}{log_level[0]}{constants.STYLE_RESET}: {message}")

class Printer():
    _singleton = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.line_count = 0
    
    def print(self, message):
        print(message)
        self.line_count += 1

    def clear_lines(self, amount):
        for _ in range(amount):
            sys.stdout.write("\x1b[1A")
            sys.stdout.write("\x1b[2K")
            sys.stdout.flush()

        self.line_count -= amount

    def clear_all(self):
        self.clear_lines(self.line_count)


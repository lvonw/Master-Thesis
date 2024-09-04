import constants
import sys

from datetime import datetime



class Printer():
    _singleton = None
    
    def __new__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

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

    def print_log(self, message, log_level=constants.LogLevel.INFO):
        self.print(f"[LOG] {datetime.now()} | {log_level[1]}{log_level[0]}{constants.STYLE_RESET}: {message}")

    def print_headline(self, message):
        self.print(f"{constants.STYLE_BOLD}{message}{constants.STYLE_RESET}")



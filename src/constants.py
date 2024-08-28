import os
import enum

# =============================================================================
# Paths
# =============================================================================
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config Paths
CONFIG_PATH = os.path.join(PROJECT_PATH, "cfg")
CONFIG_DEFAULT_FILE = "default.json"

# Data Paths
DATA_PATH_MASTER = os.path.join(PROJECT_PATH, "data")

# Model Paths
MODEL_PATH_MASTER = os.path.join(PROJECT_PATH, "models")


# =============================================================================
# Configs
# =============================================================================
CONFIGURATION_MAIN              = "Main"
CONFIGURATION_DEBUG             = "Debug"
CONFIGURATION_HYPER_PARAMETERS  = "Hyperparameters"
CONFIGURATION_PTG_MODEL         = "PTG Model"

# =============================================================================
# Terminal Formatting
# =============================================================================
STYLE_RESET         = "\033[0m"

STYLE_BOLD          = "\033[1m"
STYLE_UNDERLINE     = "\033[4m"

COLOUR_RED          = "\033[91m"
COLOUR_GREEN        = "\033[92m"
COLOUR_YELLOW       = "\033[93m"
COLOUR_BLUE         = "\033[94m"
COLOUR_CYAN         = "\033[95m"
COLOUR_PURPLE       = "\033[96m"
COLOUR_GRAY         = "\033[97m"
COLOUR_BLACK        = "\033[98m"

BACKGROUND_BLACK        = '\033[40m'
BACKGROUND_RED          = '\033[41m'
BACKGROUND_GREEN        = '\033[42m'
BACKGROUND_YELLOW       = '\033[43m'
BACKGROUND_BLUE         = '\033[44m'
BACKGROUND_MAGENTA      = '\033[45m'
BACKGROUND_CYAN         = '\033[46m'
BACKGROUND_LIGHT_GRAY   = '\033[47m'
BACKGROUND_DARK_GRAY    = '\033[100m'

# =============================================================================
# Enums
# =============================================================================
class LogLevel(enum.Enum):
    TEMP        = ("TEMP", COLOUR_GRAY)
    INFO        = ("INFO", COLOUR_BLUE)
    WARNING     = ("WARNING", COLOUR_YELLOW)
    ERROR       = ("ERROR", COLOUR_RED) 
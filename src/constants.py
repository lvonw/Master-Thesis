import os
import enum

# =============================================================================
# Paths
# =============================================================================
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Log Paths
LOG_PATH            = os.path.join(PROJECT_PATH, "log")
TRAINING_LOSS_LOG   = "training_loss.txt"
VALIDATION_LOSS_LOG = "validation_loss.txt"

# Config Paths
CONFIG_PATH         = os.path.join(PROJECT_PATH, "cfg")
CONFIG_FILE_FORMAT  = ".yaml"
CONFIG_DEFAULT_FILE = "default" + CONFIG_FILE_FORMAT
USAGES_FILE         = os.path.join(CONFIG_PATH, "usages" + CONFIG_FILE_FORMAT)

# Data Paths
DATA_PATH_MASTER    = os.path.join(PROJECT_PATH,        "data")
DATA_PATH_DEM       = os.path.join(DATA_PATH_MASTER,    "DEMs")
DATA_PATH_DEM_LIST  = os.path.join(DATA_PATH_DEM,       "SRTM_GL1_list.txt")
DATA_PATH_DEMS      = os.path.join(DATA_PATH_DEM,       "SRTM_GL1_srtm")
DATA_PATH_GLIM      = os.path.join(DATA_PATH_MASTER, 
                                   "GLiM", 
                                   "hartmann-moosdorf_2012", 
                                   "glim_wgs84_0point5deg.txt.asc")
DATA_PATH_CLIMATE   = os.path.join(DATA_PATH_MASTER, 
                                   "Climate", 
                                   "peel-et-al_2007", 
                                   "koeppen_wgs84_0point1deg.txt.asc")
DATA_PATH_DSMW      = os.path.join(DATA_PATH_MASTER, 
                                   "DSMW",
                                   "dsmw-fao",
                                   "dmsw_wgs84_0point08deg.txt.asc")
DATA_PATH_GTC       = os.path.join(DATA_PATH_MASTER, 
                                   "GTC",
                                   "Iwahashi_etal_2018",
                                   "GlobalTerrainClassification_Iwahashi_etal_2018.tif")

# Model Paths
MODEL_PATH_MASTER   = os.path.join(PROJECT_PATH, "models")
MODEL_PATH_TEST     = os.path.join(MODEL_PATH_MASTER, "test.pth")


# =============================================================================
# Configs
# =============================================================================
CONFIGURATION_MAIN              = "Main"
CONFIGURATION_DEBUG             = "Debug"
CONFIGURATION_HYPER_PARAMETERS  = "Hyperparameters"
CONFIGURATION_PTG_MODEL         = "PTG_Model"
CONFIGURATION_DATA              = "Data"

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

BACKGROUND_BLACK        = "\033[40m"
BACKGROUND_RED          = "\033[41m"
BACKGROUND_GREEN        = "\033[42m"
BACKGROUND_YELLOW       = "\033[43m"
BACKGROUND_BLUE         = "\033[44m"
BACKGROUND_MAGENTA      = "\033[45m"
BACKGROUND_CYAN         = "\033[46m"
BACKGROUND_LIGHT_GRAY   = "\033[47m"
BACKGROUND_DARK_GRAY    = "\033[100m"

FIXED_INDENT            = "   "

# =============================================================================
# Enums
# =============================================================================
class LogLevel(enum.Enum):
    TEMP    = ("TEMP",      COLOUR_GRAY)
    INFO    = ("INFO",      COLOUR_BLUE)
    WARNING = ("WARNING",   COLOUR_YELLOW)
    ERROR   = ("ERROR",     COLOUR_RED) 

class NoDataBehaviour(enum.Enum):
    GLOBAL_MINIMUM    = "Global_Minimum"
    LOCAL_MINIMUM     = "Local_Minimum"

# =============================================================================
# Seeds
# =============================================================================
DATALOADER_SEED = 42

# =============================================================================
# DATA
# =============================================================================
# Pre gathered these values as they are distributed across multiple DEMs
DEM_GLOBAL_MIN = -12269
DEM_GLOBAL_MAX =  22894
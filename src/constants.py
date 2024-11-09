import os
import enum

# =============================================================================
# Paths
# =============================================================================
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH_MASTER    = os.path.join(PROJECT_PATH, "data")

# Log Paths
LOG_PATH            = os.path.join(DATA_PATH_MASTER, "log")
TRAINING_LOSS_LOG   = "training_loss.txt"
VALIDATION_LOSS_LOG = "validation_loss.txt"
IMAGE_LOG           = os.path.join(LOG_PATH, "images")

LOG_ARCHITECTURE    = "architecture.txt"
LOG_CONFIGURATION   = "config.txt"
LOG_LOSS_FOLDER     = "losses"
LOG_IMAGES_FOLDER   = "images"

# Config Paths
CONFIG_PATH         = os.path.join(PROJECT_PATH, "cfg")
CONFIG_FILE_FORMAT  = ".yaml"
CONFIG_DEFAULT_FILE = "default" + CONFIG_FILE_FORMAT
USAGES_FILE         = os.path.join(CONFIG_PATH, "usages" + CONFIG_FILE_FORMAT)

# Data Paths
DATASET_PATH_MASTER = os.path.join(DATA_PATH_MASTER,    "datasets")
DATA_PATH_DEM       = os.path.join(DATASET_PATH_MASTER, "DEMs")
DATA_PATH_DEM_LIST  = os.path.join(DATA_PATH_DEM,       "SRTM_GL1_list.txt")
DATA_PATH_DEMS      = os.path.join(DATA_PATH_DEM,       "SRTM_GL1_srtm")
DATA_PATH_GLIM      = os.path.join(
    DATASET_PATH_MASTER, 
    "GLiM", 
    "hartmann-moosdorf_2012", 
    "glim_wgs84_0point5deg.txt.asc")
DATA_PATH_CLIMATE   = os.path.join(
    DATASET_PATH_MASTER, 
    "Climate", 
    "peel-et-al_2007", 
    "koeppen_wgs84_0point1deg.txt.asc")
DATA_PATH_DSMW      = os.path.join(
    DATASET_PATH_MASTER, 
    "DSMW",
    "dsmw-fao",
    "dmsw_wgs84_0point08deg.txt.asc")
DATA_PATH_GTC       = os.path.join(
    DATASET_PATH_MASTER, 
    "GTC",
    "Iwahashi_etal_2018",
    "3600x1800_GlobalTerrainClassification_Iwahashi_etal_2018.tif")

DEFAULT_DEM_LIST    = "SRTM_GL1_list.txt"

# Model Paths
MODEL_PATH_MASTER   = os.path.join(DATA_PATH_MASTER,
                                   "models")
MODEL_FILE_TYPE     = ".pth"
MODEL_PATH_TEST     = os.path.join(MODEL_PATH_MASTER, "test.pth")

# Resource Paths
RESOURCE_PATH_MASTER        = os.path.join(PROJECT_PATH, "resources")
RESOURCE_PATH_IMAGES        = os.path.join(RESOURCE_PATH_MASTER, "images")
RESOURCE_PATH_TEST_IMAGES   = os.path.join(RESOURCE_PATH_IMAGES, "test_images")

# =============================================================================
# Label Amounts
# =============================================================================
LABEL_AMOUNT_GTC = 16
LABEL_AMOUNT_CLIMATE = 33

NULL_LABEL = -1

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


# =============================================================================
# Seeds
# =============================================================================
DATALOADER_SEED = 42

# =============================================================================
# DATA
# =============================================================================
# Pre gathered these values as they are distributed across multiple DEMs
DEM_NODATA_VAL = -32768

DEM_GLOBAL_MIN = -1     # -1503 # -2    # 0     # -12269
DEM_GLOBAL_MAX = 8092   # 4993  # 3213  # 22894
import os

# =============================================================================
# Paths
# =============================================================================
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config Paths
CONFIG_PATH_MASTER = os.path.join(PROJECT_PATH, "cfg")
CONFIG_PATH_DEBUG = os.path.join(CONFIG_PATH_MASTER, "debug")
CONFIG_PATH_HYPER_PARAMETERS = os.path.join(CONFIG_PATH_MASTER, "hyper_parameters")
CONFIG_PATH_MAIN = os.path.join(CONFIG_PATH_MASTER, "main")
CONFIG_DEFAULT_FILE = "default.json"

# Data Paths
DATA_PATH_MASTER = os.path.join(PROJECT_PATH, "data")

# Model Paths
MODEL_PATH_MASTER = os.path.join(PROJECT_PATH, "models")

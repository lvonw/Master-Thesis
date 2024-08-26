import os


# =============================================================================
# Paths
# =============================================================================
PROJECT_PATH = os.path.join(__file__, "..")

# Config Paths
CONFIG_PATH_MASTER = os.path.join(PROJECT_PATH, "cfg")
CONFIG_PATH_DEBUG = os.path.join(CONFIG_PATH_MASTER, "debug")
CONFIG_PATH_HYPER_PARAMETERS = os.path.join(CONFIG_PATH_MASTER, "hyper_parameters")

# Data Paths
DATA_PATH_MASTER = os.path.join(PROJECT_PATH, "data")

# Model Paths
MODEL_PATH_MASTER = os.path.join(PROJECT_PATH, "models")

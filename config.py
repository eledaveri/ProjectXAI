# ============================================================================
# EXPERIMENTAL CONFIGURATION
# ============================================================================

# Data split parameters
TEST_SIZE = 0.3
N_SPLITS = 3
RANDOM_STATE = 42

# Dataset paths
DATASET_A_PATH = "data/dataset_A.csv"
DATASET_B_PATH = "data/dataset_A+B.csv"
DATASET_AB_PATH = "data/dataset_A+B.csv"

# Features to exclude in "no demographic" analysis
EXCLUDE_FEATURES = ["sex", "age"]

# Top-K features to use in feature-reduced models
TOP_K_FEATURES = 5

# Output directories
RESULTS_DIR = "results"
SHAP_PREFIX = "IS"
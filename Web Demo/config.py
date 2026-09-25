"""
Configuration file for Bank Marketing Demo App
Centralized settings and constants
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Models"
DATA_FILE = DATA_DIR / "processed_bank_data.csv"

# ============================================================================
# MODEL SETTINGS
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_SMOTE = True

# Logistic Regression Settings
LR_MAX_ITER = 1000
LR_C = 1.0
LR_SOLVER = "lbfgs"

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
SCALER_TYPE = "StandardScaler"
SMOTE_K_NEIGHBORS = 5

# ============================================================================
# UI SETTINGS
# ============================================================================
PAGE_TITLE = "Bank Marketing Prediction"
PAGE_ICON = "🏦"
LAYOUT = "wide"

# Colors
COLOR_PRIMARY = "#FF6B6B"
COLOR_SECONDARY = "#4ECDC4"
COLOR_SUCCESS = "#51CF66"
COLOR_ERROR = "#FF6B6B"
COLOR_WARNING = "#FFD93D"

# ============================================================================
# FEATURE SETTINGS
# ============================================================================
# Features to exclude from predictions
EXCLUDED_FEATURES = ["y"]

# Features for categorical encoding
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "day_of_week",
    "month",
    "poutcome",
]

# Numerical features
NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
PLOT_DPI = 100
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE_SMALL = (8, 5)
FIGURE_SIZE_MEDIUM = (12, 6)
FIGURE_SIZE_LARGE = (14, 8)

# ============================================================================
# METRICS SETTINGS
# ============================================================================
METRICS_TO_DISPLAY = ["roc_auc", "pr_auc", "f1", "balanced_accuracy"]

# ============================================================================
# CACHE SETTINGS
# ============================================================================
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour

# ============================================================================
# APP MESSAGES
# ============================================================================
MESSAGES = {
    "loading": "Loading data and training models...",
    "processing": "Processing your prediction...",
    "success": "✅ Prediction successful!",
    "error": "❌ An error occurred. Please try again.",
}

# ============================================================================
# FEATURE DESCRIPTIONS
# ============================================================================
FEATURE_DESCRIPTIONS = {
    "age": "Customer's age in years",
    "balance": "Annual balance in euros",
    "duration": "Last contact duration in seconds",
    "campaign": "Number of contacts during this campaign",
    "pdays": "Days since previous campaign contact (-1 if never contacted)",
    "previous": "Number of contacts before this campaign",
    "is_contacted_before": "Whether customer was contacted before",
}

# ============================================================================
# VALIDATION
# ============================================================================
def validate_data_file():
    """Check if the data file exists"""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
    return True

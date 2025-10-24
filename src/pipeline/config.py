"""
Configuration constants for the IDS pipeline
"""

import os

# ------------------------
# General settings
# ------------------------
SEED = 42
OUTPUT_DIR = "/kaggle/working/models_artifacts"

# ------------------------
# Data settings
# ------------------------
DATA_PATH = "/kaggle/working/All-Processed-Reduced-Cleaned.csv"
LABEL_COL = "Label"
SURROGATE_UNSEEN = ["Infiltration"]
SAMPLE_FRAC = 1.0  # For prototyping / downsampling

# ------------------------
# Autoencoder settings
# ------------------------
AE_LATENT_DIM = 64
AE_EPOCHS = 30
AE_BATCH_SIZE = 256

# ------------------------
# LightGBM settings
# ------------------------
LGBM_N_ESTIMATORS = 300
LGBM_NUM_LEAVES = 64

# ------------------------
# Fusion & threshold
# ------------------------
FUSION_WEIGHTS = (0.4, 0.4, 0.2)  # (ae, mahalanobis, softscore)
TARGET_TPR = 0.90

# ------------------------
# Torch device
# ------------------------
# DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

DEVICE = "cpu"

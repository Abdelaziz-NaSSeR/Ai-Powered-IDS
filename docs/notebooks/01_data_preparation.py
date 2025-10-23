# %% [markdown]
# 01 - Data Preparation
# Contains: imports, params, data loading, cleaning, label encoding, splits, scaling, save scaler
# %%
import os
import numpy as np
import pandas as pd
import joblib

# %% User Parameters
DATA_FILE = "All-Processed-Reduced-Cleaned.csv"
LABEL_COL = "Label"
SURROGATE_UNSEEN = ["Infiltration"]
RANDOM_STATE = 42
OUTPUT_DIR = "D:\AI_IDS_GRADUATION_PROJECT\docs\preprocessed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% Load dataset
df = pd.read_csv(DATA_FILE)
print("Initial shape:", df.shape)

# %% Preprocess
from src.data import preprocess_flow

df_preprocessed = preprocess_flow(df)
print("Preprocessing done. Shape:", df_preprocessed.shape)

# %% Split, encode, and scale using train_test_preparation
from scripts.training import train_test_preparation

data_splits = train_test_preparation(
    df=df_preprocessed,
    label_col=LABEL_COL,
    surrogate_unseen=SURROGATE_UNSEEN,
    random_state=RANDOM_STATE,
    scaler_path=os.path.join(OUTPUT_DIR, "scaler.joblib")
)

# %% Save preprocessed arrays and objects
np.save(os.path.join(OUTPUT_DIR, "X_train_s.npy"), data_splits["X_train_s"])
np.save(os.path.join(OUTPUT_DIR, "X_val_seen_s.npy"), data_splits["X_val_seen_s"])
np.save(os.path.join(OUTPUT_DIR, "X_test_seen_s.npy"), data_splits["X_test_seen_s"])
np.save(os.path.join(OUTPUT_DIR, "X_unseen_s.npy"), data_splits["X_unseen_s"])

np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), data_splits["y_train"])
np.save(os.path.join(OUTPUT_DIR, "y_val_seen.npy"), data_splits["y_val_seen"])
np.save(os.path.join(OUTPUT_DIR, "y_test_seen.npy"), data_splits["y_test_seen"])
np.save(os.path.join(OUTPUT_DIR, "y_unseen_str.npy"), data_splits["y_unseen_str"])

joblib.dump(data_splits["label_encoder"], os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
joblib.dump(data_splits["features"], os.path.join(OUTPUT_DIR, "features_list.joblib"))

print(f"Preprocessing complete. All objects saved in {OUTPUT_DIR}")
# %% [markdown]
# 02 - LightGBM Training
# Load preprocessed arrays from 01, train model on seen classes, evaluate, and save model
# %%
import os
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report

# %% Parameters
INPUT_DIR = "D:\AI_IDS_GRADUATION_PROJECT\docs\preprocessed_data"
MODEL_OUTPUT = "D:\AI_IDS_GRADUATION_PROJECT\docs\preprocessed_data\lgbm_seen.clf"

# %% Load preprocessed arrays and objects
X_train_s = np.load(os.path.join(INPUT_DIR, "X_train_s.npy"))
X_val_seen_s = np.load(os.path.join(INPUT_DIR, "X_val_seen_s.npy"))
X_test_seen_s = np.load(os.path.join(INPUT_DIR, "X_test_seen_s.npy"))
X_unseen_s = np.load(os.path.join(INPUT_DIR, "X_unseen_s.npy"))

y_train = np.load(os.path.join(INPUT_DIR, "y_train.npy"))
y_val_seen = np.load(os.path.join(INPUT_DIR, "y_val_seen.npy"))
y_test_seen = np.load(os.path.join(INPUT_DIR, "y_test_seen.npy"))
y_unseen_str = np.load(os.path.join(INPUT_DIR, "y_unseen_str.npy"))

le = joblib.load(os.path.join(INPUT_DIR, "label_encoder.joblib"))
features = joblib.load(os.path.join(INPUT_DIR, "features_list.joblib"))

# %% Train LightGBM on seen classes
clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=64, random_state=42, n_jobs=-1)
clf.fit(X_train_s, y_train)

# %% Evaluate on validation set
y_val_pred = clf.predict(X_val_seen_s)
print("Validation classification report (seen classes):")
print(classification_report(y_val_seen, y_val_pred, target_names=le.classes_))

# %% Save trained model
joblib.dump(clf, MODEL_OUTPUT)
print(f"Saved LightGBM model: {MODEL_OUTPUT}")

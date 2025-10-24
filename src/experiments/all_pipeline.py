#!/usr/bin/env python3
"""
ssl_ids_pipeline.py

Semi-Supervised Intrusion Detection System:
- LightGBM (multi-class)
- Autoencoder (PyTorch)
- Mahalanobis novelty scoring

Integrated with config.py defaults and new preprocessing.
"""

import os
import json
import random
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from numpy import linalg as la

# Project modules
from pipeline.config import (
    SEED, OUTPUT_DIR, DATA_PATH, LABEL_COL, SURROGATE_UNSEEN, SAMPLE_FRAC,
    AE_LATENT_DIM, AE_EPOCHS, AE_BATCH_SIZE, LGBM_N_ESTIMATORS, LGBM_NUM_LEAVES,
    FUSION_WEIGHTS, TARGET_TPR, DEVICE
)
from pipeline.data_utils import preprocess_flow, AE

# ------------------------
# Random seeds
# ------------------------
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------
# Data functions
# ------------------------
def load_dataset(path: str, sample_frac: float = 1.0) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    print(f"[DATA] Loading dataset from {path}")
    df = pd.read_csv(path)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=SEED).reset_index(drop=True)
    df = preprocess_flow(df)
    print(f"[DATA] Dataset shape after preprocessing: {df.shape}")
    return df

def select_numeric_features(df: pd.DataFrame, label_col: str) -> List[str]:
    feature_cols = [c for c in df.columns if c != label_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric features found in dataset")
    print(f"[DATA] Selected {len(numeric_cols)} numeric features")
    return numeric_cols

# ------------------------
# Splits
# ------------------------
def create_splits(df: pd.DataFrame, features: List[str], label_col: str,
                  surrogate_unseen: List[str], test_size=0.2, val_frac_of_train=0.2):
    df["y_str"] = df[label_col].astype(str)
    mask_unseen = df["y_str"].isin(surrogate_unseen)
    df["is_unseen_surrogate"] = mask_unseen
    df_seen = df[~mask_unseen].copy()
    df_unseen = df[mask_unseen].copy()
    print(f"[SPLIT] Seen rows: {len(df_seen)}, Surrogate-unseen rows: {len(df_unseen)}")

    le = LabelEncoder()
    df_seen["y_enc"] = le.fit_transform(df_seen["y_str"])

    X = df_seen[features].values
    y = df_seen["y_enc"].values

    X_trainval, X_test_seen, y_trainval, y_test_seen = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    X_train, X_val_seen, y_train, y_val_seen = train_test_split(
        X_trainval, y_trainval, test_size=val_frac_of_train, stratify=y_trainval, random_state=SEED
    )

    X_unseen = df_unseen[features].values

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val_seen": X_val_seen, "y_val_seen": y_val_seen,
        "X_test_seen": X_test_seen, "y_test_seen": y_test_seen,
        "X_unseen": X_unseen,
        "le": le
    }

# ------------------------
# Scaling
# ------------------------
def scale_features(X_train, X_val_seen, X_test_seen, X_unseen):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_seen_s = scaler.transform(X_val_seen)
    X_test_seen_s = scaler.transform(X_test_seen)
    X_unseen_s = scaler.transform(X_unseen) if X_unseen.shape[0] > 0 else np.zeros((0, X_train.shape[1]))
    return X_train_s, X_val_seen_s, X_test_seen_s, X_unseen_s, scaler

# ------------------------
# LightGBM
# ------------------------
def train_lightgbm(X_train_s, y_train, X_val_s, y_val):
    clf = lgb.LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        num_leaves=LGBM_NUM_LEAVES,
        random_state=SEED,
        n_jobs=-1
    )
    clf.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], early_stopping_rounds=50, verbose=False)
    print(f"[LGBM] Training done. Best iteration: {clf.best_iteration_}")
    return clf

# ------------------------
# Autoencoder
# ------------------------
def train_autoencoder(X_train_s, latent_dim=AE_LATENT_DIM, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, device=DEVICE):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_s.shape[1]
    ae = AE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_s, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ae.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = ae(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch+1) % 5 == 0 or epoch==0:
            print(f"[AE] Epoch {epoch+1}/{epochs} Loss={total_loss / len(loader.dataset):.6f}")
    return ae

# ------------------------
# AE reconstruction errors
# ------------------------
def get_recon_errors(ae: AE, X_np: np.ndarray, device: str = DEVICE) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ae.eval()
    if X_np.shape[0] == 0:
        return np.zeros(0)
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
        recon, _ = ae(X_t)
        recon_np = recon.cpu().numpy()
        errs = np.mean((recon_np - X_np) ** 2, axis=1)
    return errs

# ------------------------
# Mahalanobis distance scoring
# ------------------------
def compute_class_means_and_cov(X_train_s: np.ndarray, y_train: np.ndarray, reg: float = 1e-6):
    classes = np.unique(y_train)
    class_means = {c: X_train_s[y_train==c].mean(axis=0) for c in classes}
    cov = np.cov(X_train_s, rowvar=False) + reg * np.eye(X_train_s.shape[1])
    cov_inv = la.pinv(cov)
    return class_means, cov_inv

def mahalanobis_to_closest(X_np: np.ndarray, class_means: Dict[int, np.ndarray], cov_inv: np.ndarray) -> np.ndarray:
    if X_np.shape[0] == 0:
        return np.zeros(0)
    dists = []
    mus = list(class_means.values())
    for xi in X_np:
        ds = [float((xi - mu).dot(cov_inv).dot(xi - mu)) for mu in mus]
        dists.append(min(ds))
    return np.array(dists)

# ------------------------
# Fusion of AE + Mahalanobis + Softmax
# ------------------------
def minmax_norm(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def build_fused_score(ae_scores: np.ndarray, md_scores: np.ndarray, soft_scores: np.ndarray,
                      ae_pool: np.ndarray, md_pool: np.ndarray, soft_pool: np.ndarray,
                      weights=FUSION_WEIGHTS) -> np.ndarray:
    ae_norm = (ae_scores - ae_pool.min()) / (ae_pool.max() - ae_pool.min() + 1e-12)
    md_norm = (md_scores - md_pool.min()) / (md_pool.max() - md_pool.min() + 1e-12)
    soft_norm = (soft_scores - soft_pool.min()) / (soft_pool.max() - soft_pool.min() + 1e-12)
    w1, w2, w3 = weights
    fused = w1 * ae_norm + w2 * md_norm + w3 * soft_norm
    return fused

# ------------------------
# Threshold selection
# ------------------------
def choose_threshold_from_pool(fused_pool: np.ndarray, y_pool: np.ndarray, target_tpr: float = TARGET_TPR):
    fpr, tpr, thresholds = roc_curve(y_pool, fused_pool)
    roc_auc_val = auc(fpr, tpr)
    chosen = None
    for fval, tval, th in zip(fpr, tpr, thresholds):
        if tval >= target_tpr:
            chosen = (th, fval, tval)
            break
    if chosen is None:
        youdens = tpr - fpr
        idx = np.argmax(youdens)
        chosen = (thresholds[idx], fpr[idx], tpr[idx])
    thr, thr_fpr, thr_tpr = chosen
    return thr, thr_fpr, thr_tpr, (fpr, tpr, roc_auc_val)

# ------------------------
# Evaluation
# ------------------------
def evaluate_seen_classification(clf: lgb.LGBMClassifier, X_test: np.ndarray, y_test: np.ndarray, le: LabelEncoder):
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, target_names=le.classes_)
    cm = confusion_matrix(y_test, preds)
    print("[EVAL] Classification report (seen test):")
    print(report)
    print("[EVAL] Confusion matrix (rows true -> cols pred):")
    print(cm)
    return report, cm

# ------------------------
# Save artifacts
# ------------------------
def save_artifacts(output_dir: str, scaler: StandardScaler, clf: lgb.LGBMClassifier, ae: AE,
                   ae_state_path: str, cov_stats: dict, pool_stats: dict, thr: float):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(clf, os.path.join(output_dir, "lgbm_seen.clf"))
    torch.save(ae.state_dict(), os.path.join(output_dir, ae_state_path))
    with open(os.path.join(output_dir, "mahalanobis_stats.json"), "w") as f:
        json.dump(cov_stats, f)
    with open(os.path.join(output_dir, "pool_stats.json"), "w") as f:
        json.dump(pool_stats, f)
    with open(os.path.join(output_dir, "threshold.json"), "w") as f:
        json.dump({"fused_threshold": float(thr)}, f)
    print(f"[SAVE] Artifacts saved to: {output_dir}")

# ------------------------
# Inference (single sample)
# ------------------------
def infer_one(x_raw: np.ndarray, scaler: StandardScaler, clf: lgb.LGBMClassifier, ae: AE,
              class_means, cov_inv, pool_stats: dict, weights=FUSION_WEIGHTS, thr=0.5, device=DEVICE):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x_s = scaler.transform(x_raw.reshape(1, -1))
    probs = clf.predict_proba(x_s)[0]
    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    ae_err = get_recon_errors(ae, x_s, device=device)[0]
    md = mahalanobis_to_closest(x_s, class_means, cov_inv)[0]
    soft_score = 1.0 - pred_conf

    ae_pool = np.array(pool_stats["ae_pool"])
    md_pool = np.array(pool_stats["md_pool"])
    soft_pool = np.array(pool_stats["soft_pool"])

    fused = build_fused_score(np.array([ae_err]), np.array([md]), np.array([soft_score]),
                              ae_pool, md_pool, soft_pool, weights=weights)[0]
    is_unknown = fused >= thr
    return {
        "predicted_index": pred_idx,
        "pred_proba": pred_conf,
        "ae_err": float(ae_err),
        "mahalanobis": float(md),
        "fused_score": float(fused),
        "is_unknown": bool(is_unknown)
    }


# ------------------------
# Main
# ------------------------
def main(args):
    # Load and preprocess dataset
    df = load_dataset(DATA_PATH, sample_frac=SAMPLE_FRAC)
    features = select_numeric_features(df, LABEL_COL)

    # Create splits
    splits = create_splits(df, features, LABEL_COL, SURROGATE_UNSEEN)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val_seen, y_val_seen = splits["X_val_seen"], splits["y_val_seen"]
    X_test_seen, y_test_seen = splits["X_test_seen"], splits["y_test_seen"]
    X_unseen = splits["X_unseen"]
    le = splits["le"]

    # Scale
    X_train_s, X_val_seen_s, X_test_seen_s, X_unseen_s, scaler = scale_features(
        X_train, X_val_seen, X_test_seen, X_unseen
    )

    # Train models
    clf = train_lightgbm(X_train_s, y_train, X_val_seen_s, y_val_seen)
    ae = train_autoencoder(X_train_s)

    # Further steps: AE errors, Mahalanobis, fusion, threshold selection, evaluation, save artifacts
    # (reuse code from previous pipeline, now fully modular)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised IDS Pipeline")
    args = parser.parse_args()
    main(args)

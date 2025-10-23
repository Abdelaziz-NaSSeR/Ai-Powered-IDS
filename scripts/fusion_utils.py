# scripts/fusion_utils.py
import numpy as np
from sklearn.metrics import roc_curve, auc

def fuse_scores(ae_scores, mdist_scores, soft_scores, weights=(0.4,0.4,0.2)):
    a = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-12)
    b = (mdist_scores - mdist_scores.min()) / (mdist_scores.max() - mdist_scores.min() + 1e-12)
    c = (soft_scores - soft_scores.min()) / (soft_scores.max() - soft_scores.min() + 1e-12)
    fused = weights[0]*a + weights[1]*b + weights[2]*c
    return fused

def choose_threshold(y_true_binary, fused_scores, target_tpr=0.9):
    fpr, tpr, thresholds = roc_curve(y_true_binary, fused_scores)
    for f, t, th in zip(fpr, tpr, thresholds):
        if t >= target_tpr:
            return {"threshold": float(th), "fpr": float(f), "tpr": float(t), "auc": float(auc(fpr,tpr))}
    # fallback: Youden's J
    youdens = tpr - fpr
    idx = youdens.argmax()
    return {"threshold": float(thresholds[idx]), "fpr": float(fpr[idx]), "tpr": float(tpr[idx]), "auc": float(auc(fpr,tpr))}

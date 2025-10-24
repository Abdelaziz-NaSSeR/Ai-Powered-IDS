import numpy as np
from numpy import linalg as la
from sklearn.metrics import roc_curve, auc
from pipeline.config import FUSION_WEIGHTS, TARGET_TPR, DEVICE

# ------------------------
# AE errors
# ------------------------
def get_recon_errors(ae, X_np, device=DEVICE):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ae.eval()
    if X_np.shape[0] == 0:
        return np.zeros(0)
    import torch
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
        recon, _ = ae(X_t)
        recon_np = recon.cpu().numpy()
        errs = np.mean((recon_np - X_np) ** 2, axis=1)
    return errs

# ------------------------
# Mahalanobis
# ------------------------
def compute_class_means_and_cov(X_train, y_train, reg=1e-6):
    classes = np.unique(y_train)
    class_means = {c: X_train[y_train==c].mean(axis=0) for c in classes}
    cov = np.cov(X_train, rowvar=False) + reg * np.eye(X_train.shape[1])
    cov_inv = la.pinv(cov)
    return class_means, cov_inv

def mahalanobis_to_closest(X_np, class_means, cov_inv):
    if X_np.shape[0] == 0:
        return np.zeros(0)
    dists = []
    mus = list(class_means.values())
    for xi in X_np:
        ds = [float((xi - mu).dot(cov_inv).dot(xi - mu)) for mu in mus]
        dists.append(min(ds))
    return np.array(dists)

# ------------------------
# Fusion + normalization
# ------------------------
def minmax_norm(x):
    if len(x) == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def build_fused_score(ae_scores, md_scores, soft_scores, ae_pool, md_pool, soft_pool, weights=FUSION_WEIGHTS):
    ae_norm = (ae_scores - ae_pool.min()) / (ae_pool.max() - ae_pool.min() + 1e-12)
    md_norm = (md_scores - md_pool.min()) / (md_pool.max() - md_pool.min() + 1e-12)
    soft_norm = (soft_scores - soft_pool.min()) / (soft_pool.max() - soft_pool.min() + 1e-12)
    w1, w2, w3 = weights
    fused = w1*ae_norm + w2*md_norm + w3*soft_norm
    return fused

# ------------------------
# Threshold
# ------------------------
def choose_threshold_from_pool(fused_pool, y_pool, target_tpr=TARGET_TPR):
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

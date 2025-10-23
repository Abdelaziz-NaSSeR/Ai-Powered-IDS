# %% [markdown]
# 05 - Fusion & Threshold Tuning (split from original)
# Contains: build validation pool, normalize scores, fuse and pick threshold
# %%
import numpy as np
from sklearn.metrics import roc_curve, auc

val_unseen_sample_count = min(len(X_unseen_s), 5000)
if val_unseen_sample_count > 0:
    idxs = np.random.choice(len(X_unseen_s), val_unseen_sample_count, replace=False)
    X_val_unseen_s = X_unseen_s[idxs]
    errs_val_unseen = errs_unseen[idxs]
    mdist_val_unseen = mdist_unseen[idxs]
    maxp_val_unseen = maxp_unseen[idxs] if maxp_unseen.size else np.zeros(len(idxs))
else:
    X_val_unseen_s = np.zeros((0, X_val_seen_s.shape[1]))
    errs_val_unseen = np.zeros((0,))
    mdist_val_unseen = np.zeros((0,))
    maxp_val_unseen = np.zeros((0,))

ae_scores_pool = np.concatenate([errs_val_seen, errs_val_unseen])
mdist_pool = np.concatenate([mdist_val_seen, mdist_val_unseen])
soft_pool = np.concatenate([1.0 - maxp_val, 1.0 - maxp_val_unseen]) if maxp_val_unseen.size else (1.0 - maxp_val)

def minmax_norm(x):
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

ae_norm_pool = minmax_norm(ae_scores_pool)
mdist_norm_pool = minmax_norm(mdist_pool)
soft_norm_pool = minmax_norm(soft_pool)

w1, w2, w3 = 0.4, 0.4, 0.2
fused_pool = w1 * ae_norm_pool + w2 * mdist_norm_pool + w3 * soft_norm_pool

y_pool = np.concatenate([np.zeros(len(ae_scores_pool)-len(errs_val_unseen)), np.ones(len(errs_val_unseen))]) if len(errs_val_unseen)>0 else np.zeros(len(ae_scores_pool))

fpr, tpr, thresholds = roc_curve(y_pool, fused_pool)
roc_auc = auc(fpr, tpr)
print(f"Fusion ROC AUC on validation pool (surrogate unseen): {roc_auc:.4f}")

target_tpr = 0.90
chosen = None
for f_val, t_val, th_val in zip(fpr, tpr, thresholds):
    if t_val >= target_tpr:
        chosen = (th_val, f_val, t_val)
        break
if chosen is None:
    youdens = tpr - fpr
    idx = np.argmax(youdens)
    chosen = (thresholds[idx], fpr[idx], tpr[idx])
thr_fused, thr_fpr, thr_tpr = chosen
print(f"Chosen fused threshold: {thr_fused:.4f} -> FPR: {thr_fpr:.4f}, TPR: {thr_tpr:.4f}")

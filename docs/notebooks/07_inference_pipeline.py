# %% [markdown]
# 07 - Inference Pipeline (split from original)
# Contains: infer_one function and example usage
# %%
import numpy as np

def infer_one(x_raw):
    x_s = scaler.transform(x_raw.reshape(1,-1))
    probs = clf.predict_proba(x_s)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    maxp = probs[pred_idx]
    ae_err = get_recon_errors(ae, x_s.reshape(1,-1))[0]
    md = mahalanobis_to_closest(x_s.reshape(1,-1))[0]
    ae_norm = (ae_err - ae_scores_pool.min()) / (ae_scores_pool.max() - ae_scores_pool.min() + 1e-12)
    md_norm = (md - mdist_pool.min()) / (mdist_pool.max() - mdist_pool.min() + 1e-12)
    soft_norm = ((1.0 - maxp) - soft_pool.min()) / (soft_pool.max() - soft_pool.min() + 1e-12)
    fused = w1*ae_norm + w2*md_norm + w3*soft_norm
    is_unknown = fused >= thr_fused
    return {
        "pred_label": pred_label,
        "pred_proba": float(maxp),
        "ae_err": float(ae_err),
        "mahalanobis": float(md),
        "fused_score": float(fused),
        "is_unknown": bool(is_unknown)
    }

example = X_test_seen[0]
res = infer_one(example)
print("Example inference:", res)

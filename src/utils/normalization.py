# src/utils/normalization.py
import numpy as np

def minmax_norm(x):
    x = np.asarray(x)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

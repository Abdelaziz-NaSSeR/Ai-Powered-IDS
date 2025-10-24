import numpy as np
from pipeline.config import FUSION_WEIGHTS, DEVICE
from pipeline.scoring_utils import build_fused_score, get_recon_errors, mahalanobis_to_closest

def infer_one(x_raw, scaler, clf, ae, class_means, cov_inv, pool_stats, weights=FUSION_WEIGHTS, thr=0.5, device=DEVICE):
    """
    Infer a single sample with fused AE + Mahalanobis + soft classifier.
    Args:
        x_raw: raw feature vector (1D numpy array)
        scaler: fitted scaler to normalize features
        clf: trained LightGBM classifier
        ae: trained autoencoder
        class_means: class means from training (for Mahalanobis)
        cov_inv: inverse covariance matrix
        pool_stats: dict containing AE, Mahalanobis, soft score pools for normalization
        weights: fusion weights
        thr: threshold to decide anomaly
    Returns:
        fused_score, prediction (0=normal, 1=anomaly)
    """
    x_s = scaler.transform(x_raw.reshape(1, -1))

    # AE score
    ae_score = get_recon_errors(ae, x_s, device=device)

    # Mahalanobis score
    md_score = mahalanobis_to_closest(x_s, class_means, cov_inv)

    # Soft classifier score (probability of anomaly)
    soft_score = clf.predict_proba(x_s)[:, 1]

    # Fused score
    fused_score = build_fused_score(
        ae_score, md_score, soft_score,
        pool_stats["ae_pool"], pool_stats["md_pool"], pool_stats["soft_pool"],
        weights=weights
    )

    # Decision
    pred = int(fused_score >= thr)
    return fused_score[0], pred

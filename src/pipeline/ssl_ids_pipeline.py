import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.pipeline.config import SEED, LABEL_COL, OUTPUT_DIR, SAMPLE_FRAC, DEVICE
from src.pipeline.data_utils import load_csv, preprocess_flow
from src.pipeline.model_utils import train_autoencoder, train_lightgbm
from src.pipeline.scoring_utils import compute_class_means_and_cov, get_recon_errors, build_fused_score, choose_threshold_from_pool

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------
    # Load & preprocess
    # ------------------------
    df = load_csv(sample_frac=SAMPLE_FRAC)
    df = preprocess_flow(df)
    X = df.drop(columns=[LABEL_COL]).values
    y = df[LABEL_COL].apply(lambda x: 0 if x=="BENIGN" else 1).values

    # ------------------------
    # Train/Test split
    # ------------------------
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # ------------------------
    # Standardize
    # ------------------------
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    # ------------------------
    # Train AE
    # ------------------------
    ae = train_autoencoder(X_train_s)

    # ------------------------
    # Train LightGBM
    # ------------------------
    clf = train_lightgbm(X_train_s, y_train, X_val_s, y_val)

    # ------------------------
    # Compute Mahalanobis stats
    # ------------------------
    class_means, cov_inv = compute_class_means_and_cov(X_train_s, y_train)

    # ------------------------
    # Compute score pools
    # ------------------------
    ae_pool = get_recon_errors(ae, X_train_s, device=DEVICE)
    md_pool = np.array([min((xi - class_means[0]).dot(cov_inv).dot(xi - class_means[0]),
                            (xi - class_means[1]).dot(cov_inv).dot(xi - class_means[1]))
                        for xi in X_train_s])
    soft_pool = clf.predict_proba(X_train_s)[:, 1]

    pool_stats = {"ae_pool": ae_pool, "md_pool": md_pool, "soft_pool": soft_pool}

    # ------------------------
    # Threshold selection
    # ------------------------
    from pipeline.scoring_utils import choose_threshold_from_pool
    thr, fpr, tpr, roc_stats = choose_threshold_from_pool(
        build_fused_score(ae_pool, md_pool, soft_pool, ae_pool, md_pool, soft_pool),
        y_train
    )

    # ------------------------
    # Save artifacts
    # ------------------------
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, "ae.pkl"), "wb") as f:
        pickle.dump(ae.state_dict(), f)
    with open(os.path.join(OUTPUT_DIR, "clf.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(OUTPUT_DIR, "mah_stats.pkl"), "wb") as f:
        pickle.dump({"class_means": class_means, "cov_inv": cov_inv}, f)
    with open(os.path.join(OUTPUT_DIR, "pool_stats.pkl"), "wb") as f:
        pickle.dump(pool_stats, f)
    with open(os.path.join(OUTPUT_DIR, "threshold.pkl"), "wb") as f:
        pickle.dump(thr, f)

    print(f"[Pipeline] Finished training. Threshold={thr:.4f}")

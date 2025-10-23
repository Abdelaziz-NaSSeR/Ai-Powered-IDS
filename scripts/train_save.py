# scripts/train_save.py
import sys
import pandas as pd
from pathlib import Path
from src.models.lightgbm_wrapper import train_lgbm
from scripts.training import train_test_preparation
import joblib
import yaml

def main(csv_path, label_col, config_path="config/config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    models_dir = Path(cfg.get("paths",{}).get("models_dir","models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    out_scaler = models_dir / cfg["artifacts"]["scaler_file"]
    # load data
    df = pd.read_csv(csv_path)
    splits = train_test_preparation(df, label_col, surrogate_unseen=[])
    model = train_lgbm(splits["X_train_s"], splits["y_train"], params=cfg.get("training",{}).get("lgbm",{}))
    joblib.dump(model, models_dir / cfg["artifacts"]["model_file"])
    joblib.dump(splits["features"], models_dir / cfg["artifacts"]["features_file"])
    joblib.dump(splits["label_encoder"], models_dir / cfg["artifacts"]["label_encoder_file"])
    joblib.dump(splits["X_train_s"].shape, models_dir / "train_shape.joblib")
    print("Saved model and artifacts to", models_dir)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/train_save.py <csv_path> <label_col>")
    else:
        main(sys.argv[1], sys.argv[2])

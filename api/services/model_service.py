# api/services/model_service.py
import joblib
import yaml
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, config=None):
        app_root = Path(__file__).resolve().parents[2]
        if config is None:
            cfg_path = app_root / "config" / "config.yaml"
            try:
                config = yaml.safe_load(open(cfg_path, "r"))
            except Exception:
                config = {}

        models_dir = app_root / config.get("paths", {}).get("models_dir", "models")
        artifacts = config.get("artifacts", {})
        self.model_file = models_dir / artifacts.get("model_file", "trained_model.joblib")
        self.scaler_file = models_dir / artifacts.get("scaler_file", "scaler.joblib")
        self.features_file = models_dir / artifacts.get("features_file", "feature_columns.joblib")
        self.le_file = models_dir / artifacts.get("label_encoder_file", "label_encoder.joblib")
        self.ae_file = models_dir / artifacts.get("ae_file", "ae_model.pt")
        self.live_flow = Path(config.get("paths", {}).get("live_flow", app_root / "data" / "live" / "live_flow.csv"))

        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.label_encoder = None
        self.ae = None
        self.is_loaded = False

        self.load_model()

    def load_model(self):
        try:
            if self.model_file.exists():
                self.model = joblib.load(self.model_file)
            if self.scaler_file.exists():
                self.scaler = joblib.load(self.scaler_file)
            if self.features_file.exists():
                self.feature_columns = joblib.load(self.features_file)
            if self.le_file.exists():
                self.label_encoder = joblib.load(self.le_file)
            # AE is optional; loading is dependent on user choice
            # set is_loaded flag
            self.is_loaded = self.model is not None
        except Exception:
            logger.exception("Error loading artifacts")
            self.is_loaded = False

    def preprocess_live_data(self, df: pd.DataFrame):
        if self.feature_columns is None:
            raise RuntimeError("Feature columns not loaded")
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        for c in self.feature_columns:
            if c not in df.columns:
                df[c] = 0
        df = df[self.feature_columns]
        if self.scaler is not None:
            arr = self.scaler.transform(df)
            return pd.DataFrame(arr, columns=self.feature_columns)
        return df

    def predict_live_traffic(self):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        if not self.live_flow.exists():
            logger.warning("Live flow file not found")
            return []
        df = pd.read_csv(self.live_flow)
        if df.empty:
            return []
        processed = self.preprocess_live_data(df)
        preds = self.model.predict(processed)
        confidences = None
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(processed)
                confidences = probs.max(axis=1).tolist()
            except Exception:
                confidences = [None] * len(preds)
        else:
            confidences = [None] * len(preds)

        if self.label_encoder is not None:
            try:
                pred_labels = self.label_encoder.inverse_transform(preds)
            except Exception:
                pred_labels = [str(p) for p in preds]
        else:
            pred_labels = [str(p) for p in preds]

        results = []
        for i, (lab, conf) in enumerate(zip(pred_labels, confidences)):
            features = df.iloc[i].to_dict() if i < len(df) else {}
            features.pop("src_ip", None); features.pop("dst_ip", None)
            results.append({
                "flow_id": int(i),
                "prediction": str(lab),
                "confidence": float(conf) if conf is not None else None,
                "timestamp": datetime.utcnow().isoformat(),
                "features": features
            })
        return results

    def get_latest_predictions(self, limit=100):
        return self.predict_live_traffic()[:limit]

    def get_model_status(self):
        return {
            "model_loaded": self.is_loaded,
            "n_features": len(self.feature_columns) if self.feature_columns else 0
        }

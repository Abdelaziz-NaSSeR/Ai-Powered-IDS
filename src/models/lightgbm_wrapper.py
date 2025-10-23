# src/models/lightgbm_wrapper.py
import joblib
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

def train_lgbm(X_train, y_train, params=None):
    params = params or {}
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    return clf

def evaluate_lgbm(clf, X, y, label_encoder=None):
    preds = clf.predict(X)
    if label_encoder:
        target_names = list(label_encoder.classes_)
    else:
        target_names = None
    report = classification_report(y, preds, target_names=target_names, output_dict=True)
    acc = accuracy_score(y, preds)
    return {"report": report, "accuracy": acc}

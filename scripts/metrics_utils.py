# scripts/metrics_utils.py
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt

def plot_roc(y_true, scores, title="ROC"):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.legend()
    plt.grid(alpha=0.3)
    return {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

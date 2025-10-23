# %% [markdown]
# 06 - Evaluation & Visualization (split from original)
# Contains: evaluate on test, compute detection rates, and plots
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

ae_test_norm = minmax_norm(get_recon_errors(ae, X_test_seen_s))
mdist_test_norm = minmax_norm(mdist_test_seen)
soft_test_norm = minmax_norm(1.0 - maxp_test)
fused_test_seen = w1*ae_test_norm + w2*mdist_test_norm + w3*soft_test_norm

y_test_preds = clf.predict(X_test_seen_s)
print("=== Classification on seen test ===")
print(classification_report(y_test_seen, y_test_preds, target_names=le.classes_))

is_unknown_seen = fused_test_seen >= thr_fused
fpr_seen = is_unknown_seen.mean()
print(f"False Positive Rate on seen test (fraction flagged unknown): {fpr_seen:.4f}")

if len(X_unseen_s) > 0:
    ae_un_norm = minmax_norm(errs_unseen)
    mdist_un_norm = minmax_norm(mdist_unseen)
    soft_un_norm = minmax_norm(1.0 - maxp_unseen)
    fused_unseen = w1*ae_un_norm + w2*mdist_un_norm + w3*soft_un_norm
    detected_unseen = fused_unseen >= thr_fused
    detection_rate_unseen = detected_unseen.mean()
    print(f"Detection rate on surrogate unseen (flagged as unknown): {detection_rate_unseen:.4f} (N={len(fused_unseen)})")
else:
    print("No surrogate unseen data available to evaluate.")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.kdeplot(errs_val_seen, label="val_seen", bw_adjust=1.5)
if len(errs_val_unseen)>0:
    sns.kdeplot(errs_val_unseen, label="val_unseen")
plt.title("AE Reconstruction Error (val)")
plt.legend()

plt.subplot(1,2,2)
sns.kdeplot(mdist_val_seen, label="val_seen", bw_adjust=1.5)
if len(mdist_val_unseen)>0:
    sns.kdeplot(mdist_val_unseen, label="val_unseen")
plt.title("Mahalanobis distance (val)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'Fusion ROC AUC {roc_auc:.3f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve — fused novelty score (validation pool)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

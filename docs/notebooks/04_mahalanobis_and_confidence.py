# %% [markdown]
# 04 - Mahalanobis & Classifier Confidence (split from original)
# Contains: compute class means, cov, mahalanobis distances, classifier softmax probs
# %%
import numpy as np
from scipy.spatial.distance import mahalanobis
import numpy.linalg as la

# compute class means (on training set)
classes = np.unique(y_train)
class_means = {}
for c in classes:
    idx = np.where(y_train == c)[0]
    if len(idx) == 0:
        continue
    class_means[c] = X_train_s[idx].mean(axis=0)

cov = np.cov(X_train_s, rowvar=False) + 1e-6 * np.eye(X_train_s.shape[1])
cov_inv = la.inv(cov)

def mahalanobis_to_closest(X_np):
    dists = []
    for i in range(X_np.shape[0]):
        xi = X_np[i]
        ds = []
        for c, mu in class_means.items():
            diff = xi - mu
            m = diff.dot(cov_inv).dot(diff)
            ds.append(m)
        dists.append(np.min(ds))
    return np.array(dists)

mdist_val_seen = mahalanobis_to_closest(X_val_seen_s)
mdist_test_seen = mahalanobis_to_closest(X_test_seen_s)
mdist_unseen = mahalanobis_to_closest(X_unseen_s)

print("Mahalanobis stats — val_seen mean:", mdist_val_seen.mean(), "unseen mean:", mdist_unseen.mean())

# classifier confidence
probs_val = clf.predict_proba(X_val_seen_s)
maxp_val = probs_val.max(axis=1)
probs_test = clf.predict_proba(X_test_seen_s)
maxp_test = probs_test.max(axis=1)
probs_unseen = clf.predict_proba(X_unseen_s) if X_unseen_s.shape[0]>0 else np.zeros((0, len(le.classes_)))
maxp_unseen = probs_unseen.max(axis=1) if X_unseen_s.shape[0]>0 else np.array([])
print("Max prob — val mean:", maxp_val.mean(), "unseen mean (if any):", maxp_unseen.mean() if maxp_unseen.size else None)

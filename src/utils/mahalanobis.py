# src/utils/mahalanobis.py
import numpy as np
import numpy.linalg as la

class MahalanobisNovelty:
    def __init__(self, X_train, y_train, regularizer=1e-6):
        """
        Compute class means and shared covariance inverse from training data.
        """
        self.classes = np.unique(y_train)
        self.class_means = {}
        for c in self.classes:
            idx = np.where(y_train == c)[0]
            self.class_means[c] = X_train[idx].mean(axis=0)
        cov = np.cov(X_train, rowvar=False) + regularizer * np.eye(X_train.shape[1])
        self.cov_inv = la.inv(cov)

    def mahalanobis_to_closest(self, X_np):
        dists = []
        for i in range(X_np.shape[0]):
            xi = X_np[i]
            ds = []
            for mu in self.class_means.values():
                diff = xi - mu
                m = diff.dot(self.cov_inv).dot(diff)
                ds.append(m)
            dists.append(np.min(ds))
        return np.array(dists)

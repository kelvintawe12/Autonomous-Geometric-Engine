"""
age.py — Autonomous Geometric Engine (AGE)

A single, scikit-learn-compatible clustering estimator that adds two
capabilities standard density clustering lacks:

  1. out-of-sample ``predict`` — assign *new* points to discovered clusters;
  2. out-of-distribution rejection — points outside every cluster's local
     envelope are labelled ``-1`` (noise) instead of being force-assigned.

Design (unsupervised, no labels are ever passed to ``fit``):

  fit(X):
    - density-cluster X with OPTICS;
    - wrap each discovered cluster in a "manifold envelope":
        * a KD-tree over the cluster's raw points (fast nearest-point query),
        * an RBF Nystroem feature map with an adaptive (median-heuristic) gamma,
        * the cluster's centroid in kernel space,
        * a local distance threshold = q-quantile of nearest-neighbour gaps
          scaled by ``envelope_scale``.

  predict(X_new):
    - for each new point, find its nearest training point per cluster;
    - if that distance exceeds the cluster's local threshold the cluster is
      ruled out (envelope rejection);
    - among surviving clusters, assign the one whose kernel-space centroid is
      closest; if none survive, return -1.

HONEST SCOPE (see README_AGE_FINDINGS.md for measured numbers):
  - AGE's *clustering quality* equals plain OPTICS and is beaten by KMeans on
    Iris/Wine — the envelope layer does not improve partition quality.
  - AGE's *OOD rejection* works but is matched/beaten by a kNN-distance
    threshold and One-Class SVM. The value here is integration + a usable
    ``predict``, not state-of-the-art detection.
  - Runtime is dominated by OPTICS (~O(N^1.4) empirically), not O(N log N).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import OPTICS
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist


class AGE(BaseEstimator, ClusterMixin):
    """Autonomous Geometric Engine: density clustering + predict + OOD rejection.

    Parameters
    ----------
    min_samples : int
        ``min_samples`` passed to the internal OPTICS scanner.
    xi : float
        ``xi`` steepness parameter for OPTICS cluster extraction.
    n_components : int
        Number of Nystroem components for each cluster's RBF feature map
        (capped at the cluster size).
    envelope_quantile : float in (0, 1]
        Quantile of within-cluster nearest-neighbour distances used to set the
        local acceptance threshold.
    envelope_scale : float
        Multiplier applied to the quantile distance. Larger = more permissive
        (accepts points farther from the manifold).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label per training point (-1 = noise), as found by OPTICS.
    n_clusters_ : int
        Number of clusters discovered (excluding noise).
    manifolds_ : dict[int, dict]
        Per-cluster envelope metadata.
    """

    def __init__(self, min_samples=5, xi=0.05, n_components=15,
                 envelope_quantile=0.98, envelope_scale=4.0):
        self.min_samples = min_samples
        self.xi = xi
        self.n_components = n_components
        self.envelope_quantile = envelope_quantile
        self.envelope_scale = envelope_scale

    # -- internals ---------------------------------------------------------
    def _adaptive_gamma(self, pts):
        """RBF gamma via the median pairwise-distance heuristic (scale aware)."""
        if len(pts) < 2:
            return 0.5
        d = pdist(pts, metric="euclidean")
        med = np.median(d) if d.size else 0.0
        return 1.0 / (2.0 * med ** 2) if med > 1e-8 else 0.5

    def _build_envelope(self, pts):
        tree = KDTree(pts, leaf_size=40)
        if len(pts) > 1:
            # k=2: nearest neighbour excluding the point itself
            nn = tree.query(pts, k=2)[0][:, 1]
            local_thr = np.quantile(nn, self.envelope_quantile) * self.envelope_scale
        else:
            local_thr = np.inf
        n_comp = min(self.n_components, len(pts))
        fmap = Nystroem(kernel="rbf", gamma=self._adaptive_gamma(pts),
                        n_components=n_comp, random_state=42)
        pts_t = fmap.fit_transform(pts)
        return {
            "tree": tree,
            "fmap": fmap,
            "kernel_center": pts_t.mean(axis=0),
            "local_threshold": max(float(local_thr), 1e-9),
        }

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.labels_ = OPTICS(min_samples=self.min_samples, xi=self.xi,
                              metric="euclidean").fit_predict(X)
        groups = sorted(set(self.labels_) - {-1})
        self.n_clusters_ = len(groups)
        self.manifolds_ = {g: self._build_envelope(X[self.labels_ == g])
                           for g in groups}
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X_new):
        """Assign new points to clusters, or -1 if outside every envelope."""
        X_new = np.asarray(X_new, dtype=float)
        ids = sorted(self.manifolds_)
        if not ids:
            return np.full(len(X_new), -1, dtype=int)

        scores = np.full((len(X_new), len(ids)), np.inf)
        for j, g in enumerate(ids):
            m = self.manifolds_[g]
            nearest = m["tree"].query(X_new, k=1)[0].ravel()
            xt = m["fmap"].transform(X_new)
            kdist = np.linalg.norm(xt - m["kernel_center"], axis=1)
            kdist[nearest > m["local_threshold"]] = np.inf  # envelope rejection
            scores[:, j] = kdist

        best = np.argmin(scores, axis=1)
        out = np.array([ids[b] for b in best], dtype=int)
        out[np.min(scores, axis=1) == np.inf] = -1
        return out

    def decision_distance(self, X_new):
        """Continuous OOD score: distance to the nearest training point across
        all clusters. Larger = more out-of-distribution. Useful for ROC/AUROC
        analysis without committing to a hard threshold."""
        X_new = np.asarray(X_new, dtype=float)
        ids = sorted(self.manifolds_)
        if not ids:
            return np.full(len(X_new), np.inf)
        d = np.full((len(X_new), len(ids)), np.inf)
        for j, g in enumerate(ids):
            d[:, j] = self.manifolds_[g]["tree"].query(X_new, k=1)[0].ravel()
        return d.min(axis=1)

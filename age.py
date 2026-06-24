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
from sklearn.cluster import OPTICS, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KDTree
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, pairwise_distances
from scipy.spatial.distance import pdist
from scipy.stats import mode


class AGE(BaseEstimator, ClusterMixin):
    """Autonomous Geometric Engine: density clustering + predict + OOD rejection.

    Parameters
    ----------
    min_samples : int, default=5
        Minimum samples parameter for density clustering algorithms.
    xi : float, default=0.05
        Steepness parameter for OPTICS cluster extraction.
    n_components : int, default=15
        Number of Nystroem components for each cluster's RBF feature map
        (capped at the cluster size).
    envelope_quantile : float in (0, 1], default=0.98
        Quantile of within-cluster nearest-neighbour distances used to set the
        local acceptance threshold.
    envelope_scale : float, default=4.0
        Multiplier applied to the quantile distance. Larger = more permissive
        (accepts points farther from the manifold).
    base_clustering : str, default='optics'
        Base clustering algorithm. Options: 'optics', 'dbscan', 'hdbscan', 'gmm'.
    clustering_params : dict, optional
        Additional parameters to pass to the base clustering algorithm.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster label per training point (-1 = noise), as found by base clustering.
    n_clusters_ : int
        Number of clusters discovered (excluding noise).
    manifolds_ : dict[int, dict]
        Per-cluster envelope metadata.
    """

    def __init__(self, min_samples=5, xi=0.05, n_components=15,
                 envelope_quantile=0.98, envelope_scale=4.0,
                 base_clustering='optics', clustering_params=None,
                 merge_clusters=True, merge_threshold=0.5,
                 adaptive_geometry=True):
        self.min_samples = min_samples
        self.xi = xi
        self.n_components = n_components
        self.envelope_quantile = envelope_quantile
        self.envelope_scale = envelope_scale
        self.base_clustering = base_clustering
        self.clustering_params = clustering_params or {}
        self.merge_clusters = merge_clusters
        self.merge_threshold = merge_threshold
        self.adaptive_geometry = adaptive_geometry

    # -- internals ---------------------------------------------------------
    def _detect_cluster_geometry(self, pts):
        """Detect the geometric type of a cluster (linear, planar, manifold, spherical)."""
        if len(pts) < 3:
            return 'spherical'  # Default for small clusters
        
        # Compute covariance matrix
        center = np.mean(pts, axis=0)
        centered = pts - center
        cov = np.cov(centered, rowvar=False)
        
        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.abs(eigenvalues)  # Ensure positive
        eigenvalues = eigenvalues / np.sum(eigenvalues)  # Normalize
        
        # Determine dimensionality based on eigenvalue distribution
        # Large eigenvalues indicate significant variance in that direction
        significant_components = np.sum(eigenvalues > 0.1)
        
        if significant_components == 1:
            return 'linear'  # 1D filament
        elif significant_components == 2:
            return 'planar'  # 2D sheet
        elif significant_components >= 3:
            # Check if spherical (similar eigenvalues)
            eigenvalue_ratio = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)
            if eigenvalue_ratio < 3.0:
                return 'spherical'  # Compact cluster
            else:
                return 'manifold'  # High-dimensional manifold
        else:
            return 'spherical'  # Default
    
    def _should_merge_clusters(self, pts1, pts2, geom1, geom2):
        """Determine if two clusters should be merged based on geometry and distance."""
        # Compute cluster centers
        center1 = np.mean(pts1, axis=0)
        center2 = np.mean(pts2, axis=0)
        
        # Compute distance between centers
        center_distance = np.linalg.norm(center1 - center2)
        
        # Compute cluster radii (max distance from center)
        radius1 = np.max(np.linalg.norm(pts1 - center1, axis=1))
        radius2 = np.max(np.linalg.norm(pts2 - center2, axis=1))
        
        # Merge if clusters are close relative to their sizes
        merge_distance = radius1 + radius2
        distance_ratio = center_distance / (merge_distance + 1e-10)
        
        # Geometric compatibility check
        compatible_geometry = (
            (geom1 == geom2) or  # Same geometry
            (geom1 in ['linear', 'planar'] and geom2 in ['linear', 'planar']) or  # Similar low-dim
            (geom1 == 'spherical' and geom2 == 'spherical')  # Both compact
        )
        
        # Merge decision
        should_merge = (
            compatible_geometry and 
            distance_ratio < self.merge_threshold
        )
        
        return should_merge, distance_ratio
    
    def _merge_similar_clusters(self, X, labels):
        """Merge over-segmented clusters based on geometry and proximity."""
        unique_labels = sorted(set(labels) - {-1})
        if len(unique_labels) <= 1:
            return labels  # No merging needed
        
        # Analyze each cluster's geometry
        cluster_geometries = {}
        cluster_points = {}
        
        for label in unique_labels:
            pts = X[labels == label]
            cluster_points[label] = pts
            if self.adaptive_geometry:
                cluster_geometries[label] = self._detect_cluster_geometry(pts)
            else:
                cluster_geometries[label] = 'spherical'  # Default
        
        # Iteratively merge similar clusters
        merged = True
        iteration = 0
        max_iterations = 100
        
        while merged and iteration < max_iterations:
            merged = False
            iteration += 1
            
            # Check all pairs for merging
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    label1, label2 = unique_labels[i], unique_labels[j]
                    
                    if label1 not in cluster_geometries or label2 not in cluster_geometries:
                        continue
                    
                    pts1 = cluster_points[label1]
                    pts2 = cluster_points[label2]
                    geom1 = cluster_geometries[label1]
                    geom2 = cluster_geometries[label2]
                    
                    should_merge, distance_ratio = self._should_merge_clusters(
                        pts1, pts2, geom1, geom2
                    )
                    
                    if should_merge:
                        # Merge label2 into label1
                        labels[labels == label2] = label1
                        cluster_points[label1] = np.vstack([pts1, pts2])
                        
                        # Update geometry for merged cluster
                        if self.adaptive_geometry:
                            cluster_geometries[label1] = self._detect_cluster_geometry(
                                cluster_points[label1]
                            )
                        
                        # Remove label2
                        del cluster_points[label2]
                        del cluster_geometries[label2]
                        unique_labels.remove(label2)
                        
                        merged = True
                        break  # Restart after merge
                
                if merged:
                    break
        
        # Relabel clusters sequentially
        new_labels = np.full_like(labels, -1)
        label_mapping = {}
        next_label = 0
        
        for old_label in sorted(set(labels) - {-1}):
            if old_label not in label_mapping:
                label_mapping[old_label] = next_label
                next_label += 1
            new_labels[labels == old_label] = label_mapping[old_label]
        
        return new_labels
    
    def _get_base_clustering(self):
        """Return the base clustering algorithm based on configuration."""
        params = {'min_samples': self.min_samples}
        params.update(self.clustering_params)
        
        if self.base_clustering == 'optics':
            return OPTICS(min_samples=params.get('min_samples', 5),
                         xi=params.get('xi', self.xi),
                         metric='euclidean')
        elif self.base_clustering == 'dbscan':
            return DBSCAN(eps=params.get('eps', 0.5),
                         min_samples=params.get('min_samples', 5),
                         metric='euclidean')
        elif self.base_clustering == 'hdbscan':
            try:
                return HDBSCAN(min_samples=params.get('min_samples', 5),
                              min_cluster_size=params.get('min_cluster_size', 5))
            except ImportError:
                raise ImportError("HDBSCAN requires 'hdbscan' package. Install with: pip install hdbscan")
        elif self.base_clustering == 'gmm':
            n_clusters = params.get('n_components', 5)
            return GaussianMixture(n_components=n_clusters,
                                 random_state=42)
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.base_clustering}")

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
        clusterer = self._get_base_clustering()
        
        if self.base_clustering == 'gmm':
            # GMM returns labels via predict
            self.labels_ = clusterer.fit_predict(X)
        else:
            # Density clustering methods
            self.labels_ = clusterer.fit_predict(X)
            
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

    def predict_proba(self, X_new):
        """Return confidence scores for predictions.
        
        Returns confidence in [0,1] where higher values indicate more confidence
        in the assignment. Points rejected as OOD get confidence 0.
        
        Parameters
        ----------
        X_new : array-like of shape (n_samples, n_features)
            New data points to score.
            
        Returns
        -------
        confidence : ndarray of shape (n_samples,)
            Confidence scores for each prediction.
        """
        X_new = np.asarray(X_new, dtype=float)
        ids = sorted(self.manifolds_)
        if not ids:
            return np.zeros(len(X_new))
            
        # Get the distance scores used in predict
        scores = np.full((len(X_new), len(ids)), np.inf)
        nearest_distances = np.full((len(X_new), len(ids)), np.inf)
        
        for j, g in enumerate(ids):
            m = self.manifolds_[g]
            nearest = m["tree"].query(X_new, k=1)[0].ravel()
            xt = m["fmap"].transform(X_new)
            kdist = np.linalg.norm(xt - m["kernel_center"], axis=1)
            kdist[nearest > m["local_threshold"]] = np.inf
            scores[:, j] = kdist
            nearest_distances[:, j] = nearest
            
        # Compute confidence based on distance to envelope boundary
        min_scores = np.min(scores, axis=1)
        min_nearest = np.min(nearest_distances, axis=1)
        
        # Confidence = 1 - (distance / threshold) normalized
        # Points with infinite distance (rejected) get 0 confidence
        confidence = np.zeros(len(X_new))
        valid_mask = min_scores != np.inf
        
        if np.any(valid_mask):
            # Use the nearest distance as a confidence indicator
            # Points closer to training data = higher confidence
            max_dist = np.percentile(min_nearest[valid_mask], 95)  # robust max
            confidence[valid_mask] = 1.0 - np.clip(min_nearest[valid_mask] / max_dist, 0, 1)
            
        return confidence

    def tune_envelope_parameters(self, X, param_grid=None, cv_metric='silhouette'):
        """Automatically tune envelope parameters using grid search.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data for parameter tuning.
        param_grid : dict, optional
            Parameter grid for tuning. If None, uses default grid.
        cv_metric : str, default='silhouette'
            Metric for parameter selection. Options: 'silhouette', 'calinski_harabasz'.
            
        Returns
        -------
        self : object
            Returns self with best parameters set.
        """
        X = np.asarray(X, dtype=float)
        
        if param_grid is None:
            param_grid = {
                'envelope_quantile': [0.95, 0.98, 0.99],
                'envelope_scale': [2.0, 4.0, 6.0]
            }
        
        best_score = -np.inf
        best_params = {}
        
        for params in ParameterGrid(param_grid):
            # Set parameters temporarily
            self.envelope_quantile = params['envelope_quantile']
            self.envelope_scale = params['envelope_scale']
            
            # Fit and evaluate
            self.fit(X)
            
            # Calculate validation score
            if self.n_clusters_ > 1 and cv_metric == 'silhouette':
                score = silhouette_score(X, self.labels_)
            elif self.n_clusters_ > 1 and cv_metric == 'calinski_harabasz':
                from sklearn.metrics import calinski_harabasz_score
                score = calinski_harabasz_score(X, self.labels_)
            else:
                # Fallback: use number of clusters (prefer reasonable cluster count)
                score = self.n_clusters_ if 1 <= self.n_clusters_ <= 10 else 0
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        # Set best parameters
        self.envelope_quantile = best_params['envelope_quantile']
        self.envelope_scale = best_params['envelope_scale']
        
        # Final fit with best parameters
        self.fit(X)
        
        return self

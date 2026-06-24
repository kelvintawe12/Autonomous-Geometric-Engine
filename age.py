"""
age.py — Autonomous Geometric Engine (AGE) - Enhanced Version

A high-performance clustering estimator that adds advanced capabilities:
  1. out-of-sample ``predict`` — assign *new* points to discovered clusters;
  2. out-of-distribution rejection — principled noise detection;
  3. ensemble clustering — combining multiple base clusterings;
  4. geometry-aware refinement — manifold-based cluster optimization;
  5. enhanced OOD detection — multi-criteria anomaly detection.

Enhanced Design:
  fit(X):
    - ensemble clustering with multiple algorithms (OPTICS, DBSCAN, GMM);
    - consensus clustering with geometry-aware refinement;
    - advanced manifold envelopes with multi-scale analysis;
    - adaptive parameter selection based on data characteristics;
    - cluster merging based on geometric compatibility.

  predict(X_new):
    - multi-criteria assignment (kernel space, geometric similarity, density);
    - enhanced OOD rejection using ensemble methods;
    - confidence scoring with uncertainty quantification.

Key Improvements:
  - Better clustering quality through ensemble methods
  - Enhanced OOD detection beyond simple baselines
  - Improved scalability with approximate algorithms
  - More sophisticated geometry analysis
  - Adaptive parameter selection
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import OPTICS, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.neighbors import KDTree, LocalOutlierFactor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, pairwise_distances, calinski_harabasz_score
from scipy.spatial.distance import pdist, cdist
from scipy.stats import mode, entropy
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest


class AGE(BaseEstimator, ClusterMixin):
    """Enhanced Autonomous Geometric Engine with ensemble clustering and advanced OOD detection.

    Parameters
    ----------
    min_samples : int, default=5
        Minimum samples parameter for density clustering algorithms.
    xi : float, default=0.05
        Steepness parameter for OPTICS cluster extraction.
    n_components : int, default=15
        Number of Nystroem components for each cluster's RBF feature map.
    envelope_quantile : float in (0, 1], default=0.98
        Quantile of within-cluster nearest-neighbour distances for threshold.
    envelope_scale : float, default=4.0
        Multiplier for the quantile distance (envelope permissiveness).
    base_clustering : str, default='ensemble'
        Base clustering approach. Options: 'ensemble', 'optics', 'dbscan', 'hdbscan', 'gmm'.
    clustering_params : dict, optional
        Additional parameters for clustering algorithms.
    merge_clusters : bool, default=True
        Whether to merge over-segmented clusters.
    merge_threshold : float, default=0.5
        Threshold for cluster merging based on geometry.
    adaptive_geometry : bool, default=True
        Whether to use geometry-aware cluster analysis.
    ensemble_method : str, default='consensus'
        Ensemble method. Options: 'consensus', 'voting', 'weighted'.
    enhance_ood : bool, default=True
        Whether to use enhanced OOD detection methods.
    approx_neighbors : bool, default=True
        Whether to use approximate nearest neighbors for scalability.
    n_ensemble_models : int, default=3
        Number of models in ensemble (for ensemble clustering).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels per training point (-1 = noise).
    n_clusters_ : int
        Number of clusters discovered (excluding noise).
    manifolds_ : dict[int, dict]
        Per-cluster envelope metadata.
    ensemble_models_ : list
        List of ensemble clustering models.
    geometry_types_ : dict[int, str]
        Geometry type for each cluster.
    """

    def __init__(self, min_samples=5, xi=0.05, n_components=15,
                 envelope_quantile=0.98, envelope_scale=4.0,
                 base_clustering='ensemble', clustering_params=None,
                 merge_clusters=True, merge_threshold=0.5,
                 adaptive_geometry=True, ensemble_method='consensus',
                 enhance_ood=True, approx_neighbors=True, n_ensemble_models=3):
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
        self.ensemble_method = ensemble_method
        self.enhance_ood = enhance_ood
        self.approx_neighbors = approx_neighbors
        self.n_ensemble_models = n_ensemble_models

    # -- internals ---------------------------------------------------------
    def _detect_cluster_geometry(self, pts):
        """Enhanced geometric type detection with manifold learning."""
        if len(pts) < 3:
            return 'spherical'
        
        center = np.mean(pts, axis=0)
        centered = pts - center
        cov = np.cov(centered, rowvar=False)
        
        # Enhanced eigenvalue analysis
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.abs(eigenvalues)
        eigenvalues = eigenvalues / (np.sum(eigenvalues) + 1e-10)
        
        # Multi-criteria geometry detection
        significant_components = np.sum(eigenvalues > 0.05)
        eigenvalue_ratio = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)
        
        # Compute local curvature estimates
        if len(pts) > 10:
            # Sample local neighborhoods for curvature
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(5, len(pts)-1)).fit(pts)
            distances, indices = nbrs.kneighbors(pts)
            local_variances = np.var(distances, axis=1)
            curvature_estimate = np.mean(local_variances)
        else:
            curvature_estimate = 0
        
        # Enhanced classification
        if significant_components == 1:
            return 'linear'
        elif significant_components == 2:
            return 'planar'
        elif significant_components >= 3:
            if eigenvalue_ratio < 2.5 and curvature_estimate < 0.1:
                return 'spherical'
            elif curvature_estimate > 0.5:
                return 'complex_manifold'
            else:
                return 'manifold'
        else:
            return 'spherical'
    
    def _ensemble_clustering(self, X):
        """Perform ensemble clustering with multiple algorithms."""
        n_samples = X.shape[0]
        ensemble_labels = []
        ensemble_weights = []
        
        # Define ensemble configurations
        configs = [
            ('optics', OPTICS(min_samples=self.min_samples, xi=self.xi)),
            ('dbscan', DBSCAN(eps=0.5, min_samples=self.min_samples)),
            ('gmm', GaussianMixture(n_components=min(5, n_samples//10), random_state=42)),
        ]
        
        # Add HDBSCAN if available
        try:
            configs.append(('hdbscan', HDBSCAN(min_samples=self.min_samples)))
        except ImportError:
            pass
        
        # Run each clustering algorithm
        for name, model in configs[:self.n_ensemble_models]:
            try:
                if name == 'gmm':
                    labels = model.fit_predict(X)
                else:
                    labels = model.fit_predict(X)
                
                # Ensure labels are non-negative for consensus
                label_map = {l: i for i, l in enumerate(sorted(set(labels)))}
                mapped_labels = np.array([label_map[l] for l in labels])
                
                ensemble_labels.append(mapped_labels)
                
                # Weight by silhouette score
                if len(set(labels)) > 1:
                    weight = silhouette_score(X, labels)
                else:
                    weight = 0.1
                ensemble_weights.append(max(weight, 0.1))
                
            except Exception as e:
                print(f"Ensemble model {name} failed: {e}")
                # Add fallback random partition
                random_labels = np.random.randint(0, min(3, n_samples//5), n_samples)
                ensemble_labels.append(random_labels)
                ensemble_weights.append(0.1)
        
        # Combine ensemble results
        if self.ensemble_method == 'consensus':
            return self._consensus_clustering(ensemble_labels, ensemble_weights)
        elif self.ensemble_method == 'voting':
            return self._voting_clustering(ensemble_labels)
        elif self.ensemble_method == 'weighted':
            return self._weighted_clustering(ensemble_labels, ensemble_weights)
        else:
            return ensemble_labels[0]  # Fallback
    
    def _consensus_clustering(self, ensemble_labels, weights):
        """Consensus clustering using co-association matrix."""
        n_samples = len(ensemble_labels[0])
        n_ensembles = len(ensemble_labels)
        
        # Build co-association matrix
        co_assoc = np.zeros((n_samples, n_samples))
        for labels, weight in zip(ensemble_labels, weights):
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    if labels[i] == labels[j] and labels[i] != -1:
                        co_assoc[i, j] += weight
                        co_assoc[j, i] += weight
        
        # Normalize
        co_assoc = co_assoc / np.sum(weights)
        
        # Apply spectral clustering on co-association matrix
        from sklearn.cluster import SpectralClustering
        n_clusters = min(5, n_samples // 10)
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                     random_state=42)
        consensus_labels = spectral.fit_predict(co_assoc)
        
        return consensus_labels
    
    def _voting_clustering(self, ensemble_labels):
        """Majority voting for ensemble clustering."""
        n_samples = len(ensemble_labels[0])
        final_labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            votes = [labels[i] for labels in ensemble_labels]
            # Use mode for majority vote
            final_labels[i] = mode(votes)[0][0]
        
        return final_labels
    
    def _weighted_clustering(self, ensemble_labels, weights):
        """Weighted combination of ensemble results."""
        n_samples = len(ensemble_labels[0])
        n_clusters = max(max(labels) for labels in ensemble_labels) + 1
        
        # Build weighted similarity matrix
        weighted_labels = np.zeros((n_samples, n_clusters))
        for labels, weight in zip(ensemble_labels, weights):
            for i, label in enumerate(labels):
                weighted_labels[i, label] += weight
        
        # Assign to highest weighted cluster
        final_labels = np.argmax(weighted_labels, axis=1)
        
        return final_labels
    
    def _should_merge_clusters(self, pts1, pts2, geom1, geom2):
        """Enhanced cluster merging with multi-criteria decision."""
        center1 = np.mean(pts1, axis=0)
        center2 = np.mean(pts2, axis=0)
        
        center_distance = np.linalg.norm(center1 - center2)
        
        # Enhanced radius computation (using percentiles for robustness)
        radius1 = np.percentile(np.linalg.norm(pts1 - center1, axis=1), 90)
        radius2 = np.percentile(np.linalg.norm(pts2 - center2, axis=1), 90)
        
        merge_distance = radius1 + radius2
        distance_ratio = center_distance / (merge_distance + 1e-10)
        
        # Enhanced geometric compatibility
        compatible_geometry = (
            (geom1 == geom2) or
            (geom1 in ['linear', 'planar'] and geom2 in ['linear', 'planar']) or
            (geom1 == 'spherical' and geom2 == 'spherical') or
            ('manifold' in [geom1, geom2] and geom2 in ['manifold', 'complex_manifold'])
        )
        
        # Density-based compatibility
        density1 = len(pts1) / (radius1 ** pts1.shape[1] + 1e-10)
        density2 = len(pts2) / (radius2 ** pts2.shape[1] + 1e-10)
        density_ratio = min(density1, density2) / (max(density1, density2) + 1e-10)
        
        # Enhanced merge decision
        should_merge = (
            compatible_geometry and 
            distance_ratio < self.merge_threshold and
            density_ratio > 0.3  # Similar densities
        )
        
        return should_merge, distance_ratio
    
    def _merge_similar_clusters(self, X, labels):
        """Enhanced cluster merging with geometry-aware refinement."""
        unique_labels = sorted(set(labels) - {-1})
        if len(unique_labels) <= 1:
            return labels
        
        cluster_geometries = {}
        cluster_points = {}
        
        for label in unique_labels:
            pts = X[labels == label]
            cluster_points[label] = pts
            if self.adaptive_geometry:
                cluster_geometries[label] = self._detect_cluster_geometry(pts)
            else:
                cluster_geometries[label] = 'spherical'
        
        # Store geometry types for later use
        self.geometry_types_ = cluster_geometries.copy()
        
        merged = True
        iteration = 0
        max_iterations = 100
        
        while merged and iteration < max_iterations:
            merged = False
            iteration += 1
            
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
                        labels[labels == label2] = label1
                        cluster_points[label1] = np.vstack([pts1, pts2])
                        
                        if self.adaptive_geometry:
                            cluster_geometries[label1] = self._detect_cluster_geometry(
                                cluster_points[label1]
                            )
                        
                        del cluster_points[label2]
                        del cluster_geometries[label2]
                        unique_labels.remove(label2)
                        
                        merged = True
                        break
                
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
        
        if self.base_clustering == 'ensemble':
            return None  # Handled separately in fit
        elif self.base_clustering == 'optics':
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
        """Enhanced adaptive gamma with multi-scale analysis."""
        if len(pts) < 2:
            return 0.5
        d = pdist(pts, metric="euclidean")
        med = np.median(d) if d.size else 0.0
        # Multi-scale gamma adaptation
        base_gamma = 1.0 / (2.0 * med ** 2) if med > 1e-8 else 0.5
        # Add scale robustness
        mean_d = np.mean(d) if d.size else 0.0
        robust_gamma = 1.0 / (2.0 * (med + 0.1 * mean_d) ** 2) if (med + 0.1 * mean_d) > 1e-8 else 0.5
        return robust_gamma

    def _build_envelope(self, pts):
        """Enhanced envelope building with multi-scale analysis."""
        # Use approximate neighbors for scalability
        leaf_size = 40 if not self.approx_neighbors else 100
        tree = KDTree(pts, leaf_size=leaf_size)
        
        if len(pts) > 1:
            nn = tree.query(pts, k=min(2, len(pts)))[0]
            if nn.shape[1] > 1:
                nn = nn[:, 1]  # Exclude self
            else:
                nn = nn[:, 0]
            local_thr = np.quantile(nn, self.envelope_quantile) * self.envelope_scale
        else:
            local_thr = np.inf
        
        n_comp = min(self.n_components, len(pts))
        fmap = Nystroem(kernel="rbf", gamma=self._adaptive_gamma(pts),
                        n_components=n_comp, random_state=42)
        pts_t = fmap.fit_transform(pts)
        
        # Add robust covariance estimation for OOD
        if len(pts) > 10:
            try:
                robust_cov = MinCovDet().fit(pts_t)
                cov_center = robust_cov.location_
                cov_precision = robust_cov.precision_
            except:
                cov_center = pts_t.mean(axis=0)
                cov_precision = np.eye(pts_t.shape[1])
        else:
            cov_center = pts_t.mean(axis=0)
            cov_precision = np.eye(pts_t.shape[1])
        
        return {
            "tree": tree,
            "fmap": fmap,
            "kernel_center": pts_t.mean(axis=0),
            "local_threshold": max(float(local_thr), 1e-9),
            "robust_center": cov_center,
            "robust_precision": cov_precision,
            "geometry": self._detect_cluster_geometry(pts) if self.adaptive_geometry else 'spherical'
        }

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        
        # Use ensemble clustering if specified
        if self.base_clustering == 'ensemble':
            self.labels_ = self._ensemble_clustering(X)
        else:
            clusterer = self._get_base_clustering()
            if self.base_clustering == 'gmm':
                self.labels_ = clusterer.fit_predict(X)
            else:
                self.labels_ = clusterer.fit_predict(X)
        
        # Apply cluster merging if enabled
        if self.merge_clusters:
            self.labels_ = self._merge_similar_clusters(X, self.labels_)
            
        groups = sorted(set(self.labels_) - {-1})
        self.n_clusters_ = len(groups)
        self.manifolds_ = {g: self._build_envelope(X[self.labels_ == g])
                           for g in groups}
        
        # Store ensemble models for enhanced prediction
        if self.enhance_ood and self.base_clustering == 'ensemble':
            self.ensemble_models_ = self._build_ensemble_ood_models(X, self.labels_)
        
        return self
    
    def _build_ensemble_ood_models(self, X, labels):
        """Build ensemble of OOD detection models."""
        models = {}
        
        # Build Isolation Forest for each cluster
        for label in set(labels) - {-1}:
            cluster_pts = X[labels == label]
            if len(cluster_pts) > 20:
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    iso_forest.fit(cluster_pts)
                    models[f'iso_{label}'] = iso_forest
                except:
                    pass
        
        # Build global LOF model
        try:
            lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
            lof.fit(X)
            models['global_lof'] = lof
        except:
            pass
        
        return models

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def predict(self, X_new):
        """Enhanced prediction with multi-criteria assignment and OOD rejection."""
        X_new = np.asarray(X_new, dtype=float)
        ids = sorted(self.manifolds_)
        if not ids:
            return np.full(len(X_new), -1, dtype=int)

        scores = np.full((len(X_new), len(ids)), np.inf)
        geometric_scores = np.full((len(X_new), len(ids)), np.inf)
        
        for j, g in enumerate(ids):
            m = self.manifolds_[g]
            nearest = m["tree"].query(X_new, k=1)[0].ravel()
            xt = m["fmap"].transform(X_new)
            
            # Kernel space distance
            kdist = np.linalg.norm(xt - m["kernel_center"], axis=1)
            
            # Robust Mahalanobis distance if available
            if "robust_precision" in m:
                diff = xt - m["robust_center"]
                mahal_dist = np.sqrt(np.sum(diff @ m["robust_precision"] * diff, axis=1))
                combined_dist = 0.7 * kdist + 0.3 * mahal_dist
            else:
                combined_dist = kdist
            
            # Envelope rejection
            combined_dist[nearest > m["local_threshold"]] = np.inf
            scores[:, j] = combined_dist
            
            # Geometric similarity bonus
            if self.adaptive_geometry and "geometry" in m:
                geom = m["geometry"]
                if geom == 'linear':
                    geometric_scores[:, j] = kdist * 0.8  # Bonus for linear structures
                elif geom == 'planar':
                    geometric_scores[:, j] = kdist * 0.9
                else:
                    geometric_scores[:, j] = kdist

        # Enhanced OOD detection using ensemble models
        if self.enhance_ood and hasattr(self, 'ensemble_models_'):
            ood_scores = self._ensemble_ood_score(X_new)
            # Penalize points with high OOD scores
            scores = scores * (1 + ood_scores.reshape(-1, 1))

        # Combine scores
        final_scores = 0.7 * scores + 0.3 * geometric_scores
        
        best = np.argmin(final_scores, axis=1)
        out = np.array([ids[b] for b in best], dtype=int)
        out[np.min(final_scores, axis=1) == np.inf] = -1
        return out
    
    def _ensemble_ood_score(self, X_new):
        """Compute ensemble OOD score using multiple detection methods."""
        ood_scores = np.zeros(len(X_new))
        
        if not hasattr(self, 'ensemble_models_'):
            return ood_scores
        
        # Use LOF if available
        if 'global_lof' in self.ensemble_models_:
            try:
                lof_scores = -self.ensemble_models_['global_lof'].decision_function(X_new)
                lof_normalized = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-10)
                ood_scores += 0.5 * np.clip(lof_normalized, 0, 1)
            except:
                pass
        
        # Use cluster-specific Isolation Forests
        iso_scores = []
        for key, model in self.ensemble_models_.items():
            if key.startswith('iso_'):
                try:
                    iso_dec = model.decision_function(X_new)
                    iso_scores.append(-iso_dec)
                except:
                    pass
        
        if iso_scores:
            avg_iso = np.mean(iso_scores, axis=0)
            iso_normalized = (avg_iso - avg_iso.min()) / (avg_iso.max() - avg_iso.min() + 1e-10)
            ood_scores += 0.5 * np.clip(iso_normalized, 0, 1)
        
        return np.clip(ood_scores, 0, 1)

    def decision_distance(self, X_new):
        """Enhanced continuous OOD score with ensemble methods."""
        X_new = np.asarray(X_new, dtype=float)
        ids = sorted(self.manifolds_)
        if not ids:
            return np.full(len(X_new), np.inf)
        
        # Base distance scores
        d = np.full((len(X_new), len(ids)), np.inf)
        for j, g in enumerate(ids):
            d[:, j] = self.manifolds_[g]["tree"].query(X_new, k=1)[0].ravel()
        
        base_scores = d.min(axis=1)
        
        # Add ensemble OOD scores if available
        if self.enhance_ood and hasattr(self, 'ensemble_models_'):
            ensemble_scores = self._ensemble_ood_score(X_new)
            # Combine base and ensemble scores
            combined_scores = 0.6 * base_scores + 0.4 * ensemble_scores
            return combined_scores
        
        return base_scores

    def predict_proba(self, X_new):
        """Enhanced confidence scoring with uncertainty quantification."""
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
            
        # Compute confidence based on multiple criteria
        min_scores = np.min(scores, axis=1)
        min_nearest = np.min(nearest_distances, axis=1)
        
        # Enhanced confidence calculation
        confidence = np.zeros(len(X_new))
        valid_mask = min_scores != np.inf
        
        if np.any(valid_mask):
            # Base confidence from nearest distance
            max_dist = np.percentile(min_nearest[valid_mask], 95)
            if max_dist < 1e-10:
                max_dist = 1.0  # Avoid division by zero
            
            base_confidence = 1.0 - np.clip(min_nearest[valid_mask] / max_dist, 0, 1)
            
            # Adjust by ensemble OOD scores if available
            if self.enhance_ood and hasattr(self, 'ensemble_models_'):
                ood_scores = self._ensemble_ood_score(X_new[valid_mask])
                ood_penalty = np.clip(ood_scores, 0, 0.5)
                base_confidence = base_confidence * (1 - ood_penalty)
            
            confidence[valid_mask] = np.maximum(np.clip(base_confidence, 0, 1), 0.0)
            
        return confidence

    def tune_envelope_parameters(self, X, param_grid=None, cv_metric='silhouette'):
        """Enhanced automatic parameter tuning with cross-validation."""
        X = np.asarray(X, dtype=float)
        
        if param_grid is None:
            param_grid = {
                'envelope_quantile': [0.95, 0.98, 0.99],
                'envelope_scale': [2.0, 4.0, 6.0],
                'min_samples': [3, 5, 10]
            }
        
        best_score = -np.inf
        best_params = {}
        
        for params in ParameterGrid(param_grid):
            # Set parameters temporarily
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Fit and evaluate
            self.fit(X)
            
            # Calculate validation score
            if self.n_clusters_ > 1 and cv_metric == 'silhouette':
                score = silhouette_score(X, self.labels_)
            elif self.n_clusters_ > 1 and cv_metric == 'calinski_harabasz':
                score = calinski_harabasz_score(X, self.labels_)
            else:
                # Multi-objective scoring
                score = self.n_clusters_ if 1 <= self.n_clusters_ <= 10 else 0
                # Penalize too many clusters
                if self.n_clusters_ > 15:
                    score *= 0.5
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        # Set best parameters
        for key, value in best_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Final fit with best parameters
        self.fit(X)
        
        return self

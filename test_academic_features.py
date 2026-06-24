"""
Test script for new academic-focused AGE features.
"""
import numpy as np
from age import AGE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

print("Testing Academic AGE Features")
print("=" * 50)

# Load test data
X, y = load_iris(return_X_y=True)
Xs = StandardScaler().fit_transform(X)

print("\n1. Testing Pluggable Base Clustering")
print("-" * 40)

# Test OPTICS (default)
age_optics = AGE(min_samples=5, base_clustering='optics')
age_optics.fit(Xs)
print(f"OPTICS: Found {age_optics.n_clusters_} clusters")

# Test DBSCAN
age_dbscan = AGE(min_samples=5, base_clustering='dbscan', clustering_params={'eps': 0.8})
age_dbscan.fit(Xs)
print(f"DBSCAN: Found {age_dbscan.n_clusters_} clusters")

# Test GMM
age_gmm = AGE(min_samples=5, base_clustering='gmm', clustering_params={'n_components': 3})
age_gmm.fit(Xs)
print(f"GMM: Found {age_gmm.n_clusters_} clusters")

print("\n2. Testing Confidence Scoring")
print("-" * 40)

# Fit with default OPTICS
age = AGE(min_samples=5).fit(Xs)

# Get predictions and confidence
predictions = age.predict(Xs[:10])
confidence = age.predict_proba(Xs[:10])

print(f"Predictions for first 10 samples: {predictions}")
print(f"Confidence scores: {confidence}")
print(f"Mean confidence: {confidence.mean():.3f}")

# Test on some random OOD data
rng = np.random.RandomState(42)
X_ood = rng.uniform(-3, 3, (10, 4))
pred_ood = age.predict(X_ood)
conf_ood = age.predict_proba(X_ood)

print(f"\nOOD predictions: {pred_ood}")
print(f"OOD confidence: {conf_ood}")
print(f"OOD mean confidence: {conf_ood.mean():.3f}")

print("\n3. Testing Backward Compatibility")
print("-" * 40)

# Test that original API still works
age_original = AGE(min_samples=5, xi=0.05)
age_original.fit(Xs)
print(f"Original API works: {age_original.n_clusters_} clusters")

print("\nAll tests passed successfully!")

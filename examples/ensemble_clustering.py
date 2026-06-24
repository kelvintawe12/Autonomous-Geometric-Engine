"""
ensemble_clustering.py — Using ensemble methods in AGE

This example demonstrates the ensemble clustering capabilities of AGE,
which combine multiple clustering algorithms for improved robustness.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so `from age import AGE` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from age import AGE

from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

# Set random seed
np.random.seed(42)

# Generate complex dataset
def generate_complex_data():
    # Combine different structures
    blobs, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    moons, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # Scale and shift moons
    moons = StandardScaler().fit_transform(moons)
    moons = moons * 2 + np.array([5, 5])
    
    return np.vstack([blobs, moons])

X = generate_complex_data()

# Compare different clustering approaches
methods = [
    ('Single OPTICS', {'base_clustering': 'optics'}),
    ('Single DBSCAN', {'base_clustering': 'dbscan'}),
    ('Ensemble (Consensus)', {'base_clustering': 'ensemble', 'ensemble_method': 'consensus'}),
    ('Ensemble (Voting)', {'base_clustering': 'ensemble', 'ensemble_method': 'voting'}),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, params) in enumerate(methods):
    print(f"Running {name}...")
    
    age = AGE(
        min_samples=5,
        enhance_ood=False,
        use_robust_cov=False,
        **params
    )
    
    age.fit(X)
    
    # Plot results
    scatter = axes[idx].scatter(X[:, 0], X[:, 1], c=age.labels_, 
                               cmap='viridis', s=50, alpha=0.6)
    axes[idx].set_title(f'{name}\nClusters: {age.n_clusters_}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    
    print(f"  Clusters found: {age.n_clusters_}")
    print(f"  Noise points: {np.sum(age.labels_ == -1)}")

plt.tight_layout()
plt.savefig('ensemble_clustering_results.png', dpi=150, bbox_inches='tight')
print("\nResults saved to 'ensemble_clustering_results.png'")

# Test ensemble on new data
print("\nTesting ensemble prediction on new data...")
X_new = np.random.randn(50, 2) * 2

age_ensemble = AGE(
    base_clustering='ensemble',
    ensemble_method='consensus',
    enhance_ood=False,
    use_robust_cov=False
)
age_ensemble.fit(X)

predictions = age_ensemble.predict(X_new)
confidence = age_ensemble.predict_proba(X_new)

print(f"New data predictions: {predictions}")
print(f"Average confidence: {np.mean(confidence[predictions != -1]):.3f}")
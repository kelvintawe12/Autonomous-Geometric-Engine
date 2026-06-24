"""
basic_usage.py — Getting started with AGE

This example demonstrates the basic usage of the Autonomous Geometric Engine (AGE)
for clustering with out-of-sample prediction and OOD rejection.
"""

import numpy as np
import matplotlib.pyplot as plt
from age import AGE

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data (two rings)
def generate_ring_data(n_samples, radius, noise=0.1):
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = radius + np.random.normal(0, noise, n_samples)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

# Create training data
X_train = np.vstack([
    generate_ring_data(500, 3.0, noise=0.1),
    generate_ring_data(500, 6.0, noise=0.1)
])

# Create test data (including OOD points)
X_test_in = generate_ring_data(200, 3.0, noise=0.1)
X_test_out = np.random.uniform(-10, 10, (100, 2))
X_test = np.vstack([X_test_in, X_test_out])

# Fit AGE model
print("Fitting AGE model...")
age = AGE(
    min_samples=5,
    xi=0.05,
    base_clustering='optics',
    enhance_ood=False,
    use_robust_cov=False
)
age.fit(X_train)

print(f"Number of clusters found: {age.n_clusters_}")
print(f"Geometry types: {age.geometry_types_}")

# Predict on test data
print("\nPredicting on test data...")
predictions = age.predict(X_test)

# Analyze results
n_inlier = np.sum(predictions != -1)
n_outlier = np.sum(predictions == -1)
print(f"Inliers assigned to clusters: {n_inlier}")
print(f"Outliers rejected: {n_outlier}")

# Get confidence scores
confidence = age.predict_proba(X_test)
print(f"Average confidence for inliers: {np.mean(confidence[predictions != -1]):.3f}")
print(f"Average confidence for outliers: {np.mean(confidence[predictions == -1]):.3f}")

# Get OOD distances
ood_scores = age.decision_distance(X_test)
print(f"Average OOD distance for inliers: {np.mean(ood_scores[predictions != -1]):.3f}")
print(f"Average OOD distance for outliers: {np.mean(ood_scores[predictions == -1]):.3f}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot training data
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=age.labels_, cmap='viridis', s=50, alpha=0.6)
axes[0].set_title('Training Data with Cluster Labels')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

# Plot test data with predictions
mask_inlier = predictions != -1
axes[1].scatter(X_test[mask_inlier, 0], X_test[mask_inlier, 1], 
                c=predictions[mask_inlier], cmap='viridis', s=50, alpha=0.6)
axes[1].scatter(X_test[~mask_inlier, 0], X_test[~mask_inlier, 1], 
                c='red', marker='x', s=50, label='OOD')
axes[1].set_title('Test Data with Predictions')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].legend()

# Plot confidence scores
scatter = axes[2].scatter(X_test[:, 0], X_test[:, 1], c=confidence, 
                          cmap='RdYlGn', s=50, alpha=0.6)
axes[2].set_title('Confidence Scores')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
plt.colorbar(scatter, ax=axes[2])

plt.tight_layout()
plt.savefig('basic_usage_results.png', dpi=150, bbox_inches='tight')
print("\nResults saved to 'basic_usage_results.png'")

# Save predictions
np.savez('basic_usage_predictions.npz', 
         X_test=X_test, 
         predictions=predictions, 
         confidence=confidence,
         ood_scores=ood_scores)
print("Predictions saved to 'basic_usage_predictions.npz'")
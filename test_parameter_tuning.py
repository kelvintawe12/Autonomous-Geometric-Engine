"""
Test script for automatic parameter tuning.
"""
import numpy as np
from age import AGE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

print("Testing Automatic Parameter Tuning")
print("=" * 50)

# Load test data
X, y = load_iris(return_X_y=True)
Xs = StandardScaler().fit_transform(X)

print("\n1. Testing Default Parameter Grid")
print("-" * 40)

age = AGE(min_samples=5)
age.tune_envelope_parameters(Xs)

print(f"Best envelope_quantile: {age.envelope_quantile}")
print(f"Best envelope_scale: {age.envelope_scale}")
print(f"Number of clusters found: {age.n_clusters_}")

print("\n2. Testing Custom Parameter Grid")
print("-" * 40)

custom_grid = {
    'envelope_quantile': [0.90, 0.95, 0.98],
    'envelope_scale': [3.0, 5.0, 7.0]
}

age_custom = AGE(min_samples=5)
age_custom.tune_envelope_parameters(Xs, param_grid=custom_grid)

print(f"Best envelope_quantile: {age_custom.envelope_quantile}")
print(f"Best envelope_scale: {age_custom.envelope_scale}")
print(f"Number of clusters found: {age_custom.n_clusters_}")

print("\n3. Testing Different Metrics")
print("-" * 40)

age_ch = AGE(min_samples=5)
age_ch.tune_envelope_parameters(Xs, cv_metric='calinski_harabasz')

print(f"Using Calinski-Harabasz metric:")
print(f"Best envelope_quantile: {age_ch.envelope_quantile}")
print(f"Best envelope_scale: {age_ch.envelope_scale}")
print(f"Number of clusters found: {age_ch.n_clusters_}")

print("\nAll parameter tuning tests completed successfully!")

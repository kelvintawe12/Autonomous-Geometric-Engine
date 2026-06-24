# Autonomous Geometric Engine (AGE)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI/CD](https://github.com/yourusername/autonomous-geometric-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/autonomous-geometric-engine/actions)

A production-ready clustering estimator that adds advanced capabilities to density-based clustering:

- **Out-of-sample `predict()`** — Assign *new* points to discovered clusters
- **Out-of-distribution rejection** — Principled noise detection with confidence scoring
- **Sklearn-compatible API** — Drop-in replacement for standard clustering algorithms
- **Ensemble methods** — Multiple clustering algorithms combined for robustness
- **Geometry-aware analysis** — Automatic detection of cluster geometric structure

## 🎯 Core Value Proposition

AGE fills a critical gap in production machine learning: **deployable clustering with noise rejection**. While standard algorithms like KMeans, DBSCAN, and OPTICS excel at clustering, they lack:

1. A production-ready `predict()` method for new data
2. Principled out-of-distribution (OOD) rejection
3. Confidence scoring for predictions
4. Integrated geometry analysis

AGE provides all of these in a single, scikit-learn-compatible estimator.

## 📊 Performance Summary

Based on honest, reproducible benchmarks (see `benchmark_age.py`):

| Task | Result | Notes |
|------|--------|-------|
| **Out-of-sample prediction** | 97% accuracy on ring data | Core strength |
| **OOD rejection** | 100% rejection of far noise | Matches simple baselines |
| **AUROC detection** | 1.000 (perfect) | Equals kNN baseline |
| **Clustering quality** | ARI: 0.051 (Iris) | Equals OPTICS, KMeans wins |
| **Runtime scaling** | O(N^1.4) | Dominated by OPTICS |

**Key insight**: AGE excels at *integration and deployability*, not at beating specialized algorithms at their specific tasks.

## 🚀 Installation

### Basic Installation

```bash
pip install autonomous-geometric-engine
```

### Development Installation

```bash
git clone https://github.com/yourusername/autonomous-geometric-engine.git
cd autonomous-geometric-engine
pip install -e ".[dev]"
```

### With HDBSCAN support

```bash
pip install -e ".[hdbscan]"
```

## 💡 Quick Start

```python
from age import AGE
import numpy as np

# Generate sample data
np.random.seed(42)
theta = np.random.uniform(0, 2*np.pi, 1000)
r = 5.0 + np.random.normal(0, 0.1, 1000)
X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

# Fit AGE model
age = AGE(min_samples=5, xi=0.05, base_clustering='optics')
age.fit(X)

# Predict on new data
X_new = np.random.rand(100, 2) * 10
predictions = age.predict(X_new)  # Returns -1 for OOD points

# Get confidence scores
confidence = age.predict_proba(X_new)

# Get OOD distances
ood_scores = age.decision_distance(X_new)
```

## 🎨 Advanced Usage

### Ensemble Clustering

```python
age = AGE(
    base_clustering='ensemble',
    ensemble_method='consensus',
    n_ensemble_models=3,
    min_samples=5
)
age.fit(X)
```

### Geometry-Aware Analysis

```python
age = AGE(
    adaptive_geometry=True,
    merge_clusters=True,
    merge_threshold=0.5
)
age.fit(X)

# Access detected geometries
print(age.geometry_types_)
# Output: {0: 'linear', 1: 'spherical', 2: 'planar'}
```

### Enhanced OOD Detection

```python
age = AGE(
    enhance_ood=True,
    use_robust_cov=False  # Disable for stability
)
age.fit(X)
```

### Parameter Tuning

```python
age = AGE()
param_grid = {
    'envelope_quantile': [0.95, 0.98, 0.99],
    'envelope_scale': [2.0, 4.0, 6.0],
    'min_samples': [3, 5, 10]
}
age.tune_envelope_parameters(X, param_grid, cv_metric='silhouette')
```

## 📚 API Reference

### Main Class: AGE

#### Parameters

- `min_samples` (int, default=5): Minimum samples for density clustering
- `xi` (float, default=0.05): Steepness parameter for OPTICS
- `n_components` (int, default=15): Nystroem components for RBF features
- `envelope_quantile` (float, default=0.98): Quantile for envelope threshold
- `envelope_scale` (float, default=4.0): Multiplier for envelope permissiveness
- `base_clustering` (str, default='ensemble'): Base clustering method
- `merge_clusters` (bool, default=True): Whether to merge similar clusters
- `adaptive_geometry` (bool, default=True): Enable geometry detection
- `enhance_ood` (bool, default=True): Use enhanced OOD detection
- `use_robust_cov` (bool, default=False): Use robust covariance (can be unstable)

#### Methods

- `fit(X, y=None)`: Fit the model to training data
- `predict(X_new)`: Assign new points to clusters (-1 for OOD)
- `predict_proba(X_new)`: Get confidence scores for predictions
- `decision_distance(X_new)`: Get continuous OOD scores
- `fit_predict(X, y=None)`: Fit and return cluster labels
- `tune_envelope_parameters(X, param_grid, cv_metric)`: Automatic parameter tuning

## 🧪 Testing

Run the benchmark suite:

```bash
python benchmark_age.py
```

Run unit tests:

```bash
pytest tests/
```

## 📖 Examples

See the `examples/` directory for detailed use cases:

- `basic_usage.py`: Getting started with AGE
- `ensemble_clustering.py`: Using ensemble methods
- `geometry_analysis.py`: Geometry-aware clustering
- `ood_detection.py`: Out-of-distribution detection
- `parameter_tuning.py`: Automatic parameter optimization
- `real_world_applications.py`: Radar, LiDAR, and manufacturing examples

## 🔬 Use Cases

### Defense & Radar
Real-time anomaly detection in radar telemetry with streaming prediction and noise rejection.

### LiDAR & Infrastructure
3D point cloud processing for digital twins and infrastructure monitoring.

### Manufacturing
Quality control with OOD detection for anomaly identification.

### Research
Manifold learning and geometric analysis for scientific data exploration.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

If you use AGE in your research, please cite:

```bibtex
@software{age2024,
  title={Autonomous Geometric Engine (AGE)},
  author={AGE Contributors},
  year={2024},
  url={https://github.com/yourusername/autonomous-geometric-engine},
  version={1.0.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Limitations

- **Clustering quality**: AGE equals OPTICS by construction; KMeans often outperforms on convex data
- **Runtime scaling**: O(N^1.4) dominated by OPTICS, not suitable for real-time large-scale clustering
- **OOD detection**: Matches but doesn't beat simple baselines like kNN distance
- **Parameter sensitivity**: Performance depends on appropriate parameter selection

See [README_AGE_FINDINGS.md](README_AGE_FINDINGS.md) for detailed analysis and honest evaluation.

## 🙏 Acknowledgments

- Built on scikit-learn ecosystem
- Inspired by OPTICS, DBSCAN, and manifold learning research
- Community feedback and contributions

## 📞 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/autonomous-geometric-engine/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/autonomous-geometric-engine/discussions)

---

**Note**: AGE is positioned as a production-ready integration tool, not as state-of-the-art clustering or OOD detection. Its value lies in combining these capabilities in a deployable, sklearn-compatible package.
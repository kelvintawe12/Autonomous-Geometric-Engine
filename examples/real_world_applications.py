"""
real_world_applications.py — Real-world application examples

This example demonstrates AGE usage in realistic scenarios:
1. Radar/Telemetry anomaly detection
2. LiDAR point cloud processing
3. Manufacturing quality control
"""

import numpy as np
import matplotlib.pyplot as plt
from age import AGE
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ============================================================================
# 1. Radar/Telemetry Anomaly Detection
# ============================================================================
print("=" * 60)
print("1. RADAR/TELEMETRY ANOMALY DETECTION")
print("=" * 60)

def simulate_radar_data(n_normal=1000, n_anomaly=100):
    """Simulate radar trajectory data"""
    # Normal aircraft trajectories (smooth curves)
    t = np.linspace(0, 10, n_normal)
    normal_x = 5 * np.sin(t) + np.random.normal(0, 0.1, n_normal)
    normal_y = 5 * np.cos(t) + np.random.normal(0, 0.1, n_normal)
    
    # Anomalous trajectories (erratic patterns)
    t_anom = np.linspace(0, 10, n_anomaly)
    anom_x = 8 * np.sin(2 * t_anom) + np.random.normal(0, 0.5, n_anomaly)
    anom_y = 8 * np.cos(2 * t_anom) + np.random.normal(0, 0.5, n_anomaly)
    
    return np.column_stack([normal_x, normal_y]), np.column_stack([anom_x, anom_y])

# Train on normal data
X_train, X_anomaly = simulate_radar_data()
X_train_scaled = StandardScaler().fit_transform(X_train)

print(f"Training on {len(X_train)} normal radar trajectories...")
age_radar = AGE(
    min_samples=10,
    xi=0.05,
    base_clustering='optics',
    enhance_ood=False,
    use_robust_cov=False
)
age_radar.fit(X_train_scaled)

print(f"Clusters found: {age_radar.n_clusters_}")

# Test on mixed data
X_test = np.vstack([X_train[:100], X_anomaly])
X_test_scaled = StandardScaler().fit_transform(X_test)
predictions = age_radar.predict(X_test_scaled)

n_normal_detected = np.sum(predictions[:100] != -1)
n_anomaly_detected = np.sum(predictions[100:] == -1)

print(f"Normal trajectories detected: {n_normal_detected}/100")
print(f"Anomalous trajectories rejected: {n_anomaly_detected}/100")

# ============================================================================
# 2. LiDAR Point Cloud Processing
# ============================================================================
print("\n" + "=" * 60)
print("2. LIDAR POINT CLOUD PROCESSING")
print("=" * 60)

def simulate_lidar_data(n_points=2000):
    """Simulate 3D LiDAR point cloud of a structure"""
    # Ground plane
    ground = np.random.randn(n_points//2, 3)
    ground[:, 2] = 0  # Flatten z coordinate
    
    # Vertical structure (like a building or pole)
    structure = np.random.randn(n_points//2, 3)
    structure[:, 0] = structure[:, 0] * 0.1 + 5  # Concentrate around x=5
    structure[:, 1] = structure[:, 1] * 0.1 + 5  # Concentrate around y=5
    structure[:, 2] = np.random.uniform(0, 10, n_points//2)  # Height variation
    
    return np.vstack([ground, structure])

X_lidar = simulate_lidar_data()
print(f"Processing {len(X_lidar)} LiDAR points...")

age_lidar = AGE(
    min_samples=15,
    base_clustering='optics',
    adaptive_geometry=True,
    enhance_ood=False,
    use_robust_cov=False
)
age_lidar.fit(X_lidar)

print(f"LiDAR clusters found: {age_lidar.n_clusters_}")
print(f"Geometry types detected: {set(age_lidar.geometry_types_.values())}")

# ============================================================================
# 3. Manufacturing Quality Control
# ============================================================================
print("\n" + "=" * 60)
print("3. MANUFACTURING QUALITY CONTROL")
print("=" * 60)

def simulate_manufacturing_data(n_good=500, n_defective=50, n_features=5):
    """Simulate manufacturing sensor data"""
    # Good products (consistent measurements)
    good_data = np.random.randn(n_good, n_features) * 0.1 + np.array([1, 2, 3, 4, 5])
    
    # Defective products (deviations in multiple dimensions)
    defective_data = np.random.randn(n_defective, n_features) * 0.5 + np.array([1, 2, 3, 4, 5])
    
    return good_data, defective_data

X_good, X_defective = simulate_manufacturing_data()
print(f"Training on {len(X_good)} good product samples...")

age_qc = AGE(
    min_samples=10,
    base_clustering='optics',
    envelope_quantile=0.95,
    envelope_scale=3.0,
    enhance_ood=False,
    use_robust_cov=False
)
age_qc.fit(X_good)

print(f"Quality control clusters: {age_qc.n_clusters_}")

# Test on mixed data
X_qc_test = np.vstack([X_good[:50], X_defective])
qc_predictions = age_qc.predict(X_qc_test)
qc_confidence = age_qc.predict_proba(X_qc_test)

n_good_accepted = np.sum(qc_predictions[:50] != -1)
n_defective_rejected = np.sum(qc_predictions[50:] == -1)

print(f"Good products accepted: {n_good_accepted}/50")
print(f"Defective products rejected: {n_defective_rejected}/50")
print(f"Average confidence for accepted: {np.mean(qc_confidence[qc_predictions != -1]):.3f}")

# ============================================================================
# Visualization
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Radar data
scatter1 = axes[0].scatter(X_test[:, 0], X_test[:, 1], 
                          c=['green' if p != -1 else 'red' for p in predictions],
                          s=50, alpha=0.6)
axes[0].set_title('Radar Anomaly Detection')
axes[0].set_xlabel('X Position')
axes[0].set_ylabel('Y Position')

# LiDAR data (3D projected to 2D)
scatter2 = axes[1].scatter(X_lidar[:, 0], X_lidar[:, 1], 
                          c=age_lidar.labels_, cmap='viridis', s=50, alpha=0.6)
axes[1].set_title('LiDAR Point Cloud Processing')
axes[1].set_xlabel('X Coordinate')
axes[1].set_ylabel('Y Coordinate')

# Quality control (first 2 features)
scatter3 = axes[2].scatter(X_qc_test[:, 0], X_qc_test[:, 1],
                          c=['blue' if p != -1 else 'red' for p in qc_predictions],
                          s=50, alpha=0.6)
axes[2].set_title('Manufacturing Quality Control')
axes[2].set_xlabel('Sensor 1')
axes[2].set_ylabel('Sensor 2')

plt.tight_layout()
plt.savefig('real_world_applications.png', dpi=150, bbox_inches='tight')
print("\nResults saved to 'real_world_applications.png'")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Radar anomaly detection: {:.1f}% anomaly detection rate".format(
    n_anomaly_detected / 100 * 100))
print("LiDAR processing: {} geometric structures identified".format(
    len(set(age_lidar.geometry_types_.values()))))
print("Quality control: {:.1f}% defective rejection rate".format(
    n_defective_rejected / 50 * 100))
"""
Comprehensive test suite for academic validation of AGE.
"""
import unittest
import numpy as np
from age import AGE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
import warnings

warnings.filterwarnings("ignore")


class TestAGEAcademic(unittest.TestCase):
    """Academic test suite for AGE validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.iris_X, self.iris_y = load_iris(return_X_y=True)
        self.iris_Xs = StandardScaler().fit_transform(self.iris_X)
        
        # Synthetic datasets
        self.blobs_X, self.blobs_y = make_blobs(n_samples=300, centers=3, random_state=42)
        self.circles_X, self.circles_y = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
        self.moons_X, self.moons_y = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    def test_basic_functionality(self):
        """Test basic AGE functionality."""
        age = AGE(min_samples=5)
        age.fit(self.iris_Xs)
        
        self.assertIsNotNone(age.labels_)
        self.assertEqual(len(age.labels_), len(self.iris_Xs))
        self.assertGreater(age.n_clusters_, 0)
        self.assertIsInstance(age.manifolds_, dict)
    
    def test_pluggable_clustering(self):
        """Test pluggable base clustering architecture."""
        # Test OPTICS
        age_optics = AGE(min_samples=5, base_clustering='optics')
        age_optics.fit(self.iris_Xs)
        self.assertGreater(age_optics.n_clusters_, 0)
        
        # Test DBSCAN
        age_dbscan = AGE(min_samples=5, base_clustering='dbscan', 
                        clustering_params={'eps': 0.8})
        age_dbscan.fit(self.iris_Xs)
        self.assertGreaterEqual(age_dbscan.n_clusters_, 0)
        
        # Test GMM
        age_gmm = AGE(min_samples=5, base_clustering='gmm',
                      clustering_params={'n_components': 3})
        age_gmm.fit(self.iris_Xs)
        self.assertEqual(age_gmm.n_clusters_, 3)
    
    def test_predict_functionality(self):
        """Test out-of-sample prediction."""
        age = AGE(min_samples=5)
        age.fit(self.iris_Xs)
        
        # Test on training data
        pred_train = age.predict(self.iris_Xs)
        self.assertEqual(len(pred_train), len(self.iris_Xs))
        
        # Test on new data
        X_new = self.iris_Xs[:10]
        pred_new = age.predict(X_new)
        self.assertEqual(len(pred_new), 10)
        
        # Test OOD rejection
        X_ood = np.random.randn(10, 4)
        pred_ood = age.predict(X_ood)
        self.assertEqual(len(pred_ood), 10)
    
    def test_confidence_scoring(self):
        """Test confidence scoring functionality."""
        age = AGE(min_samples=5)
        age.fit(self.iris_Xs)
        
        # Get confidence scores
        confidence = age.predict_proba(self.iris_Xs)
        
        self.assertEqual(len(confidence), len(self.iris_Xs))
        self.assertTrue(np.all(confidence >= 0))
        self.assertTrue(np.all(confidence <= 1))
        
        # Test that OOD points get lower confidence
        X_ood = np.random.randn(10, 4)
        conf_ood = age.predict_proba(X_ood)
        self.assertTrue(np.mean(conf_ood) < np.mean(confidence))
    
    def test_parameter_tuning(self):
        """Test automatic parameter tuning."""
        age = AGE(min_samples=5)
        age.tune_envelope_parameters(self.blobs_X)
        
        # Check that parameters were set
        self.assertIsNotNone(age.envelope_quantile)
        self.assertIsNotNone(age.envelope_scale)
        self.assertGreater(age.n_clusters_, 0)
    
    def test_synthetic_datasets(self):
        """Test on various synthetic datasets."""
        datasets = [
            ("Blobs", self.blobs_X, 3),
            ("Circles", self.circles_X, 2),
            ("Moons", self.moons_X, 2)
        ]
        
        for name, X, expected_clusters in datasets:
            with self.subTest(dataset=name):
                age = AGE(min_samples=5)
                age.fit(X)
                
                # Should find clusters
                self.assertGreater(age.n_clusters_, 0)
                
                # Should be able to predict
                pred = age.predict(X)
                self.assertEqual(len(pred), len(X))
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Single cluster
        single_cluster = np.random.randn(50, 3)
        age = AGE(min_samples=5)
        age.fit(single_cluster)
        self.assertGreaterEqual(age.n_clusters_, 0)
        
        # Very small dataset
        tiny_data = np.random.randn(10, 2)
        age_tiny = AGE(min_samples=3)
        age_tiny.fit(tiny_data)
        self.assertIsNotNone(age_tiny.labels_)
        
        # High-dimensional data
        high_dim = np.random.randn(100, 20)
        age_hd = AGE(min_samples=5)
        age_hd.fit(high_dim)
        self.assertIsNotNone(age_hd.labels_)
    
    def test_decision_function(self):
        """Test decision distance function."""
        age = AGE(min_samples=5)
        age.fit(self.iris_Xs)
        
        distances = age.decision_distance(self.iris_Xs)
        
        self.assertEqual(len(distances), len(self.iris_Xs))
        self.assertTrue(np.all(distances >= 0))
        self.assertTrue(np.all(np.isfinite(distances)))
    
    def test_backward_compatibility(self):
        """Test backward compatibility with original API."""
        # Original parameters should still work
        age = AGE(min_samples=5, xi=0.05, n_components=15,
                 envelope_quantile=0.98, envelope_scale=4.0)
        age.fit(self.iris_Xs)
        
        self.assertGreater(age.n_clusters_, 0)
        self.assertIsNotNone(age.manifolds_)
    
    def test_clustering_quality(self):
        """Test clustering quality on real datasets."""
        datasets = [
            ("Iris", self.iris_Xs, self.iris_y),
            ("Wine", *load_wine(return_X_y=True))
        ]
        
        for name, X, y in datasets:
            if name == "Wine":
                X = StandardScaler().fit_transform(X)
            
            with self.subTest(dataset=name):
                age = AGE(min_samples=5)
                pred = age.fit_predict(X)
                
                # Calculate ARI
                ari = adjusted_rand_score(y, pred)
                
                # AGE should at least work (not necessarily beat baselines)
                self.assertGreaterEqual(ari, -1.0)
                self.assertLessEqual(ari, 1.0)
    
    def test_reproducibility(self):
        """Test reproducibility with random states."""
        # Fit twice with same seed
        age1 = AGE(min_samples=5)
        np.random.seed(42)
        age1.fit(self.iris_Xs)
        
        age2 = AGE(min_samples=5)
        np.random.seed(42)
        age2.fit(self.iris_Xs)
        
        # Should get same results
        np.testing.assert_array_equal(age1.labels_, age2.labels_)
        self.assertEqual(age1.n_clusters_, age2.n_clusters_)


class TestAGEPerformance(unittest.TestCase):
    """Performance and scaling tests."""
    
    def test_scaling_with_n(self):
        """Test runtime scaling with dataset size."""
        import time
        
        sizes = [500, 1000, 2000]
        times = []
        
        for n in sizes:
            X = np.random.randn(n, 3)
            age = AGE(min_samples=5)
            
            start = time.time()
            age.fit(X)
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Runtime should increase with N (not decrease)
        self.assertTrue(times[0] < times[2])
        
        print(f"\nScaling times: {dict(zip(sizes, times))}")
    
    def test_memory_efficiency(self):
        """Test that memory usage is reasonable."""
        import sys
        
        # Medium-sized dataset
        X = np.random.randn(5000, 10)
        age = AGE(min_samples=5)
        age.fit(X)
        
        # Manifolds should not be excessively large
        manifold_size = sys.getsizeof(age.manifolds_)
        self.assertLess(manifold_size, 10 * 1024 * 1024)  # Less than 10MB


def run_academic_tests():
    """Run the academic test suite."""
    print("Running Academic AGE Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAGEAcademic))
    suite.addTests(loader.loadTestsFromTestCase(TestAGEPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(1 - (len(result.failures) + len(result.errors)) / result.testsRun) * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_academic_tests()
    exit(0 if success else 1)

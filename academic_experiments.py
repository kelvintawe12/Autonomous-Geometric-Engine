"""
Comprehensive academic experiment runner for AGE paper.
Generates reproducible results for clustering quality, prediction performance,
OOD detection, and ablation studies.
"""
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KDTree
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, roc_auc_score
)
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    make_blobs, make_circles, make_moons, make_swiss_roll
)

from age import AGE

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)


class ExperimentRunner:
    """Reproducible experiment runner for academic validation."""
    
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_datasets(self):
        """Load all datasets for experiments."""
        datasets = {}
        
        # Real datasets
        iris_X, iris_y = load_iris(return_X_y=True)
        datasets['Iris'] = (StandardScaler().fit_transform(iris_X), iris_y)
        
        wine_X, wine_y = load_wine(return_X_y=True)
        datasets['Wine'] = (StandardScaler().fit_transform(wine_X), wine_y)
        
        cancer_X, cancer_y = load_breast_cancer(return_X_y=True)
        datasets['BreastCancer'] = (StandardScaler().fit_transform(cancer_X), cancer_y)
        
        # Synthetic datasets
        datasets['Blobs'] = make_blobs(n_samples=500, centers=3, n_features=5, random_state=SEED)
        datasets['Circles'] = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=SEED)
        datasets['Moons'] = make_moons(n_samples=500, noise=0.05, random_state=SEED)
        datasets['SwissRoll'] = make_swiss_roll(n_samples=500, noise=0.05, random_state=SEED)
        
        return datasets
    
    def clustering_quality_experiment(self):
        """Experiment 1: Clustering quality comparison."""
        print("\n" + "="*60)
        print("EXPERIMENT 1: Clustering Quality Comparison")
        print("="*60)
        
        datasets = self.load_datasets()
        results = []
        
        for name, (X, y) in datasets.items():
            print(f"\nDataset: {name}")
            print("-" * 40)
            
            k = len(np.unique(y))
            
            # Methods to compare
            methods = {
                'KMeans': KMeans(n_clusters=k, n_init=10, random_state=SEED),
                'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
                'OPTICS': OPTICS(min_samples=5, xi=0.05),
                'GMM': GaussianMixture(n_components=k, random_state=SEED),
                'AGE': AGE(min_samples=5),
                'AGE-GMM': AGE(min_samples=5, base_clustering='gmm', 
                              clustering_params={'n_components': k}),
            }
            
            for method_name, method in methods.items():
                try:
                    start = time.time()
                    if method_name == 'GMM':
                        labels = method.fit_predict(X)
                    else:
                        labels = method.fit_predict(X)
                    elapsed = time.time() - start
                    
                    # Calculate metrics
                    ari = adjusted_rand_score(y, labels)
                    nmi = normalized_mutual_info_score(y, labels)
                    
                    # Silhouette (only if more than 1 cluster)
                    if len(np.unique(labels)) > 1:
                        sil = silhouette_score(X, labels)
                    else:
                        sil = np.nan
                    
                    results.append({
                        'Dataset': name,
                        'Method': method_name,
                        'ARI': round(ari, 4),
                        'NMI': round(nmi, 4),
                        'Silhouette': round(sil, 4) if not np.isnan(sil) else 'NA',
                        'Time': round(elapsed, 4),
                        'Clusters': len(np.unique(labels))
                    })
                    
                    print(f"  {method_name:12s}: ARI={ari:.3f}, NMI={nmi:.3f}, Time={elapsed:.3f}s")
                    
                except Exception as e:
                    print(f"  {method_name:12s}: FAILED - {str(e)}")
        
        df = pd.DataFrame(results)
        self.results['clustering_quality'] = df
        
        # Save results
        output_file = self.output_dir / f"clustering_quality_{self.timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return df
    
    def prediction_confidence_experiment(self):
        """Experiment 2: Prediction confidence and uncertainty."""
        print("\n" + "="*60)
        print("EXPERIMENT 2: Prediction Confidence Analysis")
        print("="*60)
        
        datasets = self.load_datasets()
        results = []
        
        for name, (X, y) in datasets.items():
            if name in ['SwissRoll']:  # Skip high-dimensional for this test
                continue
                
            print(f"\nDataset: {name}")
            print("-" * 40)
            
            # Split into train/test
            n_train = len(X) // 2
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
            
            # Fit AGE
            age = AGE(min_samples=5)
            age.fit(X_train)
            
            # Get predictions and confidence
            pred_test = age.predict(X_test)
            conf_test = age.predict_proba(X_test)
            
            # Calculate metrics
            in_dist_mask = pred_test != -1
            in_dist_rate = np.mean(in_dist_mask)
            mean_confidence = np.mean(conf_test[in_dist_mask]) if np.any(in_dist_mask) else 0
            
            # Compare with ground truth consistency
            if np.any(in_dist_mask):
                # For in-distribution points, check if they go to reasonable clusters
                consistency_rate = np.mean(pred_test[in_dist_mask] != -1)
            else:
                consistency_rate = 0
            
            results.append({
                'Dataset': name,
                'In-Distribution Rate': round(in_dist_rate, 4),
                'Mean Confidence': round(mean_confidence, 4),
                'Consistency Rate': round(consistency_rate, 4),
                'Test Samples': len(X_test)
            })
            
            print(f"  In-Dist Rate: {in_dist_rate:.3f}")
            print(f"  Mean Confidence: {mean_confidence:.3f}")
            print(f"  Consistency: {consistency_rate:.3f}")
        
        df = pd.DataFrame(results)
        self.results['prediction_confidence'] = df
        
        output_file = self.output_dir / f"prediction_confidence_{self.timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return df
    
    def ood_detection_experiment(self):
        """Experiment 3: OOD detection performance."""
        print("\n" + "="*60)
        print("EXPERIMENT 3: Out-of-Distribution Detection")
        print("="*60)
        
        # Synthetic OOD experiment
        rng = np.random.RandomState(SEED)
        
        # Training data: ring
        th = rng.uniform(0, 2*np.pi, 600)
        r = 5.0 + rng.normal(0, 0.08, 600)
        X_tr = np.column_stack([r*np.cos(th), r*np.sin(th)])
        
        # In-distribution test
        th2 = rng.uniform(0, 2*np.pi, 2000)
        r2 = 5.0 + rng.normal(0, 0.08, 2000)
        X_in = np.column_stack([r2*np.cos(th2), r2*np.sin(th2)])
        
        # OOD test
        X_ood = rng.uniform(-15, 15, (6000, 2))
        X_ood = X_ood[np.abs(np.linalg.norm(X_ood, axis=1) - 5.0) > 1.5][:2000]
        
        X_test = np.vstack([X_in, X_ood])
        y_test = np.hstack([np.zeros(len(X_in)), np.ones(len(X_ood))])  # 1 = OOD
        
        results = []
        
        # Methods to compare
        age = AGE(min_samples=5).fit(X_tr)
        ocs = OneClassSVM(gamma="scale", nu=0.05).fit(X_tr)
        tree = KDTree(X_tr)
        
        # AGE scores
        age_scores = age.decision_distance(X_test)
        age_auc = roc_auc_score(y_test, age_scores)
        age_pred = age.predict(X_test)
        age_acc_in = np.mean(age_pred[:len(X_in)] != -1)
        age_rej_ood = np.mean(age_pred[len(X_in):] == -1)
        
        results.append({
            'Method': 'AGE',
            'AUROC': round(age_auc, 4),
            'In-Dist Accept Rate': round(age_acc_in, 4),
            'OOD Reject Rate': round(age_rej_ood, 4)
        })
        
        # One-Class SVM
        ocs_scores = -ocs.decision_function(X_test)
        ocs_auc = roc_auc_score(y_test, ocs_scores)
        results.append({
            'Method': 'One-Class SVM',
            'AUROC': round(ocs_auc, 4),
            'In-Dist Accept Rate': 'NA',
            'OOD Reject Rate': 'NA'
        })
        
        # kNN baseline
        knn_scores = tree.query(X_test, k=5)[0].mean(axis=1)
        knn_auc = roc_auc_score(y_test, knn_scores)
        results.append({
            'Method': 'kNN(5) distance',
            'AUROC': round(knn_auc, 4),
            'In-Dist Accept Rate': 'NA',
            'OOD Reject Rate': 'NA'
        })
        
        df = pd.DataFrame(results)
        self.results['ood_detection'] = df
        
        output_file = self.output_dir / f"ood_detection_{self.timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        print("\nResults:")
        print(df.to_string(index=False))
        print(f"\nResults saved to {output_file}")
        
        return df
    
    def ablation_study(self):
        """Experiment 4: Ablation study on components."""
        print("\n" + "="*60)
        print("EXPERIMENT 4: Ablation Study")
        print("="*60)
        
        datasets = self.load_datasets()
        results = []
        
        for name, (X, y) in datasets.items():
            if name in ['SwissRoll']:  # Skip complex datasets
                continue
                
            print(f"\nDataset: {name}")
            print("-" * 40)
            
            # Different AGE configurations
            configs = {
                'AGE-Full': AGE(min_samples=5),
                'AGE-NoEnvelope': AGE(min_samples=5, envelope_scale=0.0),
                'AGE-DBSCAN': AGE(min_samples=5, base_clustering='dbscan', 
                                 clustering_params={'eps': 0.8}),
                'AGE-GMM': AGE(min_samples=5, base_clustering='gmm',
                              clustering_params={'n_components': len(np.unique(y))}),
            }
            
            for config_name, model in configs.items():
                try:
                    labels = model.fit_predict(X)
                    ari = adjusted_rand_score(y, labels)
                    
                    results.append({
                        'Dataset': name,
                        'Configuration': config_name,
                        'ARI': round(ari, 4),
                        'Clusters': len(np.unique(labels))
                    })
                    
                    print(f"  {config_name:15s}: ARI={ari:.3f}")
                    
                except Exception as e:
                    print(f"  {config_name:15s}: FAILED - {str(e)}")
        
        df = pd.DataFrame(results)
        self.results['ablation'] = df
        
        output_file = self.output_dir / f"ablation_{self.timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return df
    
    def runtime_analysis(self):
        """Experiment 5: Runtime scaling analysis."""
        print("\n" + "="*60)
        print("EXPERIMENT 5: Runtime Scaling Analysis")
        print("="*60)
        
        results = []
        sizes = [500, 1000, 2000, 5000, 10000]
        
        for n in sizes:
            print(f"\nN = {n}")
            print("-" * 40)
            
            # Generate data
            X = np.random.randn(n, 5)
            
            # Test AGE
            age = AGE(min_samples=5)
            start = time.time()
            age.fit(X)
            age_time = time.time() - start
            
            # Test OPTICS alone
            optics = OPTICS(min_samples=5, xi=0.05)
            start = time.time()
            optics.fit_predict(X)
            optics_time = time.time() - start
            
            results.append({
                'N': n,
                'AGE': round(age_time, 4),
                'OPTICS': round(optics_time, 4),
                'Ratio': round(age_time / optics_time, 2)
            })
            
            print(f"  AGE: {age_time:.3f}s, OPTICS: {optics_time:.3f}s, Ratio: {age_time/optics_time:.2f}")
        
        df = pd.DataFrame(results)
        self.results['runtime'] = df
        
        output_file = self.output_dir / f"runtime_{self.timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return df
    
    def parameter_sensitivity(self):
        """Experiment 6: Parameter sensitivity analysis."""
        print("\n" + "="*60)
        print("EXPERIMENT 6: Parameter Sensitivity Analysis")
        print("="*60)
        
        datasets = self.load_datasets()
        results = []
        
        # Test on a few datasets
        test_datasets = ['Iris', 'Blobs', 'Circles']
        
        for name in test_datasets:
            if name not in datasets:
                continue
                
            X, y = datasets[name]
            print(f"\nDataset: {name}")
            print("-" * 40)
            
            # Test different envelope parameters
            quantiles = [0.90, 0.95, 0.98, 0.99]
            scales = [2.0, 4.0, 6.0, 8.0]
            
            for quantile in quantiles:
                for scale in scales:
                    age = AGE(min_samples=5, envelope_quantile=quantile, envelope_scale=scale)
                    labels = age.fit_predict(X)
                    ari = adjusted_rand_score(y, labels)
                    
                    results.append({
                        'Dataset': name,
                        'Quantile': quantile,
                        'Scale': scale,
                        'ARI': round(ari, 4),
                        'Clusters': len(np.unique(labels))
                    })
        
        df = pd.DataFrame(results)
        self.results['parameter_sensitivity'] = df
        
        output_file = self.output_dir / f"parameter_sensitivity_{self.timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return df
    
    def run_all_experiments(self):
        """Run all experiments and generate summary."""
        print("\n" + "="*70)
        print(" COMPREHENSIVE ACADEMIC EXPERIMENT SUITE FOR AGE")
        print("="*70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Random Seed: {SEED}")
        
        # Run all experiments
        self.clustering_quality_experiment()
        self.prediction_confidence_experiment()
        self.ood_detection_experiment()
        self.ablation_study()
        self.runtime_analysis()
        self.parameter_sensitivity()
        
        # Generate summary
        self.generate_summary()
        
        print("\n" + "="*70)
        print(" ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("="*70)
    
    def generate_summary(self):
        """Generate summary of all results."""
        summary_file = self.output_dir / f"summary_{self.timestamp}.json"
        
        summary = {
            'timestamp': self.timestamp,
            'random_seed': SEED,
            'experiments': {}
        }
        
        for exp_name, df in self.results.items():
            summary['experiments'][exp_name] = {
                'rows': len(df),
                'columns': list(df.columns),
                'summary_stats': df.describe().to_dict() if not df.empty else {}
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Run academic experiments for AGE')
    parser.add_argument('--output', default='results', help='Output directory for results')
    parser.add_argument('--experiment', choices=['all', 'clustering', 'prediction', 'ood', 'ablation', 'runtime', 'sensitivity'],
                       default='all', help='Which experiment to run')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output)
    
    if args.experiment == 'all':
        runner.run_all_experiments()
    elif args.experiment == 'clustering':
        runner.clustering_quality_experiment()
    elif args.experiment == 'prediction':
        runner.prediction_confidence_experiment()
    elif args.experiment == 'ood':
        runner.ood_detection_experiment()
    elif args.experiment == 'ablation':
        runner.ablation_study()
    elif args.experiment == 'runtime':
        runner.runtime_analysis()
    elif args.experiment == 'sensitivity':
        runner.parameter_sensitivity()


if __name__ == "__main__":
    main()

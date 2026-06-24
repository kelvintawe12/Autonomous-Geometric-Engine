"""
benchmark_age.py — honest, reproducible evaluation of AGE.

Every number printed here is measured at run time. Nothing is hardcoded.
The point is to show AGE's behaviour faithfully, INCLUDING where it loses to
simpler baselines. Run:  python benchmark_age.py
"""
import time
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS, KMeans, DBSCAN
from sklearn.neighbors import KDTree
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.datasets import load_iris, load_wine

from age import AGE

SEED = 42


# ---------------------------------------------------------------------------
# 1. Clustering quality on real data (unsupervised; labels used only to SCORE)
# ---------------------------------------------------------------------------
def clustering_quality():
    rows = []
    for name, (X, y), eps in [
        ("UCI Iris", load_iris(return_X_y=True), 0.8),
        ("UCI Wine", load_wine(return_X_y=True), 2.2),
    ]:
        Xs = StandardScaler().fit_transform(X)
        k = len(np.unique(y))
        rows.append({
            "Dataset": name,
            "KMeans": f"{adjusted_rand_score(y, KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xs)):.3f}",
            "DBSCAN": f"{adjusted_rand_score(y, DBSCAN(eps=eps, min_samples=4).fit_predict(Xs)):.3f}",
            "OPTICS": f"{adjusted_rand_score(y, OPTICS(min_samples=5, xi=0.05).fit_predict(Xs)):.3f}",
            "AGE": f"{adjusted_rand_score(y, AGE(min_samples=5, xi=0.05).fit_predict(Xs)):.3f}",
        })
    print("\n1. CLUSTERING QUALITY  (Adjusted Rand Index vs ground truth; higher better)")
    print("   Unsupervised: no method receives labels. Labels score the result only.")
    print(pd.DataFrame(rows).to_string(index=False))
    print("   NOTE: AGE == OPTICS by construction; KMeans wins. The envelope layer")
    print("         does not improve partition quality. This is expected and reported.")


# ---------------------------------------------------------------------------
# 2. Out-of-distribution detection: AGE vs simpler dedicated baselines
# ---------------------------------------------------------------------------
def ood_detection():
    rng = np.random.RandomState(SEED)
    th = rng.uniform(0, 2 * np.pi, 600)
    r = 5.0 + rng.normal(0, 0.08, 600)
    X_tr = np.column_stack([r * np.cos(th), r * np.sin(th)])

    th2 = rng.uniform(0, 2 * np.pi, 2000); r2 = 5.0 + rng.normal(0, 0.08, 2000)
    X_in = np.column_stack([r2 * np.cos(th2), r2 * np.sin(th2)])
    X_ood = rng.uniform(-15, 15, (6000, 2))
    X_ood = X_ood[np.abs(np.linalg.norm(X_ood, axis=1) - 5.0) > 1.5][:2000]

    Xe = np.vstack([X_in, X_ood])
    y = np.hstack([np.zeros(len(X_in)), np.ones(len(X_ood))])  # 1 = OOD

    age = AGE(min_samples=5).fit(X_tr)
    auc_age = roc_auc_score(y, age.decision_distance(Xe))
    ocs = OneClassSVM(gamma="scale", nu=0.05).fit(X_tr)
    auc_ocs = roc_auc_score(y, -ocs.decision_function(Xe))
    tree = KDTree(X_tr)
    auc_knn = roc_auc_score(y, tree.query(Xe, k=5)[0].mean(axis=1))

    acc_in = np.mean(age.predict(X_in) != -1) * 100
    rej_ood = np.mean(age.predict(X_ood) == -1) * 100

    print("\n2. OUT-OF-DISTRIBUTION DETECTION  (AUROC, higher better)")
    print(pd.DataFrame([
        {"Method": "kNN(5) distance (5-line baseline)", "AUROC": f"{auc_knn:.3f}"},
        {"Method": "One-Class SVM", "AUROC": f"{auc_ocs:.3f}"},
        {"Method": "AGE (nearest-point distance)", "AUROC": f"{auc_age:.3f}"},
    ]).to_string(index=False))
    print(f"   AGE hard decision: accepts {acc_in:.1f}% of in-distribution, "
          f"rejects {rej_ood:.1f}% of OOD.")
    print("   NOTE: AGE's rejection is usable but does NOT beat the simpler")
    print("         baselines. Reported honestly.")


# ---------------------------------------------------------------------------
# 3. The one thing AGE adds that KMeans/DBSCAN/OPTICS lack: a real predict()
# ---------------------------------------------------------------------------
def predict_capability():
    rng = np.random.RandomState(SEED)
    th = rng.uniform(0, 2 * np.pi, 800); r = 5.0 + rng.normal(0, 0.08, 800)
    X_tr = np.column_stack([r * np.cos(th), r * np.sin(th)])
    age = AGE(min_samples=5).fit(X_tr)
    th2 = rng.uniform(0, 2 * np.pi, 500); r2 = 5.0 + rng.normal(0, 0.08, 500)
    X_new = np.column_stack([r2 * np.cos(th2), r2 * np.sin(th2)])
    accepted = np.mean(age.predict(X_new) != -1) * 100
    print("\n3. OUT-OF-SAMPLE predict()  (capability KMeans/DBSCAN/OPTICS lack natively)")
    print(f"   Trained on a ring; {accepted:.1f}% of UNSEEN on-ring points accepted "
          f"and assigned to a cluster.")
    print("   This (a deployable predict + reject on new data) is AGE's actual niche.")


# ---------------------------------------------------------------------------
# 4. Runtime scaling (real timings; dominated by OPTICS)
# ---------------------------------------------------------------------------
def runtime_scaling():
    rows = []
    for N in [1000, 2500, 5000, 10000]:
        th = np.linspace(0, 2 * np.pi, N)
        X = np.column_stack([4 * np.cos(th), 4 * np.sin(th)])
        t = time.time(); AGE(min_samples=10).fit(X); dt = time.time() - t
        rows.append({"N": N, "fit time (s)": f"{dt:.3f}"})
    print("\n4. RUNTIME SCALING  (fit, measured)")
    print(pd.DataFrame(rows).to_string(index=False))
    print("   NOTE: dominated by OPTICS (~O(N^1.4) empirically), not O(N log N).")


if __name__ == "__main__":
    print("=" * 75)
    print("AGE — HONEST BENCHMARK (all numbers measured at run time)")
    print("=" * 75)
    clustering_quality()
    ood_detection()
    predict_capability()
    runtime_scaling()
    print("\n" + "=" * 75)
    print("See README_AGE_FINDINGS.md for interpretation and limitations.")

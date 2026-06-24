# AGE — Honest Findings & Review

This document is the candid debrief of the Autonomous Geometric Engine (AGE).
It records what the model **actually does**, measured with reproducible code
(`benchmark_age.py`, engine in `age.py`), and where the original notebook's
claims did not hold up. Every number below is produced at run time.

> TL;DR: AGE is a clean, scikit-learn-compatible clustering estimator that adds
> a real `predict()` and out-of-distribution (OOD) rejection on top of OPTICS.
> That integration is its genuine, useful niche. It does **not** beat standard
> baselines at clustering quality or at OOD detection, and it is **not**
> real-time at scale. Claims to the contrary in the original notebook were
> artifacts of methodological errors (listed below), not real results.

## What AGE is

`AGE(BaseEstimator, ClusterMixin)`:
- `fit(X)` — density-clusters with OPTICS, then wraps each discovered cluster
  in a manifold envelope (KD-tree + adaptive-gamma RBF Nystroem map + local
  acceptance radius).
- `predict(X_new)` — assigns unseen points to the nearest manifold, or `-1`
  if they fall outside every envelope.
- `decision_distance(X_new)` — continuous OOD score for ROC analysis.

The value proposition is **a deployable `predict()` + principled noise
rejection** — things KMeans/DBSCAN/OPTICS do not offer natively in one object.

## Measured results (reproduce with `python benchmark_age.py`)

### 1. Clustering quality — ARI vs ground truth (unsupervised)
| Dataset | KMeans | DBSCAN | OPTICS | AGE |
|---|---|---|---|---|
| Iris | **0.620** | 0.552 | 0.051 | 0.051 |
| Wine | **0.897** | 0.378 | 0.036 | 0.036 |

AGE equals OPTICS *by construction* (it reuses OPTICS labels). KMeans wins.
The envelope layer does not improve partition quality. **Reported, not hidden.**

### 2. OOD detection — AUROC (higher better)
| Method | AUROC |
|---|---|
| kNN(5) distance (≈5 lines of code) | 1.000 |
| One-Class SVM | ~0.94 |
| AGE (nearest-point distance) | 1.000 |

AGE matches the trivial kNN baseline — because its rejection score *is*
essentially a nearest-training-point distance. Useful, not novel.

### 3. Out-of-sample `predict()`
Trained on a ring; ~96% of unseen on-ring points are accepted and assigned,
~100% of far noise is rejected. **This is the actual contribution.**

### 4. Runtime (fit, measured)
~0.58s @ 1k → ~8.6s @ 10k points, scaling ~O(N^1.4), dominated by OPTICS.
Not the "millions of vectors/sec" claimed; that figure timed rejection speed
with no correctness check.

## Limitations (state these plainly anywhere AGE is used)
1. **Clustering quality = OPTICS.** No improvement over the underlying scanner;
   KMeans beats it on convex/real data.
2. **OOD rejection is matched by simpler baselines** (kNN distance, One-Class
   SVM). It is a textbook capability, cleanly integrated — not a new detector.
3. **Inherits OPTICS fragility.** OPTICS can over-segment a single clean ring
   into dozens of micro-clusters; `predict()` then returns fragment IDs.
4. **Runtime is OPTICS-bound**, not sub-linear; not real-time at large N.
5. **Shape classifier (notebook Proof 12) was tuned to 3 hand-built shapes** and
   collapses under realistic noise (σ≥1). Treat as illustrative, not validated.

## Errors found in the original notebook (do not reintroduce)
- **Label leakage:** the flagship Iris/Wine/Proof-11 "wins" came from
  `fit_class(class_id, pts)` being fed ground-truth labels while baselines got
  none — a supervised-vs-unsupervised mismatch. Removed.
- **Fabricated statistics:** the Wilcoxon test (notebook Cell 21) ran on
  `np.random.normal(loc=0.83…)` vs `np.random.normal(loc=0.12…)` — random draws
  from hand-picked means, presented as a significance result. Delete.
- **Hardcoded "measured" values:** ablation runtimes given as string literals;
  LaTeX table contained numbers produced by no cell (and didn't compile —
  `lccccc` vs 7 columns).
- **Normalization circularity:** "generalization across radii/shifts" tested
  data that `normalize_shape()` had already mapped to the same unit circle, so
  100% was guaranteed and meaningless.
- **Correctness bugs:** complex eigenvalues from `np.linalg.eig` on covariance
  (use `eigvalsh`); N×N tensor allocation in the streaming predict; in-sample
  evaluation reported as accuracy; `stats.t.interval(0, …)`; `import hdbscan`
  with no such package installed; FastAPI bound to `0.0.0.0`.

## Honest positioning for citation
AGE is worth referencing as a **correct, transparent reference implementation**
of manifold-envelope clustering with out-of-sample prediction and OOD
rejection — and as a **case study in evaluating such a model without fooling
yourself**. It is not state-of-the-art and should not be marketed as such
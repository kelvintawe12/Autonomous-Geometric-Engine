"""
integration_guide.py — Integration guide for using AGE in existing projects

This example demonstrates how to integrate AGE into various machine learning
pipelines and frameworks.
"""

import numpy as np
from age import AGE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle

np.random.seed(42)

# ============================================================================
# 1. Basic sklearn Pipeline Integration
# ============================================================================
print("=" * 60)
print("1. SKLEARN PIPELINE INTEGRATION")
print("=" * 60)

# Create sample data
X = np.random.randn(1000, 5)
y = np.random.randint(0, 3, 1000)

# Create a pipeline with AGE
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clusterer', AGE(min_samples=5, base_clustering='optics', enhance_ood=False, use_robust_cov=False))
])

# Fit the pipeline
pipeline.fit(X)
print(f"Pipeline fitted successfully")
print(f"Clusters found: {pipeline.named_steps['clusterer'].n_clusters_}")

# ============================================================================
# 2. Hybrid Pipeline: Clustering + Classification
# ============================================================================
print("\n" + "=" * 60)
print("2. HYBRID PIPELINE: CLUSTERING + CLASSIFICATION")
print("=" * 60)

# Use AGE for feature engineering
class ClusterFeatureExtractor:
    """Custom transformer that uses AGE cluster assignments as features"""
    
    def __init__(self, age_model):
        self.age_model = age_model
        
    def fit(self, X, y=None):
        self.age_model.fit(X)
        return self
        
    def transform(self, X):
        cluster_labels = self.age_model.predict(X)
        confidence = self.age_model.predict_proba(X)
        ood_scores = self.age_model.decision_distance(X)
        
        # Combine features
        features = np.column_stack([
            cluster_labels.reshape(-1, 1),
            confidence.reshape(-1, 1),
            ood_scores.reshape(-1, 1)
        ])
        
        # Handle -1 labels (OOD) with special encoding
        features[features[:, 0] == -1, 0] = -999
        
        return features

# Create hybrid pipeline
age_model = AGE(min_samples=5, base_clustering='optics', enhance_ood=False, use_robust_cov=False)
cluster_extractor = ClusterFeatureExtractor(age_model)

hybrid_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('cluster_features', cluster_extractor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit hybrid pipeline
hybrid_pipeline.fit(X_train, y_train)
accuracy = hybrid_pipeline.score(X_test, y_test)
print(f"Hybrid pipeline accuracy: {accuracy:.3f}")

# ============================================================================
# 3. Model Persistence
# ============================================================================
print("\n" + "=" * 60)
print("3. MODEL PERSISTENCE")
print("=" * 60)

# Save AGE model
age_model = AGE(min_samples=5, base_clustering='optics', enhance_ood=False, use_robust_cov=False)
age_model.fit(X)

# Save using joblib
joblib.dump(age_model, 'age_model.joblib')
print("Model saved to 'age_model.joblib'")

# Load model
loaded_model = joblib.load('age_model.joblib')
print("Model loaded successfully")

# Verify loaded model works
predictions = loaded_model.predict(X[:10])
print(f"Predictions from loaded model: {predictions}")

# Alternative: save using pickle
with open('age_model.pkl', 'wb') as f:
    pickle.dump(age_model, f)
print("Model also saved to 'age_model.pkl'")

# ============================================================================
# 4. Integration with Web Services (FastAPI example)
# ============================================================================
print("\n" + "=" * 60)
print("4. WEB SERVICE INTEGRATION (FastAPI)")
print("=" * 60)

# Example FastAPI endpoint code (commented out as it requires FastAPI installation)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Load model at startup
age_model = joblib.load('age_model.joblib')

class PredictionRequest(BaseModel):
    data: list[list[float]]

class PredictionResponse(BaseModel):
    predictions: list[int]
    confidence: list[float]
    ood_scores: list[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        X = np.array(request.data)
        predictions = age_model.predict(X).tolist()
        confidence = age_model.predict_proba(X).tolist()
        ood_scores = age_model.decision_distance(X).tolist()
        
        return PredictionResponse(
            predictions=predictions,
            confidence=confidence,
            ood_scores=ood_scores
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

print("FastAPI integration code provided (commented out)")

# ============================================================================
# 5. Batch Processing Integration
# ============================================================================
print("\n" + "=" * 60)
print("5. BATCH PROCESSING INTEGRATION")
print("=" * 60)

def batch_predict(model, data, batch_size=100):
    """Process large datasets in batches"""
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        predictions = model.predict(batch)
        confidence = model.predict_proba(batch)
        results.append({
            'predictions': predictions,
            'confidence': confidence,
            'batch_start': i,
            'batch_end': min(i+batch_size, len(data))
        })
    return results

# Simulate large dataset
large_data = np.random.randn(10000, 5)
age_model = AGE(min_samples=5, base_clustering='optics', enhance_ood=False, use_robust_cov=False)
age_model.fit(large_data[:1000])  # Train on subset

# Process in batches
batch_results = batch_predict(age_model, large_data[1000:], batch_size=500)
print(f"Processed {len(batch_results)} batches")
print(f"Total predictions: {sum(len(r['predictions']) for r in batch_results)}")

# ============================================================================
# 6. Monitoring and Logging Integration
# ============================================================================
print("\n" + "=" * 60)
print("6. MONITORING AND LOGGING INTEGRATION")
print("=" * 60)

class MonitoredAGE:
    """Wrapper class for AGE with monitoring capabilities"""
    
    def __init__(self, age_model):
        self.age_model = age_model
        self.stats = {
            'total_predictions': 0,
            'ood_rejections': 0,
            'avg_confidence': 0.0,
            'prediction_times': []
        }
    
    def fit(self, X, y=None):
        return self.age_model.fit(X, y)
    
    def predict(self, X):
        import time
        start_time = time.time()
        
        predictions = self.age_model.predict(X)
        confidence = self.age_model.predict_proba(X)
        
        prediction_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_predictions'] += len(predictions)
        self.stats['ood_rejections'] += np.sum(predictions == -1)
        self.stats['avg_confidence'] = np.mean(confidence[predictions != -1])
        self.stats['prediction_times'].append(prediction_time)
        
        return predictions
    
    def get_stats(self):
        avg_time = np.mean(self.stats['prediction_times']) if self.stats['prediction_times'] else 0
        return {
            **self.stats,
            'avg_prediction_time': avg_time,
            'rejection_rate': self.stats['ood_rejections'] / self.stats['total_predictions'] if self.stats['total_predictions'] > 0 else 0
        }

# Use monitored model
monitored_age = MonitoredAGE(AGE(min_samples=5, base_clustering='optics', enhance_ood=False, use_robust_cov=False))
monitored_age.fit(X)

# Make predictions
for _ in range(5):
    test_data = np.random.randn(100, 5)
    monitored_age.predict(test_data)

# Get statistics
stats = monitored_age.get_stats()
print(f"Monitoring statistics:")
print(f"  Total predictions: {stats['total_predictions']}")
print(f"  OOD rejections: {stats['ood_rejections']}")
print(f"  Rejection rate: {stats['rejection_rate']:.3f}")
print(f"  Average confidence: {stats['avg_confidence']:.3f}")
print(f"  Average prediction time: {stats['avg_prediction_time']:.4f}s")

print("\n" + "=" * 60)
print("INTEGRATION GUIDE COMPLETE")
print("=" * 60)
print("Key integration patterns demonstrated:")
print("1. sklearn Pipeline integration")
print("2. Hybrid clustering + classification pipelines")
print("3. Model persistence with joblib/pickle")
print("4. Web service integration (FastAPI)")
print("5. Batch processing for large datasets")
print("6. Monitoring and logging wrapper")
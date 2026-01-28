import os
import joblib
import numpy as np
import pytest
from pathlib import Path

# Mock data for testing
@pytest.fixture
def mock_transaction():
    # 30 features (Time, V1-V28, Amount)
    return np.random.randn(1, 30)

def test_model_loading():
    """Verify that the model package is valid and loads correctly."""
    model_path = Path("fraud_model.pkl")
    assert model_path.exists(), "Model file missing!"
    
    payload = joblib.load(model_path)
    assert isinstance(payload, dict), "Invalid model payload format!"
    assert 'pipeline' in payload, "Pipeline missing from payload!"
    assert 'feature_stats' in payload, "Feature stats missing from payload!"

def test_model_inference(mock_transaction):
    """Verify that the model can perform inference and returns expected probability/prediction."""
    payload = joblib.load("fraud_model.pkl")
    pipeline = payload['pipeline']
    
    prediction = pipeline.predict(mock_transaction)
    probabilities = pipeline.predict_proba(mock_transaction)
    
    assert prediction[0] in [0, 1], "Invalid prediction class!"
    assert probabilities.shape == (1, 2), "Invalid probability output shape!"
    assert 0 <= probabilities[0][1] <= 1, "Probability out of bounds!"

def test_feature_stats_consistency():
    """Verify that all 30 features have associated statistics."""
    payload = joblib.load("fraud_model.pkl")
    stats = payload['feature_stats']
    
    expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    for col in expected_cols:
        assert col in stats, f"Missing stats for {col}"
        assert 'median' in stats[col]
        assert 'min' in stats[col]
        assert 'max' in stats[col]

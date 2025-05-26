import os
import pickle
import pytest
from src.train import train_model, load_data

@pytest.fixture(scope="module")
def training_result():
    return train_model("test")

# Make sure no missing values
def test_feature_integrity():
    df = load_data()
    assert df.isnull().sum().sum() == 0

# Accuracy is valid output
def test_model_development(training_result):
    _, accuracy = training_result
    assert 0.0 <= accuracy <= 1.0

# Check if file is saved at the right location
def test_infrastructure(training_result):
    model_path, _ = training_result
    assert os.path.exists(model_path)


import os
from pathlib import Path

import pytest

from model_training.dataset import load_data
from model_training.pipeline import run_pipeline


@pytest.fixture(scope="module")
def training_result(tmp_path_factory):
    temp_path = tmp_path_factory.mktemp("training_test")

    # Run the pipeline and return the result
    return run_pipeline(
        input_dataset=Path("data/processed/dataset.csv"),
        vectorizer_path=Path("data/processed/vectorizer.pkl"),
        output_data_dir=temp_path,
        model_dir=temp_path,
        model_name="sentiment_model_test",
        model_version="1.0.0",
    )


# Make sure no missing values
def test_feature_integrity():
    df = load_data(Path("data/processed/dataset.csv"))
    assert df.isnull().sum().sum() == 0


# Accuracy is valid output
def test_model_development(training_result):
    _, accuracy = training_result
    assert 0.0 <= accuracy <= 1.0


# Check if file is saved at the right location
def test_infrastructure(training_result):
    model_path, _ = training_result
    assert os.path.exists(model_path)

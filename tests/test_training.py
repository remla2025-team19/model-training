import os

import pytest

from model_training.dataset import load_data
from model_training.pipeline import run_pipeline


@pytest.fixture(scope="module")
def training_result(tmp_path_factory):
    temp_path = tmp_path_factory.mktemp("training_test")

    # Run the pipeline and return the result
    return run_pipeline(
        output_data_dir=temp_path,
        model_dir=temp_path,
        model_name="sentiment_model_test",
        model_version="1.0.0",
    ), temp_path


# Make sure no missing values
def test_feature_integrity(training_result):
    (_, _), temp_path = training_result
    df = load_data(temp_path / "dataset.csv")
    assert df.isnull().sum().sum() == 0


# Accuracy is valid output
def test_model_development(training_result):
    (_, accuracy), _ = training_result
    assert 0.0 <= accuracy <= 1.0


# Check if file is saved at the right location
def test_infrastructure(training_result):
    (model_path, _), _ = training_result
    assert os.path.exists(model_path)

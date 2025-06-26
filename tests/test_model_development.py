import os
from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer


import model_training.dataset as dataset_module
from model_training.dataset import preprocess_data
from model_training.pipeline import run_pipeline

TEST_SAMPLES = [
    ("Wow... Loved this place.", 1),
    ("Crust is not good.", 0),
]


@pytest.mark.model_dev
def test_dummy_accuracy_vs_real(tmp_path, monkeypatch):
    tiny_df = pd.DataFrame(
        {
            "Review": [t for t, _ in TEST_SAMPLES],
            "Liked": [is_liked for _, is_liked in TEST_SAMPLES],
        }
    )
    corpus = tiny_df["Review"].tolist()
    cv = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()  # type: ignore
    y = tiny_df["Liked"].values

    # Create temporary CSV file with tab-delimited format (same as original data)
    dummy_csv = tmp_path / "dummy_dataset.csv"
    tiny_df.to_csv(dummy_csv, sep="\t", index=False)

    # Monkeypatch preprocess_data to return dummy data (same as original test)
    monkeypatch.setattr(dataset_module, "preprocess_data", lambda *args, **kwargs: (X, y, cv))

    dummy_path, dummy_acc = run_pipeline(
        input_dataset=dummy_csv,
        output_data_dir=tmp_path,
        model_dir=tmp_path,
        model_name="sentiment_model_dummy",
        model_version="0.0.0-dummy",
    )
    assert os.path.exists(dummy_path)
    assert 0.0 <= dummy_acc <= 1.0

    # Reset preprocess_data to original function
    monkeypatch.setattr(dataset_module, "preprocess_data", preprocess_data)

    real_path, real_acc = run_pipeline(
        input_dataset=Path("data/raw/restaurant_sentiment.csv"),
        output_data_dir=tmp_path,
        model_dir=tmp_path,
        model_name="sentiment_model_real",
        model_version="0.0.0-real",
    )
    assert os.path.exists(real_path)
    assert 0.0 <= real_acc <= 1.0

    assert dummy_acc < real_acc


@pytest.mark.model_dev
@pytest.mark.parametrize("text,expected", TEST_SAMPLES)
def test_known_predictions(model, text, expected):
    assert model(text) == expected


@pytest.mark.model_dev
def test_long_review_handles(model):
    long_text = "word " * 600
    assert model(long_text) in (0, 1)

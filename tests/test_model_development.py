import os, pytest, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import model_training.pipeline as train_module
from model_training.dataset import preprocess_data

TEST_SAMPLES = [
    ("Wow... Loved this place.", 1),
    ("Crust is not good.", 0),
]

@pytest.mark.model_dev
def test_dummy_accuracy_vs_real(tmp_path, monkeypatch):
    tiny_df = pd.DataFrame({
        "Review": [t for t,_ in TEST_SAMPLES],
        "Liked":  [l for _,l in TEST_SAMPLES],
    })
    corpus = tiny_df["Review"].tolist()
    cv = CountVectorizer()
    X = cv.fit_transform(corpus).toarray()
    y = tiny_df["Liked"].values

    monkeypatch.setattr(
        train_module,
        "preprocess_data",
        lambda *args, **kwargs: (X, y, cv)
    )

    dummy_path, dummy_acc = train_module.run_pipeline("0.0.0-dummy")
    assert os.path.exists(dummy_path)
    assert 0.0 <= dummy_acc <= 1.0

    monkeypatch.setattr(
        train_module,
        "preprocess_data",
        preprocess_data
    )

    real_path, real_acc = train_module.run_pipeline("0.0.0-real")
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

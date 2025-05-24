import os
import pytest
import pandas as pd
import src.train as train_module

TEST_SAMPLES = [
    ("Wow I loved this place!", 1),
    ("Food is not good!", 0),
]

def test_dummy_accuracy_vs_real(tmp_path, monkeypatch):
    tiny_df = pd.DataFrame({
        "Review": [text for text, _ in TEST_SAMPLES],
        "Liked":  [label for _, label in TEST_SAMPLES],
    })
    tiny_file = tmp_path / "tiny.tsv"
    tiny_df.to_csv(tiny_file, sep="\t", index=False)

    original_loader = train_module.load_data
    monkeypatch.setattr(train_module, "load_data", lambda: pd.read_csv(tiny_file, sep="\t"))

    dummy_model_path, dummy_accuracy = train_module.train_model("0.0.0-dummy")
    assert os.path.exists(dummy_model_path)
    assert 0.0 <= dummy_accuracy <= 1.0

    monkeypatch.setattr(train_module, "load_data", original_loader)
    _, real_accuracy = train_module.train_model("0.0.0-real")
    assert 0.0 <= real_accuracy <= 1.0

    assert dummy_accuracy < real_accuracy, (
        f"Dummy accuracy ({dummy_accuracy:.4f}) should be lower than real accuracy ({real_accuracy:.4f})"
    )

@pytest.mark.parametrize("text,expected", TEST_SAMPLES)
def test_known_predictions(model, text, expected):
    assert model(text) == expected

def test_long_review_handles(model):
    long_text = "word " * 600
    assert model(long_text) in (0, 1)

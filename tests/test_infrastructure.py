import os, hashlib, pickle, pytest
from model_training.pipeline import run_pipeline

MODEL_V = "1.0.0"

def _hash_bytes(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def test_roundtrip_predictions(tmp_path, model):

    examples = [
        "The food was delicious!",
        "I did not enjoy the service."
    ]

    orig_path, _ = run_pipeline(MODEL_V)
    with open(orig_path, "rb") as f:
        orig_data = pickle.load(f)

    re_path = tmp_path / "roundtrip.pkl"
    with open(re_path, "wb") as f:
        pickle.dump(orig_data, f)

    with open(re_path, "rb") as f:
        new_data = pickle.load(f)

    clf_new, vec_new = new_data["classifier"], new_data["vectorizer"]
    for text in examples:
        assert model(text) == clf_new.predict(vec_new.transform([text]).toarray())[0]

def test_deterministic_training(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    p1, acc1 = run_pipeline("1.0.0-det-1")
    p2, acc2 = run_pipeline("1.0.0-det-2")

    assert abs(acc1 - acc2) < 0.02, "Accuracy drifted >2 pp between runs"

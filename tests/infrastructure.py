import os, hashlib, pickle, pytest
from src.train import train_model

MODEL_V = "1.0.0"

def _hash_bytes(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def test_save_and_load_identical(tmp_path):
    model_path, _ = train_model(MODEL_V)

    # Hash before reload
    h_before = _hash_bytes(model_path)

    with open(model_path, "rb") as f:
        before = pickle.load(f)

    # Re-save immediately
    re_path = tmp_path / "re.pkl"
    with open(re_path, "wb") as f:
        pickle.dump(before, f)

    h_after = _hash_bytes(re_path)
    assert h_before == h_after, "Byte-wise hash changed after save/load cycle"

def test_deterministic_training(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    p1, acc1 = train_model("1.0.0-det-1")
    p2, acc2 = train_model("1.0.0-det-2")

    assert abs(acc1 - acc2) < 0.02, "Accuracy drifted >2 pp between runs"

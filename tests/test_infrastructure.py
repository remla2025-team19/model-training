import os, hashlib, pickle, pytest, time
import psutil
from model_training.pipeline import run_pipeline
from model_training.dataset import preprocess_data

MODEL_V = "1.0.0"

def _hash_bytes(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

@pytest.mark.infrastructure
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

@pytest.mark.infrastructure
def test_deterministic_training(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    p1, acc1 = run_pipeline("1.0.0-det-1")
    p2, acc2 = run_pipeline("1.0.0-det-2")

    assert abs(acc1 - acc2) < 0.02, "Accuracy drifted >2 pp between runs"

@pytest.mark.infrastructure
def test_feature_extraction_cost():
    """Test computational cost of feature extraction (time and memory)."""
    # Measure time for feature extraction
    start_time = time.perf_counter()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run feature extraction
    X, y, cv = preprocess_data()
    
    end_time = time.perf_counter()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    extraction_time = end_time - start_time
    memory_usage = end_memory - start_memory
    
    # Assert reasonable computational costs
    assert extraction_time < 30.0, f"Feature extraction took {extraction_time:.2f}s, should be < 30s"
    assert memory_usage < 500, f"Feature extraction used {memory_usage:.1f}MB, should be < 500MB"
    
    # Test feature dimensionality cost
    n_features = X.shape[1]
    assert n_features <= 1420, f"Feature count {n_features} exceeds expected max of 1420"
    
    # Test sparsity - efficient representation
    non_zero_ratio = (X != 0).sum() / X.size
    assert non_zero_ratio < 0.5, f"Feature matrix too dense ({non_zero_ratio:.2%}), should be sparse"

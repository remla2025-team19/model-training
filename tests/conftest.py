import os, pickle, pytest, pandas as pd
from src.train import train_model

_MODEL_PATH = "models/sentiment_model_v1.0.0.pkl"

@pytest.fixture(scope="session")
def model():
    """
    Session-wide fixture that either loads an existing trained model or
    trains a fresh one (quickly) if the .pkl file is missing.
    Returns a simple .predict(text) callable.
    """
    if not os.path.exists(_MODEL_PATH):
        # Train on the full dataset (or a small sample if you prefer)
        train_model("1.0.0")

    with open(_MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    clf, vec = data["classifier"], data["vectorizer"]
    return lambda text: clf.predict(vec.transform([text]).toarray())[0]

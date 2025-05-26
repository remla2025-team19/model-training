import os, pickle, pytest
from model_training.pipeline import run_pipeline

_MODEL_PATH = os.path.join("models", "sentiment_model_v1.0.0.pkl")

@pytest.fixture(scope="session")
def model():

    if not os.path.exists(_MODEL_PATH):
        # run pipeline to generate the model
        run_pipeline("1.0.0")

    with open(_MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    clf, vec = data["classifier"], data["vectorizer"]
    return lambda text: clf.predict(vec.transform([text]).toarray())[0]
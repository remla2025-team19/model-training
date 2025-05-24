import pickle, pytest

@pytest.fixture(scope="session")
def model():
    with open("models/sentiment_model_v1.0.0.pkl", "rb") as f:
        data = pickle.load(f)
    # simple predict API:
    clf, vec = data["classifier"], data["vectorizer"]
    return lambda text: clf.predict(vec.transform([text]).toarray())[0]

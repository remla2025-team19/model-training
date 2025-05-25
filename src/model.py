import os
import pickle
from sklearn.naive_bayes import GaussianNB


def train_model(X_train, y_train):
    """Train the sentiment analysis model"""

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    return classifier


def save_model(classifier, vectorizer, version):
    """Save the trained model and vectorizer"""
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_data = {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "version": version,
    }

    model_path = os.path.join(model_dir, f"sentiment_model_v{version}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {model_path}")
    return model_path

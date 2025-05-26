import os
import pickle

from loguru import logger
from sklearn.naive_bayes import GaussianNB

from model_training.config import MODELS_DIR


def train_model(X_train, y_train):
    """Train the sentiment analysis model"""
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier


def save_model(classifier, vectorizer, version):
    """Save the trained model and vectorizer"""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_data = {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "version": version,
    }

    model_path = MODELS_DIR / f"sentiment_model_v{version}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {model_path}")
    return model_path

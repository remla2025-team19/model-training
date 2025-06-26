import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger
from sklearn.naive_bayes import GaussianNB

from model_training.config import MODELS_DIR, PROCESSED_DATA_DIR
from model_training.utils import load_params

app = typer.Typer()


def train_classifier(X_train, y_train):
    """Train the sentiment analysis model"""
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier


def load_vectorizer(vectorizer_path: Path):
    """Load the CountVectorizer from file"""
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info(f"CountVectorizer loaded from {vectorizer_path}")
    return vectorizer


def save_model(classifier, vectorizer, version, output_path: Path):
    """Save the trained model and vectorizer"""
    model_data = {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "version": version,
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {output_path}")


@app.command()
def train(
    train_dataset_path: Path = PROCESSED_DATA_DIR / "train_dataset.csv",
    vectorizer_path: Path = PROCESSED_DATA_DIR / "vectorizer.pkl",
    model_output_dir: Path = MODELS_DIR,
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    params_file: Path = Path("params.yaml"),
):
    """Train the sentiment analysis model.

    Args:
        train_dataset_path: Path to the training dataset CSV
        vectorizer_path: Path to the saved CountVectorizer
        model_output_dir: Directory to save the trained model
        model_name: Name of the model file (without extension)
        version: Version string for the model
        params_file: Path to parameters YAML file
    """
    # Load parameters from YAML file
    params = load_params(params_file)
    model_params = params.get("model", {})

    # Use CLI arguments if provided, otherwise use params.yaml values, otherwise use defaults
    final_model_name = (
        model_name if model_name is not None else model_params.get("name", "sentiment_model")
    )
    final_version = (
        version if version is not None else model_params.get("version", "1.0.0")
    )

    logger.info(f"Training model version: {final_version}")
    logger.info("Loading training dataset...")
    train_data = pd.read_csv(train_dataset_path)

    # Separate features and labels
    X_train = train_data.drop(columns=["Label"]).to_numpy()
    y_train = train_data["Label"].to_numpy()

    logger.info("Loading CountVectorizer...")
    vectorizer = load_vectorizer(vectorizer_path)

    logger.info("Training model...")
    classifier = train_classifier(X_train, y_train)

    logger.info("Saving model...")
    model_output_path = model_output_dir / f"{final_model_name}_v{final_version}.pkl"
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(classifier, vectorizer, final_version, model_output_path)

    logger.success(f"Model training completed! Model saved to: {model_output_path}")
    return model_output_path


if __name__ == "__main__":
    app()

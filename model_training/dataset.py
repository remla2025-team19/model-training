import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import typer
from lib_ml.preprocessing import TextPreprocessor
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from model_training.config import (
    DEFAULT_TRAINING_DATA_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()


def get_data(
    input_dataset_url: str = DEFAULT_TRAINING_DATA_URL,
    output_path: Path = RAW_DATA_DIR / "dataset.csv",
):
    """Download dataset from the given URL and save it to the specified path.
    Args:
        input_dataset_url (str): URL to download the dataset from.
        output_path (Path): Path to save the downloaded dataset.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading dataset from {input_dataset_url} to {output_path}...")
    try:
        response = requests.get(input_dataset_url, stream=True, timeout=15)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.success(f"Successfully downloaded dataset to {output_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading dataset from {input_dataset_url}: {e}")
        raise typer.Exit(code=1)


@app.command()
def download(
    input_dataset_url: str = DEFAULT_TRAINING_DATA_URL,
    output_path: Path = RAW_DATA_DIR / "restaurant_sentiment.csv",
):
    """Download the restaurant sentiment dataset from the given URL.
    Args:
        input_dataset_url (str): URL to download the dataset from.
        output_path (Path): Path to save the downloaded dataset.
    """
    return get_data(input_dataset_url, output_path)


def preprocess_data(
    input_path: Path = RAW_DATA_DIR / "restaurant_sentiment.csv",
    output_path: Optional[Path] = None,
    vectorizer_path: Optional[Path] = None,
):
    """Preprocess the restaurant sentiment dataset.
    Args:
        input_path (Path): Path to the raw dataset.
        output_path (Optional[Path]): Path to save the processed dataset.
        vectorizer_path (Optional[Path]): Path to save the CountVectorizer."""

    logger.info("Loading raw data...")
    dataset = pd.read_csv(input_path, delimiter="\t", quoting=3)

    logger.info("Processing dataset...")
    preprocessor = TextPreprocessor()

    corpus = preprocessor.preprocess_texts(dataset["Review"].tolist())

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()  # type: ignore # pylint: disable=data-leakage-detected
    y = dataset.iloc[:, -1].values

    logger.info("Saving processed dataset...")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(X, columns=cv.get_feature_names_out())
        df["Label"] = y
        df.to_csv(output_path, index=False)

    # Save the CountVectorizer for later use in training
    if vectorizer_path is not None:
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(cv, f)
        logger.info(f"CountVectorizer saved to {vectorizer_path}")

    logger.success("Processing dataset complete.")
    return X, y, cv


@app.command()
def preprocess(
    input_path: Path = RAW_DATA_DIR / "restaurant_sentiment.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    vectorizer_path: Path = PROCESSED_DATA_DIR / "vectorizer.pkl",
):
    """Preprocess the restaurant sentiment dataset and save it to the specified path.

    Args:
        input_path (Path): Path to the raw dataset.
        output_path (Path): Path to save the processed dataset.
        vectorizer_path (Path): Path to save the CountVectorizer.
    """
    return preprocess_data(input_path, output_path, vectorizer_path)


def split_data(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train_dataset.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test_dataset.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split processed dataset into train and test sets.

    Args:
        input_path: Path to the processed dataset
        train_output_path: Path to save the training dataset
        test_output_path: Path to save the test dataset
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducible splits
    """
    logger.info(f"Loading processed dataset from {input_path}")
    dataset = pd.read_csv(input_path)

    # Separate features and labels
    X = dataset.drop(columns=["Label"])
    y = dataset["Label"]

    # Split data
    logger.info(
        f"Splitting data with test_size={test_size}, random_state={random_state}"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save train dataset
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    train_dataset = X_train.copy()
    train_dataset["Label"] = y_train
    train_dataset.to_csv(train_output_path, index=False)
    logger.info(f"Training dataset saved to {train_output_path}")

    # Save test dataset
    test_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_dataset = X_test.copy()
    test_dataset["Label"] = y_test
    test_dataset.to_csv(test_output_path, index=False)
    logger.info(f"Test dataset saved to {test_output_path}")

    logger.success(
        f"Dataset split complete. Train: {len(train_dataset)}, Test: {len(test_dataset)}"
    )


@app.command()
def split(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    train_output_path: Path = PROCESSED_DATA_DIR / "train_dataset.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test_dataset.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split processed dataset into train and test sets.

    Args:
        input_path: Path to the processed dataset
        train_output_path: Path to save the training dataset
        test_output_path: Path to save the test dataset
        test_size: Proportion of dataset to include in test split
        random_state: Random state for reproducible splits
    """
    return split_data(
        input_path, train_output_path, test_output_path, test_size, random_state
    )


def load_data(path: Optional[Path] = None):
    """Load the processed restaurant reviews dataset for training/testing.

    Args:
        path: Optional path to the dataset. If None, uses default processed dataset path.

    Returns:
        pd.DataFrame: Loaded dataset
    """
    if path is None:
        path = PROCESSED_DATA_DIR / "dataset.csv"

    logger.info(f"Loading dataset from {path}")
    return pd.read_csv(path)


@app.command()
def prepare(
    input_dataset_url: str = DEFAULT_TRAINING_DATA_URL,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    vectorizer_path: Path = PROCESSED_DATA_DIR / "vectorizer.pkl",
    split: bool = True,
    train_output_path: Path = PROCESSED_DATA_DIR / "train_dataset.csv",
    test_output_path: Path = PROCESSED_DATA_DIR / "test_dataset.csv",
):
    """Download, process, and split the restaurant sentiment dataset."""
    raw_path = RAW_DATA_DIR / "restaurant_sentiment.csv"
    get_data(input_dataset_url, raw_path)
    preprocess_data(
        input_path=raw_path, output_path=output_path, vectorizer_path=vectorizer_path
    )
    if split:
        split_data(
            input_path=output_path,
            train_output_path=train_output_path,
            test_output_path=test_output_path,
        )


if __name__ == "__main__":
    app()

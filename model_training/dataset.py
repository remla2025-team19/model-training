from pathlib import Path
from typing import Optional

from lib_ml.preprocessing import TextPreprocessor
from loguru import logger
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
import typer

from model_training.config import DEFAULT_TRAINING_DATA_URL, PROCESSED_DATA_DIR, RAW_DATA_DIR

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
    get_data(input_dataset_url, output_path)

def preprocess_data(
    input_path: str = RAW_DATA_DIR / "restaurant_sentiment.csv",
    output_path: Optional[Path] = None,
):

    """Preprocess the restaurant sentiment dataset.
    Args:
        input_path (str): Path to the raw dataset.
        output_path (Optional[Path]): Path to save the processed dataset."""

    logger.info("Loading raw data...")
    dataset = pd.read_csv(input_path, delimiter="\t", quoting=3)

    logger.info("Processing dataset...")
    preprocessor = TextPreprocessor()

    corpus = preprocessor.preprocess_texts(dataset["Review"].tolist())

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray() # pylint: disable=data-leakage-detected
    y = dataset.iloc[:, -1].values

    logger.info("Saving processed dataset...")

    # TODO: ? # pylint: disable=fixme
    if output_path is not None:
        df = pd.DataFrame(X, columns=cv.get_feature_names_out())
        df["Label"] = y
        df.to_csv(output_path, index=False)

    logger.success("Processing dataset complete.")
    return X, y, cv


@app.command()
def main(
    input_dataset_url: str = DEFAULT_TRAINING_DATA_URL,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """Download and preprocess the restaurant sentiment dataset."""
    raw_path = RAW_DATA_DIR / "restaurant_sentiment.csv"
    get_data(input_dataset_url,raw_path)
    preprocess_data(input_path=raw_path, output_path=output_path)
    #preprocess_data(input_dataset_url, output_path)


if __name__ == "__main__":
    app()

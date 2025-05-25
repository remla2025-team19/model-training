from pathlib import Path
from typing import Optional

from lib_ml.preprocessing import TextPreprocessor
from loguru import logger
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import typer

from model_training.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def preprocess_data(input_path: Path, output_path: Optional[Path] = None):
    logger.info("Loading raw data...")
    dataset = pd.read_csv(input_path, delimiter="\t", quoting=3)

    logger.info("Processing dataset...")
    preprocessor = TextPreprocessor()

    corpus = preprocessor.preprocess_texts(dataset["Review"].tolist())

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    logger.info("Saving processed dataset...")

    # TODO: ?
    if output_path is not None:
        df = pd.DataFrame(X, columns=cv.get_feature_names_out())
        df["Label"] = y
        df.to_csv(output_path, index=False)

    logger.success("Processing dataset complete.")
    return X, y, cv


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    preprocess_data(input_path, output_path)


if __name__ == "__main__":
    app()

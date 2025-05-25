import pandas as pd
from lib_ml.preprocessing import TextPreprocessor
from sklearn.feature_extraction.text import CountVectorizer


def load_data():
    """Load the restaurant reviews dataset"""
    return pd.read_csv(
        "data/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3
    )


def preprocess_data():
    """Preprocess the dataset"""

    dataset = load_data()

    preprocessor = TextPreprocessor()

    corpus = preprocessor.preprocess_texts(dataset["Review"].tolist())

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    return X, y, cv

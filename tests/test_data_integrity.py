import pandas as pd
import pytest

from model_training.dataset import preprocess_data
from model_training.config import RAW_DATA_DIR, DEFAULT_TRAINING_DATA_URL

@pytest.fixture(scope="session")
def raw_df():
    """Load the raw restaurant reviews dataset, downloading if necessary."""
    file_name = "restaurant_sentiment.csv"
    raw_file = RAW_DATA_DIR / file_name

    # Ensure raw data is downloaded
    if not raw_file.exists():
        preprocess_data(input_path=DEFAULT_TRAINING_DATA_URL, output_path=None)

    print(f"Loading file from: {raw_file.resolve()} | Exists: {raw_file.exists()}")

    # Read the TSV into a DataFrame
    df = pd.read_csv(raw_file, delimiter='\t', quoting=3)
    return df

@pytest.mark.feature_data
def test_emptiness_and_columns(raw_df):
    assert not raw_df.empty, "DataFrame should not be empty"
    expected_cols = {'Review', 'Liked'}
    assert set(raw_df.columns) == expected_cols, f"DataFrame should contain exactly columns {expected_cols}"

@pytest.mark.feature_data
def test_types_and_values(raw_df):
    assert raw_df['Review'].apply(lambda x: isinstance(x, str)).all(), "All reviews should be strings"
    assert raw_df['Liked'].apply(lambda x: x in [0, 1]).all(), "Labels should be binary (0 or 1)"

@pytest.mark.feature_data
def test_review_length(raw_df):
    lengths = raw_df['Review'].str.len()
    assert lengths.min() > 0, "Reviews should not be empty"
    assert lengths.max() <= 500, "Reviews should not exceed 500 characters"

@pytest.mark.feature_data
def test_no_missing_values(raw_df):
    assert raw_df.isnull().sum().sum() == 0, "There should be no missing values in the dataset"

@pytest.mark.feature_data
@pytest.mark.parametrize("invalid", ["", "  ", "\n", "\t"])
def test_no_invalid_placeholders(raw_df, invalid):
    reviews = raw_df['Review'].dropna().astype(str)
    assert not reviews.str.fullmatch(invalid).any(), f"'Review' column should not contain placeholder '{invalid}'"

@pytest.mark.feature_data
def test_label_distribution(raw_df):
    unique_labels = set(raw_df['Liked'].unique())
    assert unique_labels == {0, 1}, "Both classes (0 and 1) must be present in the labels"

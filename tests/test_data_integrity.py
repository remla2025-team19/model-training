import pandas as pd
import pytest

@pytest.fixture(scope="session")
def raw_df():
    return pd.read_csv('data/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)

def test_emptiness_and_columns(raw_df):
    assert not raw_df.empty, "DataFrame should not be empty"
    expected_cols = {'Review', 'Liked'}
    assert set(raw_df.columns) == expected_cols, f"DataFrame should contain exactly columns {expected_cols}"


def test_types_and_values(raw_df):
    assert raw_df['Review'].apply(lambda x: isinstance(x, str)).all(), "All reviews should be strings"
    assert raw_df['Liked'].apply(lambda x: x in [0, 1]).all(), "Labels should be binary (0 or 1)"


def test_review_length(raw_df):
    lengths = raw_df['Review'].str.len()
    assert lengths.min() > 0, "Reviews should not be empty"
    assert lengths.max() <= 500, "Reviews should not exceed 500 characters"


def test_no_missing_values(raw_df):
    assert raw_df.isnull().sum().sum() == 0, "There should be no missing values in the dataset"

@pytest.mark.parametrize("invalid", ["", "  ", None, "\n", "\t"])
def test_no_invalid_placeholders(raw_df, invalid):
    reviews = raw_df['Review'].dropna().astype(str)
    assert not reviews.str.fullmatch(invalid).any(), f"'Review' column should not contain placeholder '{invalid}'"


def test_label_distribution(raw_df):
    unique_labels = set(raw_df['Liked'].unique())
    assert unique_labels == {0, 1}, "Both classes (0 and 1) must be present in the labels"

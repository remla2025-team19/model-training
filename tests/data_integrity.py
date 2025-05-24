import pandas as pd
import pytest

@pytest.fixture
def raw_df():
    return pd.read_csv('data/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t')

def emptiness_checks(raw_df):
    assert not raw_df.empty, "DataFrame should not be empty"
    assert 'Review' in raw_df.columns, "DataFrame should contain 'Review' column"
    assert 'Like' in raw_df.columns, "DataFrame should contain 'Like' column"
    assert len(raw_df.columns) == 2, "DataFrame should only contain 'Review' and 'Like' columns"

def type_checks(raw_df):
    assert raw_df['Review'].apply(lambda x: isinstance(x, str)).all(), "All reviews should be strings"
    assert raw_df['Like'].apply(lambda x: x in [0, 1]).all(), "Likes should be binary (0 or 1)"

def length_checks(raw_df):
    assert raw_df['Review'].str.len().max() <= 500, "Reviews should not exceed 500 characters"
    assert raw_df['Review'].str.len().min() > 0, "Reviews should not be empty"
    assert raw_df['Like'].nunique() == 2, "Likes should only contain two unique values (0 and 1)"

def missing_values_checks(raw_df):
    assert raw_df['Review'].isnull().sum() == 0, "There should be no missing values in 'Review' column"
    assert raw_df['Like'].isnull().sum() == 0, "There should be no missing values in 'Like' column"

@pytest.mark.parametrize("bad", ["", " ", None, "\n", "\t"])
def bad_review(raw_df, bad):
    
    assert not raw_df['Review'].str.contains(bad).any(), f"'Review' column should not contain '{bad}'"

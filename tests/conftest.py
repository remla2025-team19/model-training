import pickle
from pathlib import Path
import pytest
from model_training.pipeline import run_pipeline

_MODEL_PATH = Path("models") / "sentiment_model_v1.0.0.pkl"

# Rubric definition
RUBRIC_QUESTIONS = {
    "feature_data": 7,
    "model_dev": 5,
    "infrastructure": 4,
    "monitoring": 4,
}

# Tracker: how many passing tests per category
category_pass_counts = {key: 0 for key in RUBRIC_QUESTIONS}


@pytest.fixture(scope="session")
def model():
    if not _MODEL_PATH.exists():
        run_pipeline(
            input_dataset=Path("data/raw/restaurant_sentiment.csv"),
            output_data_dir=Path("data/processed"),
            model_dir=Path("models"),
            model_name="sentiment_model",
            model_version="1.0.0",
        )
    with open(_MODEL_PATH, "rb", encoding="utf-8") as f:
        data = pickle.load(f)
    clf, vec = data["classifier"], data["vectorizer"]
    return lambda text: clf.predict(vec.transform([text]).toarray())[0]


# Add marker info to each item
def pytest_collection_modifyitems(items):
    for item in items:
        for mark in item.iter_markers():
            if mark.name in RUBRIC_QUESTIONS:
                item.user_properties.append(("category", mark.name))


# After test runs, count passed tests per marker
def pytest_runtest_logreport(report):
    if report.when == "call" and report.passed:
        for prop in getattr(report, "user_properties", []):
            if prop[0] == "category":
                category_pass_counts[prop[1]] += 1


# Print detailed summary based on rubric
def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_sep("=", "ML Test Score Summary")
    total_score = 0
    max_score = 0

    for cat, max_cat_score in RUBRIC_QUESTIONS.items():
        passed = category_pass_counts.get(cat, 0)
        score = min(passed, max_cat_score)
        total_score += score
        max_score += max_cat_score
        terminalreporter.write_line(
            f"{cat.title().replace('_', ' '):25}: {score} / {max_cat_score}"
        )

    percentage = (total_score / max_score) * 100 if max_score else 0
    terminalreporter.write_line(f"::ML_TEST_SCORE::{percentage:.0f}")

    with open("reports/ml_test_score.txt", "w", encoding="utf-8") as f:
        f.write(f"{percentage:.0f}")

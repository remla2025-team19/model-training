# import os, pickle, pytest
# from model_training.pipeline import run_pipeline

# _MODEL_PATH = os.path.join("models", "sentiment_model_v1.0.0.pkl")

# # Store covered categories here
# COVERED_CATEGORIES = set()

# @pytest.fixture(scope="session")
# def model():

#     if not os.path.exists(_MODEL_PATH):
#         # run pipeline to generate the model
#         run_pipeline("1.0.0")

#     with open(_MODEL_PATH, "rb") as f:
#         data = pickle.load(f)

#     clf, vec = data["classifier"], data["vectorizer"]
#     return lambda text: clf.predict(vec.transform([text]).toarray())[0]

# def pytest_runtest_logreport(report):
#     """Called after each test phase (setup/call/teardown)."""
#     if report.when == "call" and report.passed:
#         for mark in getattr(report, "user_properties", []):
#             if mark[0] == "category":
#                 COVERED_CATEGORIES.add(mark[1])

# def pytest_collection_modifyitems(items):
#     """Store marker info so we can use it later."""
#     for item in items:
#         for mark in item.iter_markers():
#             if mark.name in {"feature_data", "model_dev", "infrastructure", "monitoring"}:
#                 item.user_properties.append(("category", mark.name))

# def pytest_terminal_summary(terminalreporter):
#     categories = {
#         "feature_data": "Feature and Data",
#         "model_dev": "Model Development",
#         "infrastructure": "ML Infrastructure",
#         "monitoring": "Monitoring",
#     }

#     terminalreporter.write_sep("=", "ML Test Score Summary")
#     for key, name in categories.items():
#         status = "✅ Covered" if key in COVERED_CATEGORIES else "❌ Missing"
#         terminalreporter.write_line(f"{name:25}: {status}")

#     score = len(COVERED_CATEGORIES) / len(categories) * 100
#     terminalreporter.write_line(f"\n✅ ML Test Score Adequacy: {score:.0f}%")

import os, pickle, pytest
from model_training.pipeline import run_pipeline

_MODEL_PATH = os.path.join("models", "sentiment_model_v1.0.0.pkl")

# Rubric definition
RUBRIC_QUESTIONS = {
    "feature_data": 7,
    "model_dev": 5,
    "infrastructure": 4,
    "monitoring": 4
}

# Tracker: how many passing tests per category
category_pass_counts = {key: 0 for key in RUBRIC_QUESTIONS}

@pytest.fixture(scope="session")
def model():
    if not os.path.exists(_MODEL_PATH):
        run_pipeline("1.0.0")
    with open(_MODEL_PATH, "rb") as f:
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
        terminalreporter.write_line(f"{cat.title().replace('_', ' '):25}: {score} / {max_cat_score}")

    percentage = (total_score / max_score) * 100 if max_score else 0
    terminalreporter.write_line(f"::ML_TEST_SCORE::{percentage:.0f}")

    with open("ml_test_score.txt", "w") as f:
        f.write(percentage)

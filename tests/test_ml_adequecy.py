import pytest

RUBRIC_QUESTIONS = {
    "feature_data": 7,
    "model_dev": 5,
    "infra": 4,
    "monitoring": 4
}

def pytest_sessionfinish(session, exitstatus):
    category_counts = {k: 0 for k in RUBRIC_QUESTIONS}

    for item in session.items:
        for marker in RUBRIC_QUESTIONS:
            if item.get_closest_marker(marker):
                category_counts[marker] += 1

    print("\n=== ML Test Score Summary ===")
    total_score = 0
    max_score = 0

    for cat, covered in category_counts.items():
        max_possible = RUBRIC_QUESTIONS[cat]
        score = min(covered, max_possible)
        total_score += score
        max_score += max_possible
        print(f"{cat.title().replace('_', ' ')}: {score} / {max_possible}")

    print(f"\nTotal ML Test Score: {total_score} / {max_score}")

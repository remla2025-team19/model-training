from pathlib import Path

from loguru import logger
from sklearn.metrics import accuracy_score, classification_report

from model_training.config import REPORTS_DIR


def evaluate_model(classifier, X_test, y_test, report_path: str | Path | None = None):
    """
    Evaluate a trained model and optionally save results to a report file.

    Args:
        classifier: Trained classifier model
        X_test: Test features
        y_test: Test labels
        report_path: Optional path to save evaluation report. If None, uses default in REPORTS_DIR

    Returns:
        float: Model accuracy score
    """
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    class_report = classification_report(y_test, y_pred)

    if report_path is not None:
        # Use provided path or default to REPORTS_DIR
        if report_path == "report.txt":  # Handle legacy default
            report_file = REPORTS_DIR / "evaluation_report.txt"
        else:
            report_file = Path(report_path) if isinstance(report_path, str) else report_path

        # Ensure reports directory exists
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Model accuracy: {accuracy:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(class_report)

        logger.info(f"Evaluation report saved to {report_file}")
    else:
        # Log results instead of printing
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(f"\n{class_report}")

    return accuracy

import json
import pickle
from pathlib import Path
from typing import Optional

from loguru import logger
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import typer


from model_training.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from model_training.utils import load_params

app = typer.Typer()


def load_model(model_path: Path):
    """Load the trained model from file"""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return model_data["classifier"], model_data["vectorizer"]


def compute_internal_metrics(classifier, X_test, y_test):
    """Internal function to evaluate a model and return metrics"""
    y_pred = classifier.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "test_samples": len(y_test),
    }

    # Generate classification report
    class_report = classification_report(y_test, y_pred)

    return metrics, class_report


@app.command()
def evaluate(
    model_dir: Path = MODELS_DIR,
    model_name: str = "sentiment_model",
    model_version: Optional[str] = None,
    test_dataset_path: Path = PROCESSED_DATA_DIR / "test_dataset.csv",
    metrics_output_path: Path = REPORTS_DIR / "evaluation_metrics.json",
    report_output_path: Path = REPORTS_DIR / "evaluation_report.txt",
    params_file: Path = Path("params.yaml"),
):
    """Evaluate the trained sentiment analysis model.

    Args:
        model_dir: Directory containing the trained model
        model_name: Name of the model file (without version and extension)
        model_version: Version of the model to evaluate
        test_dataset_path: Path to the test dataset CSV
        metrics_output_path: Path to save the evaluation metrics JSON
        report_output_path: Path to save the detailed evaluation report
        params_file: Path to parameters YAML file
    """
    # Load parameters from YAML file
    params = load_params(params_file)
    train_params = params.get("train", {})  # Version comes from train params

    # Use CLI arguments if provided, otherwise use params.yaml values, otherwise use defaults
    final_version = (
        model_version
        if model_version is not None
        else train_params.get("version", "1.0.0")
    )

    logger.info(f"Evaluating model version: {final_version}")
    logger.info("Loading trained model...")
    model_path = model_dir / f"{model_name}_v{final_version}.pkl"
    classifier, _ = load_model(model_path)

    logger.info("Loading test dataset...")
    test_data = pd.read_csv(test_dataset_path)

    # Separate features and labels
    X_test = test_data.drop(columns=["Label"]).to_numpy()
    y_test = test_data["Label"].to_numpy()

    logger.info("Evaluating model...")
    metrics, class_report = compute_internal_metrics(classifier, X_test, y_test)

    # Save metrics as JSON
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_output_path}")

    # Save detailed report
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(f"Model accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Test samples: {metrics['test_samples']}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)  # type: ignore
    logger.info(f"Evaluation report saved to {report_output_path}")

    logger.success(f"Model evaluation completed! Accuracy: {metrics['accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    app()

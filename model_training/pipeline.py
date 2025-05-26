from loguru import logger
from sklearn.model_selection import train_test_split
import typer

from model_training.config import REPORTS_DIR
from model_training.dataset import preprocess_data
from model_training.modeling import evaluate_model, save_model, train_model

app = typer.Typer()


def run_pipeline(version: str = "1.0.0"):
    """
    Run the complete training pipeline.

    Args:
        version: Model version string for saving and reports

    Returns:
        tuple: (model_path, accuracy) - Path to saved model and accuracy score
    """
    logger.info(f"Starting training pipeline for version {version}")

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    X, y, cv = preprocess_data()

    # Split data
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    logger.info("Training model...")
    model = train_model(X_train, y_train)

    # Save model
    logger.info("Saving model...")
    model_path = save_model(model, cv, version)

    # Evaluate model
    logger.info("Evaluating model...")
    report_path = REPORTS_DIR / f"report_v{version}.txt"
    accuracy = evaluate_model(model, X_test, y_test, report_path)

    logger.success(f"Pipeline complete! Model saved to: {model_path}")
    logger.success(f"Model accuracy: {accuracy:.4f}")

    return model_path, accuracy


@app.command()
def main(version: str = typer.Argument("1.0.0", help="Model version for saving and reports")):
    """Run the complete training pipeline."""
    model_path, accuracy = run_pipeline(version)

    logger.success(f"Model training complete. Model path: {model_path}")
    logger.success(f"Model accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    app()

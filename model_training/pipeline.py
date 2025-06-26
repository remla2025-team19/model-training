from json import load
from pathlib import Path

from model_training.dataset import load_data, split_data, preprocess_data
from model_training.modeling.evaluate import compute_internal_metrics
from model_training.modeling.train import load_vectorizer, save_model, train_classifier


def run_pipeline(
    input_dataset: Path = Path("data/raw/restaurant_sentiment.csv"),
    vectorizer_path: Path = Path("data/processed/vectorizer.pkl"),
    output_data_dir: Path = Path("data/processed"),
    model_dir: Path = Path("models"),
    model_name: str = "sentiment_model",
    model_version: str = "1.0.0",
):
    """Run the complete training pipeline and return (model_path, accuracy)
    Args:
        input_dataset: Path to the input dataset CSV file
        vectorizer_path: Path to the saved CountVectorizer
        output_data_dir: Directory to save the split datasets
        model_dir: Directory to save the trained model
        model_name: Name of the model file (without extension)
        model_version: Version string for the model
    Returns:
        Tuple[str, float]: Path to the saved model and its accuracy on the test set
    """
    # Step 0: Preprocess data
    clean_data_path = output_data_dir / "dataset.csv"
    preprocess_data(input_dataset, clean_data_path, vectorizer_path)

    # Step 1: Split data
    train_path = output_data_dir / "train_dataset.csv"
    test_path = output_data_dir / "test_dataset.csv"
    split_data(
        input_path=clean_data_path,
        train_output_path=train_path,
        test_output_path=test_path,
        test_size=0.2,
        random_state=42,
    )

    # Step 2: Load training data
    train_data = load_data(train_path)
    X_train = train_data.drop(columns=["Label"]).to_numpy()
    y_train = train_data["Label"].to_numpy()

    # Step 3: Load vectorizer and train model
    vectorizer = load_vectorizer(vectorizer_path)
    classifier = train_classifier(X_train, y_train)

    # Step 4: Save model
    model_path = model_dir / f"{model_name}_v{model_version}.pkl"
    save_model(classifier, vectorizer, model_version, model_path)

    # Step 5: Evaluate for accuracy
    test_data = load_data(test_path)
    X_test = test_data.drop(columns=["Label"]).to_numpy()
    y_test = test_data["Label"].to_numpy()

    metrics, _ = compute_internal_metrics(classifier, X_test, y_test)
    accuracy = metrics["accuracy"]

    return str(model_path), accuracy

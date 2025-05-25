from dataset import preprocess_data
from model import train_model, save_model
from sklearn.model_selection import train_test_split
from evaluate import evaluate_model


if __name__ == "__main__":
    import sys

    version = sys.argv[1] if len(sys.argv) > 1 else "1.0.0"

    X, y, cv = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train)
    model_path = save_model(model, cv, version)
    accuracy = evaluate_model(model, X_test, y_test, f"reports/report_v{version}.txt")
    print(f"Model training complete. Model path: {model_path}")
    print(f"Model accuracy: {accuracy:.4f}")

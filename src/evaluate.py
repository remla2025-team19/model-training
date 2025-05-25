from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(classifier, X_test, y_test, report_path: str | None = "report.txt"):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if report_path:
        with open(report_path, "w") as f:
            f.write(f"Model accuracy: {accuracy:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, y_pred))
    else:
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return accuracy

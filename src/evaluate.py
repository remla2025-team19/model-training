import pandas as pd
import joblib
import os
import json
from lib_ml.preprocessing import TextPreprocessor
from sklearn.feature_extraction.text import CountVectorizer
import sys
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(test_path, model_path, metrics_path, params):
    """Evaluate restaurant sentiment classification model"""
    # Load test data and model
    test_data = pd.read_csv(test_path, delimiter="\t", quoting=3)
    model = joblib.load(model_path)
    classifier = model['classifier']
    vectorizer = model['vectorizer']
    preprocessor = TextPreprocessor()

    corpus = preprocessor.preprocess_texts(test_data["Review"].tolist())

    cv = CountVectorizer(max_features=1420)
    X_test = vectorizer.transform(corpus).toarray()
    y_test = test_data['Liked']
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "test_samples": len(y_test)
    }
    
    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)['evaluate']
    
    evaluate_model(sys.argv[1], sys.argv[2], sys.argv[3], params)

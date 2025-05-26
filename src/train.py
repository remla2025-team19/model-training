import os
import pickle
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
#from lib_ml.preprocessing import TextPreprocessor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib-ml', 'src')))
from lib_ml.preprocessing import TextPreprocessor

def load_data():
    """Load the restaurant reviews dataset"""
    #return pd.read_csv('data/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
    return pd.read_csv("data/processed/dataset.csv")

def train_model(model_version="1.0.0"):
    """Train the sentiment analysis model"""
    # Load data
    dataset = load_data()
    
    # Use preprocessed dataset
    X = dataset.drop(columns=["Label"]).to_numpy()
    y = dataset["Label"].to_numpy()
    cv = None  # or load from disk if needed

    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = save_model(classifier, cv, model_version)
    
    return model_path, accuracy

def save_model(classifier, vectorizer, version):
    """Save the trained model and vectorizer"""
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'classifier': classifier,
        'vectorizer': vectorizer,
        'version': version
    }
    
    model_path = os.path.join(model_dir, f"sentiment_model_v{version}.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else "1.0.0"
    model_path, accuracy = train_model(version)
    print(f"Model training complete. Model path: {model_path}")
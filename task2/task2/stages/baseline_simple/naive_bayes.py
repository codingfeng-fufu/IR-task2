"""Simple Naive Bayes classifier - no feature engineering, basic implementation"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from typing import List
import numpy as np
import joblib


class SimpleNaiveBayes:
    """
    Simple Naive Bayes classifier using basic TF-IDF features.
    No feature engineering or optimization.
    """

    def __init__(self):
        # Use simple TF-IDF with default parameters
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=5000,  # Limit vocabulary size
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        self.classifier = MultinomialNB()
        self.is_trained = False

    def train(self, titles: List[str], labels: List[int]):
        """
        Train the Naive Bayes classifier.

        Args:
            titles: List of title strings
            labels: List of labels (1 or 0)
        """
        print("\n=== Training Naive Bayes Classifier ===")
        print(f"Training samples: {len(titles)}")

        # Convert text to TF-IDF features
        X = self.vectorizer.fit_transform(titles)
        print(f"Feature dimensions: {X.shape}")

        # Train classifier
        self.classifier.fit(X, labels)
        self.is_trained = True

        print("Training completed!")

    def predict(self, titles: List[str]) -> np.ndarray:
        """
        Predict labels for titles.

        Args:
            titles: List of title strings

        Returns:
            predictions: Array of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        X = self.vectorizer.transform(titles)
        predictions = self.classifier.predict(X)
        return predictions

    def get_feature_vectors(self, titles: List[str]) -> np.ndarray:
        """
        Get TF-IDF feature vectors for visualization.

        Args:
            titles: List of title strings

        Returns:
            Feature vectors as dense array
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        X = self.vectorizer.transform(titles)
        return X.toarray()

    def save_model(self, path: str):
        """
        Save the trained model.

        Args:
            path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a trained model.

        Args:
            path: Path to load model from
        """
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.is_trained = True
        print(f"Model loaded from {path}")

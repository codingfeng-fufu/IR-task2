"""Simple Word2Vec + SVM classifier - basic implementation"""

from gensim.models import Word2Vec
from sklearn.svm import SVC
from typing import List
import numpy as np
import joblib


class SimpleWord2VecSVM:
    """
    Simple Word2Vec + SVM classifier.
    Uses basic word embeddings averaged to get sentence representations.
    """

    def __init__(self, vector_size: int = 100, window: int = 5):
        """
        Initialize Word2Vec + SVM classifier.

        Args:
            vector_size: Dimension of word embeddings
            window: Context window size for Word2Vec
        """
        self.vector_size = vector_size
        self.window = window
        self.w2v_model = None
        self.svm_model = None
        self.is_trained = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and lowercasing."""
        return text.lower().split()

    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Convert text to vector by averaging word embeddings.

        Args:
            text: Input text

        Returns:
            Vector representation of text
        """
        tokens = self._tokenize(text)
        vectors = []

        for token in tokens:
            if token in self.w2v_model.wv:
                vectors.append(self.w2v_model.wv[token])

        if len(vectors) == 0:
            # Return zero vector if no words found
            return np.zeros(self.vector_size)

        # Average all word vectors
        return np.mean(vectors, axis=0)

    def train(self, titles: List[str], labels: List[int]):
        """
        Train Word2Vec + SVM classifier.

        Args:
            titles: List of title strings
            labels: List of labels (1 or 0)
        """
        print("\n=== Training Word2Vec + SVM Classifier ===")
        print(f"Training samples: {len(titles)}")

        # Tokenize all titles
        tokenized_titles = [self._tokenize(title) for title in titles]

        # Train Word2Vec model
        print("Training Word2Vec model...")
        self.w2v_model = Word2Vec(
            sentences=tokenized_titles,
            vector_size=self.vector_size,
            window=self.window,
            min_count=2,  # Ignore words that appear less than 2 times
            workers=4,
            epochs=10
        )
        print(f"Word2Vec vocabulary size: {len(self.w2v_model.wv)}")

        # Convert titles to vectors
        print("Converting titles to vectors...")
        X = np.array([self._text_to_vector(title) for title in titles])
        print(f"Feature dimensions: {X.shape}")

        # Train SVM classifier
        print("Training SVM classifier...")
        self.svm_model = SVC(kernel='linear', C=1.0)
        self.svm_model.fit(X, labels)

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

        X = np.array([self._text_to_vector(title) for title in titles])
        predictions = self.svm_model.predict(X)
        return predictions

    def get_feature_vectors(self, titles: List[str]) -> np.ndarray:
        """
        Get word2vec feature vectors for visualization.

        Args:
            titles: List of title strings

        Returns:
            Feature vectors
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        X = np.array([self._text_to_vector(title) for title in titles])
        return X

    def save_model(self, path_prefix: str):
        """
        Save the trained model.

        Args:
            path_prefix: Prefix for model files
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        # Save Word2Vec model
        w2v_path = f"{path_prefix}_w2v.model"
        self.w2v_model.save(w2v_path)

        # Save SVM model
        svm_path = f"{path_prefix}_svm.pkl"
        joblib.dump(self.svm_model, svm_path)

        print(f"Models saved to {path_prefix}_*")

    def load_model(self, path_prefix: str):
        """
        Load a trained model.

        Args:
            path_prefix: Prefix for model files
        """
        # Load Word2Vec model
        w2v_path = f"{path_prefix}_w2v.model"
        self.w2v_model = Word2Vec.load(w2v_path)

        # Load SVM model
        svm_path = f"{path_prefix}_svm.pkl"
        self.svm_model = joblib.load(svm_path)

        self.is_trained = True
        print(f"Models loaded from {path_prefix}_*")

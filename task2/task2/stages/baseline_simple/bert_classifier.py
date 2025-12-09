"""Simple BERT classifier - basic implementation using pre-trained BERT"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from typing import List
import numpy as np
from tqdm import tqdm


class TitleDataset(Dataset):
    """Simple dataset for title classification."""

    def __init__(self, titles: List[str], labels: List[int], tokenizer, max_length: int = 64):
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class SimpleBERT:
    """
    Simple BERT classifier for binary classification.
    Uses pre-trained bert-base-uncased without optimization.
    """

    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 64):
        """
        Initialize BERT classifier.

        Args:
            model_name: Name of pre-trained BERT model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)

        self.is_trained = False

    def train(self, titles: List[str], labels: List[int], epochs: int = 3, batch_size: int = 16):
        """
        Train BERT classifier.

        Args:
            titles: List of title strings
            labels: List of labels (1 or 0)
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        print("\n=== Training BERT Classifier ===")
        print(f"Training samples: {len(titles)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

        # Create dataset and dataloader
        dataset = TitleDataset(titles, labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_loss = 0

            for batch in tqdm(dataloader, desc="Training"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            print(f"Average loss: {avg_loss:.4f}")

        self.is_trained = True
        print("Training completed!")

    def predict(self, titles: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Predict labels for titles.

        Args:
            titles: List of title strings
            batch_size: Batch size for prediction

        Returns:
            predictions: Array of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        # Create dataset and dataloader (with dummy labels)
        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Prediction loop
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        return np.array(predictions)

    def get_feature_vectors(self, titles: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Get BERT embeddings for visualization.

        Args:
            titles: List of title strings
            batch_size: Batch size

        Returns:
            Feature vectors (CLS token embeddings)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Get hidden states from BERT base model
                outputs = self.model.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                # Use CLS token embedding from last hidden state
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.extend(cls_embeddings.cpu().numpy())

        return np.array(embeddings)

    def save_model(self, path: str):
        """
        Save the trained model.

        Args:
            path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a trained model.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_name = checkpoint['model_name']
        self.max_length = checkpoint['max_length']
        self.is_trained = True
        print(f"Model loaded from {path}")

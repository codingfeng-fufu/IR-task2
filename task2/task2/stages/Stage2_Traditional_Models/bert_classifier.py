"""
bert_classifier.py
==================
BERT Classifier Implementation (Enhanced with Flexible Save/Load)
Use pre-trained BERT model for sequence classification
"""

import numpy as np
import torch
import os
from typing import List, Dict
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW
from tqdm import tqdm


class TitleDataset(Dataset):
    """PyTorch dataset for BERT model"""
    
    def __init__(self, titles: List[str], labels: List[int], tokenizer, max_length=64):
        """
        Initialize dataset
        
        Args:
            titles: Title list
            labels: Label list
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        title = self.titles[idx]
        label = self.labels[idx]
        
        # Encode using BERT tokenizer
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


class BERTClassifier:
    """
    BERT-based classifier with flexible save/load functionality
    Fine-tune pre-trained BERT model
    """
    
    def __init__(self, model_name='bert-base-uncased', max_length=64, model_path='models/best_bert_model.pt'):
        """
        Initialize BERT classifier

        Args:
            model_name: Pre-trained BERT model name
            max_length: Maximum sequence length
            model_path: Path to save/load model
        """
        self.model_name = model_name
        self.max_length = max_length
        self.model_path = model_path
        
        # Set device (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using device: {self.device} ({torch.cuda.get_device_name(0)})")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading BERT tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)

        # Load classification model
        print("Loading BERT classification model...")
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # Binary classification
            output_attentions=False,
            output_hidden_states=False,
            local_files_only=True
        ).to(self.device)

        # Model for feature extraction (without classification head)
        self.feature_model = BertModel.from_pretrained(model_name, local_files_only=True).to(self.device)
        
        # Training status
        self.is_trained = False
        
    def train(self, titles: List[str], labels: List[int],
              epochs=5, batch_size=32, learning_rate=2e-5, warmup_steps=500,
              save_model=True):
        """
        Fine-tune BERT model (Optimized version v2)

        Args:
            titles: Training title list
            labels: Training label list
            epochs: Number of training epochs (increased to 5)
            batch_size: Batch size (keep 32 for better generalization)
            learning_rate: Learning rate
            warmup_steps: Learning rate warmup steps
            save_model: Whether to save the model after training
        """
        print("\n" + "="*60)
        print("Training BERT Classifier (Optimized v2)")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Training samples: {len(titles)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Warmup steps: {warmup_steps}")

        # Create dataset and dataloader
        dataset = TitleDataset(titles, labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Setup learning rate scheduler (with warmup)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(f"Total training steps: {total_steps}")

        # Record best loss (for monitoring, but no early stopping)
        best_loss = float('inf')
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # Use tqdm for progress bar
            progress_bar = tqdm(dataloader, desc=f"Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate

                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)

                # Update progress bar (show current learning rate)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_predictions:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # Calculate average loss and accuracy
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions

            print(f"\nEpoch {epoch + 1} complete:")
            print(f"  - Average loss: {avg_loss:.4f}")
            print(f"  - Training accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  - Current learning rate: {scheduler.get_last_lr()[0]:.2e}")

            # Record best loss (no early stopping triggered)
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  ✓ Loss improved (best loss: {best_loss:.4f})")
            else:
                print(f"  - Loss did not improve (best loss: {best_loss:.4f})")
        
        self.is_trained = True
        print("\n✓ BERT training complete!")
        
        # Save model
        if save_model:
            self.save_model()
    
    def save_model(self):
        """Save BERT model to file"""
        # Create directory
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model state, tokenizer config, and other settings
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'is_trained': self.is_trained
        }
        
        torch.save(checkpoint, self.model_path)
        print(f"\n✓ BERT model saved to: {self.model_path}")
    
    def load_model(self):
        """
        Load BERT model from file (FLEXIBLE - supports multiple formats)
        
        Supports:
        1. Checkpoint dict with 'model_state_dict' key
        2. Direct state_dict
        3. Checkpoint dict with 'state_dict' key
        4. Entire model object
        """
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model file does not exist: {self.model_path}")
            return False
        
        print(f"Loading BERT model: {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Determine format and load accordingly
            if isinstance(checkpoint, dict):
                # It's a dictionary - check for different key formats
                if 'model_state_dict' in checkpoint:
                    # Format 1: Our standard format
                    print("  → Detected format: checkpoint with 'model_state_dict'")
                    state_dict = checkpoint['model_state_dict']
                    self.model_name = checkpoint.get('model_name', self.model_name)
                    self.max_length = checkpoint.get('max_length', self.max_length)
                    self.is_trained = checkpoint.get('is_trained', True)
                elif 'state_dict' in checkpoint:
                    # Format 2: Checkpoint with 'state_dict' key
                    print("  → Detected format: checkpoint with 'state_dict'")
                    state_dict = checkpoint['state_dict']
                    self.is_trained = True
                else:
                    # Format 3: Direct state_dict (no wrapper)
                    print("  → Detected format: direct state_dict")
                    state_dict = checkpoint
                    self.is_trained = True
            else:
                # Format 4: Entire model object
                print("  → Detected format: entire model object")
                print("  ⚠️  Warning: Loading entire model - this is not recommended")
                self.model = checkpoint
                self.model.to(self.device)
                self.is_trained = True
                print("✓ BERT model loaded successfully!")
                return True
            
            # Load the state dict into model
            self.model.load_state_dict(state_dict, strict=False)
            
            # Move model to device
            self.model.to(self.device)
            
            print("✓ BERT model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load BERT model: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Additional diagnostic information
            print("\n=== DIAGNOSTIC INFO ===")
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                print(f"Checkpoint type: {type(checkpoint)}")
                if isinstance(checkpoint, dict):
                    print(f"Checkpoint keys: {list(checkpoint.keys())}")
                    for key in checkpoint.keys():
                        print(f"  - {key}: {type(checkpoint[key])}")
            except Exception as diag_e:
                print(f"Could not load checkpoint for diagnostics: {diag_e}")
            print("======================")
            
            return False
    
    def predict(self, titles: List[str], batch_size=16) -> np.ndarray:
        """
        Predict labels for given titles
        
        Args:
            titles: List of titles to classify
            batch_size: Batch size
            
        Returns:
            Prediction label array (0=incorrect title, 1=correct title)
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Please call train() or load_model() first")
        
        self.model.eval()
        predictions = []
        
        # Create dataloader (no labels needed)
        dummy_labels = [0] * len(titles)  # Dummy labels
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    def predict_proba(self, titles: List[str], batch_size=16) -> np.ndarray:
        """
        Predict probabilities for given titles
        
        Args:
            titles: List of titles to classify
            batch_size: Batch size
            
        Returns:
            Probability array with shape (n_samples, 2)
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Please call train() or load_model() first")
        
        self.model.eval()
        probabilities = []
        
        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing probabilities"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.append(probs)
        
        return np.vstack(probabilities)
    
    def get_feature_vectors(self, titles: List[str], batch_size=16) -> np.ndarray:
        """
        Get BERT feature vectors ([CLS] token embeddings)
        Used for visualization
        
        Args:
            titles: Title list
            batch_size: Batch size
            
        Returns:
            Feature matrix
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Please call train() or load_model() first")
        
        self.feature_model.eval()
        embeddings = []
        
        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.feature_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Use [CLS] token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)


def main():
    """
    Main function: Demonstrate BERT classifier usage
    """
    from data_loader import DataLoader, create_sample_data
    
    print("="*70)
    print(" BERT Classifier Demo (with Flexible Save/Load)")
    print("="*70)
    
    # Load data
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'positive_titles.txt',
        'negative_titles.txt',
        'test_titles.txt'
    )
    
    # If no actual files, use sample data
    if len(train_titles) == 0:
        train_titles, train_labels, test_titles, test_labels = create_sample_data()
    
    # For demonstration, use only part of data (BERT training is slow)
    print("\nNote: Using only part of data for demonstration")
    train_titles = train_titles[:200]
    train_labels = train_labels[:200]
    test_titles = test_titles[:50]
    test_labels = test_labels[:50]
    
    # Create classifier
    classifier = BERTClassifier(
        model_name='bert-base-uncased', 
        max_length=64,
        model_path='models/best_bert_model.pt'
    )
    
    # Try to load existing model
    if not classifier.load_model():
        # If no existing model, train new one
        print("Training new model...")
        classifier.train(
            train_titles, 
            train_labels,
            epochs=2,  # Fewer epochs for demonstration
            batch_size=8,
            learning_rate=2e-5,
            save_model=True
        )
    else:
        print("Using loaded model for predictions")
    
    # Make predictions
    print("\n" + "="*60)
    print("Making predictions on test set")
    print("="*60)
    
    predictions = classifier.predict(test_titles, batch_size=8)
    probabilities = classifier.predict_proba(test_titles, batch_size=8)
    
    # Display some prediction results
    print("\nPrediction results sample:")
    print(f"{'Title':<50} {'True':<10} {'Pred':<10} {'Confidence':<10}")
    print("-" * 85)
    
    for i in range(min(10, len(test_titles))):
        title = test_titles[i][:47] + "..." if len(test_titles[i]) > 50 else test_titles[i]
        true_label = "Correct" if test_labels[i] == 1 else "Incorrect"
        pred_label = "Correct" if predictions[i] == 1 else "Incorrect"
        confidence = probabilities[i][predictions[i]]
        
        print(f"{title:<50} {true_label:<10} {pred_label:<10} {confidence:.3f}")
    
    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels)
    print(f"\nTest set accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()
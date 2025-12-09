"""
bert_classifier_optimized.py
============================
Optimized BERT Classifier Implementation
Enhanced with multiple pre-trained models and advanced training strategies
"""

import numpy as np
import torch
import torch.nn as nn
import os
from typing import List, Dict, Optional
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


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
        
        # Encode using tokenizer
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


class CustomClassificationHead(nn.Module):
    """Custom classification head with dropout and multiple layers"""
    
    def __init__(self, hidden_size, num_labels=2, dropout_rate=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.out_proj = nn.Linear(hidden_size, num_labels)
    
    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class OptimizedBERTClassifier:
    """
    Optimized BERT-based classifier with multiple model options
    """
    
    # Available model options
    MODEL_OPTIONS = {
        'bert-base': 'bert-base-uncased',
        'bert-large': 'bert-large-uncased',
        'scibert': 'allenai/scibert_scivocab_uncased',  # Best for academic papers
        'roberta-base': 'roberta-base',
        'roberta-large': 'roberta-large',
        'albert-base': 'albert-base-v2',
        'albert-large': 'albert-large-v2',
        'deberta-v3': 'microsoft/deberta-v3-base',
        'deberta-v3-large': 'microsoft/deberta-v3-large',
    }
    
    def __init__(
        self, 
        model_name='scibert',  # Default to SciBERT for academic titles
        max_length=64, 
        model_path='models/best_bert_model.pt',
        use_custom_head=True,
        dropout_rate=0.3
    ):
        """
        Initialize optimized BERT classifier

        Args:
            model_name: Model name (from MODEL_OPTIONS or HuggingFace model path)
            max_length: Maximum sequence length
            model_path: Path to save/load model
            use_custom_head: Whether to use custom classification head
            dropout_rate: Dropout rate for regularization
        """
        # Resolve model name
        if model_name in self.MODEL_OPTIONS:
            self.model_name = self.MODEL_OPTIONS[model_name]
            print(f"Using pre-defined model: {model_name} -> {self.model_name}")
        else:
            self.model_name = model_name
            print(f"Using custom model: {self.model_name}")
        
        self.max_length = max_length
        self.model_path = model_path
        self.use_custom_head = use_custom_head
        self.dropout_rate = dropout_rate
        
        # Set device (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using device: {self.device} ({torch.cuda.get_device_name(0)})")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print(f"Using device: {self.device}")
        
        # Load tokenizer
        print(f"Loading tokenizer: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load classification model
        print(f"Loading classification model: {self.model_name}...")
        if use_custom_head:
            # Load base model and add custom head
            self.base_model = AutoModel.from_pretrained(self.model_name)
            hidden_size = self.base_model.config.hidden_size
            self.classifier = CustomClassificationHead(
                hidden_size, 
                num_labels=2, 
                dropout_rate=dropout_rate
            )
            self.model = nn.Sequential()  # Will be properly set up in forward pass
        else:
            # Use default classification head
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                hidden_dropout_prob=dropout_rate,
                attention_probs_dropout_prob=dropout_rate,
                output_attentions=False,
                output_hidden_states=False
            )
        
        if use_custom_head:
            self.base_model.to(self.device)
            self.classifier.to(self.device)
        else:
            self.model.to(self.device)

        # Model for feature extraction
        self.feature_model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
        # Training status
        self.is_trained = False
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def forward_with_custom_head(self, input_ids, attention_mask):
        """Forward pass with custom classification head"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits
        
    def train(
        self, 
        train_titles: List[str], 
        train_labels: List[int],
        val_titles: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs=10,
        batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        scheduler_type='linear',  # 'linear' or 'cosine'
        early_stopping_patience=3,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        save_model=True,
        freeze_layers=0  # Number of transformer layers to freeze (0 = no freezing)
    ):
        """
        Fine-tune BERT model with advanced training strategies

        Args:
            train_titles: Training title list
            train_labels: Training label list
            val_titles: Validation title list (optional)
            val_labels: Validation label list (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_ratio: Ratio of warmup steps
            scheduler_type: Learning rate scheduler type
            early_stopping_patience: Early stopping patience
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            save_model: Whether to save the model
            freeze_layers: Number of initial layers to freeze
        """
        print("\n" + "="*70)
        print("Training Optimized BERT Classifier")
        print("="*70)
        print(f"Model: {self.model_name}")
        print(f"Training samples: {len(train_titles)}")
        print(f"Validation samples: {len(val_titles) if val_titles else 0}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Scheduler: {scheduler_type}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Frozen layers: {freeze_layers}")

        # Freeze layers if requested
        if freeze_layers > 0 and not self.use_custom_head:
            print(f"\nFreezing first {freeze_layers} transformer layers...")
            if hasattr(self.model, 'bert'):
                encoder = self.model.bert.encoder
            elif hasattr(self.model, 'roberta'):
                encoder = self.model.roberta.encoder
            elif hasattr(self.model, 'albert'):
                encoder = self.model.albert.encoder
            elif hasattr(self.model, 'deberta'):
                encoder = self.model.deberta.encoder
            else:
                print("Warning: Could not identify encoder to freeze layers")
                encoder = None
            
            if encoder and hasattr(encoder, 'layer'):
                for i in range(min(freeze_layers, len(encoder.layer))):
                    for param in encoder.layer[i].parameters():
                        param.requires_grad = False
                    print(f"  - Frozen layer {i}")

        # Create dataset and dataloader
        train_dataset = TitleDataset(train_titles, train_labels, self.tokenizer, self.max_length)
        train_dataloader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Create validation dataloader if provided
        val_dataloader = None
        if val_titles and val_labels:
            val_dataset = TitleDataset(val_titles, val_labels, self.tokenizer, self.max_length)
            val_dataloader = TorchDataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )

        # Setup optimizer (only optimize parameters that require grad)
        if self.use_custom_head:
            optimizer = AdamW([
                {'params': self.base_model.parameters(), 'lr': learning_rate},
                {'params': self.classifier.parameters(), 'lr': learning_rate * 10}  # Higher LR for head
            ])
        else:
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=learning_rate
            )

        # Setup learning rate scheduler
        total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        
        if scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

        # Early stopping setup
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        
        # Training loop
        if self.use_custom_head:
            self.base_model.train()
            self.classifier.train()
        else:
            self.model.train()
        
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*70}")
            
            # Training phase
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Training")
            
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['label'].to(self.device)
                
                # Forward pass
                if self.use_custom_head:
                    logits = self.forward_with_custom_head(input_ids, attention_mask)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, batch_labels)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=batch_labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_custom_head:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.base_model.parameters()) + list(self.classifier.parameters()),
                            max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # Statistics
                total_loss += loss.item() * gradient_accumulation_steps
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'acc': f'{correct_predictions/total_predictions:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # Calculate training metrics
            avg_train_loss = total_loss / len(train_dataloader)
            train_accuracy = correct_predictions / total_predictions
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(train_accuracy)

            print(f"\nEpoch {epoch + 1} Training Results:")
            print(f"  - Average loss: {avg_train_loss:.4f}")
            print(f"  - Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
            print(f"  - Current learning rate: {scheduler.get_last_lr()[0]:.2e}")

            # Validation phase
            if val_dataloader:
                val_loss, val_acc = self._validate(val_dataloader)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
                
                print(f"\nValidation Results:")
                print(f"  - Validation loss: {val_loss:.4f}")
                print(f"  - Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience_counter = 0
                    print(f"  ✓ Validation loss improved! (best: {best_val_loss:.4f})")
                    
                    # Save best model
                    if save_model:
                        self.save_model(suffix='_best')
                else:
                    patience_counter += 1
                    print(f"  - No improvement (patience: {patience_counter}/{early_stopping_patience})")
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\n⚠️  Early stopping triggered!")
                        print(f"Best validation loss: {best_val_loss:.4f}")
                        print(f"Best validation accuracy: {best_val_acc:.4f}")
                        break
        
        self.is_trained = True
        print("\n✓ Training complete!")
        print(f"\nFinal Training History:")
        print(f"  - Best validation loss: {best_val_loss:.4f}")
        print(f"  - Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        
        # Save final model
        if save_model:
            self.save_model()
    
    def _validate(self, val_dataloader):
        """Validation during training"""
        if self.use_custom_head:
            self.base_model.eval()
            self.classifier.eval()
        else:
            self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.use_custom_head:
                    logits = self.forward_with_custom_head(input_ids, attention_mask)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        if self.use_custom_head:
            self.base_model.train()
            self.classifier.train()
        else:
            self.model.train()
        
        return total_loss / len(val_dataloader), correct / total
    
    def save_model(self, suffix=''):
        """Save model to file"""
        # Create directory
        save_path = self.model_path.replace('.pt', f'{suffix}.pt') if suffix else self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state
        checkpoint = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'is_trained': self.is_trained,
            'use_custom_head': self.use_custom_head,
            'dropout_rate': self.dropout_rate,
            'training_history': self.training_history
        }
        
        if self.use_custom_head:
            checkpoint['base_model_state_dict'] = self.base_model.state_dict()
            checkpoint['classifier_state_dict'] = self.classifier.state_dict()
        else:
            checkpoint['model_state_dict'] = self.model.state_dict()
        
        torch.save(checkpoint, save_path)
        print(f"\n✓ Model saved to: {save_path}")
    
    def load_model(self):
        """Load model from file"""
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model file does not exist: {self.model_path}")
            return False
        
        print(f"Loading model: {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                # Load metadata
                self.model_name = checkpoint.get('model_name', self.model_name)
                self.max_length = checkpoint.get('max_length', self.max_length)
                self.is_trained = checkpoint.get('is_trained', True)
                self.use_custom_head = checkpoint.get('use_custom_head', False)
                self.training_history = checkpoint.get('training_history', {})
                
                # Load model weights
                if self.use_custom_head:
                    self.base_model.load_state_dict(checkpoint['base_model_state_dict'])
                    self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
                    self.base_model.to(self.device)
                    self.classifier.to(self.device)
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.to(self.device)
                
                print("✓ Model loaded successfully!")
                return True
            else:
                print("⚠️  Unexpected checkpoint format")
                return False
                
        except Exception as e:
            print(f"⚠️  Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, titles: List[str], batch_size=16) -> np.ndarray:
        """Predict labels for given titles"""
        if not self.is_trained:
            raise ValueError("Model not trained! Please call train() or load_model() first")
        
        if self.use_custom_head:
            self.base_model.eval()
            self.classifier.eval()
        else:
            self.model.eval()
        
        predictions = []
        
        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.use_custom_head:
                    logits = self.forward_with_custom_head(input_ids, attention_mask)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    def predict_proba(self, titles: List[str], batch_size=16) -> np.ndarray:
        """Predict probabilities for given titles"""
        if not self.is_trained:
            raise ValueError("Model not trained! Please call train() or load_model() first")
        
        if self.use_custom_head:
            self.base_model.eval()
            self.classifier.eval()
        else:
            self.model.eval()
        
        probabilities = []
        
        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing probabilities"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.use_custom_head:
                    logits = self.forward_with_custom_head(input_ids, attention_mask)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.append(probs)
        
        return np.vstack(probabilities)
    
    def get_feature_vectors(self, titles: List[str], batch_size=16) -> np.ndarray:
        """Get feature vectors for visualization"""
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
                
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)


def main():
    """Main function: Demonstrate optimized BERT classifier"""
    print("="*70)
    print(" Optimized BERT Classifier Demo")
    print("="*70)
    
    # Show available models
    print("\nAvailable models:")
    for key, value in OptimizedBERTClassifier.MODEL_OPTIONS.items():
        print(f"  - {key}: {value}")
    
    # For demonstration, create sample data
    print("\nCreating sample data for demonstration...")
    
    train_titles = [
        "Deep Learning for Computer Vision Applications",
        "Natural Language Processing with Transformers",
        "Introduction to Machine Learning Algorithms",
    ] * 30
    
    train_labels = [1] * 90
    
    neg_titles = [
        "Call for Papers......41 Fragments......42",
        "Special Issue on......Page 1-25......Vol 3",
        "Conference Proceedings......Abstract......",
    ] * 30
    
    train_titles.extend(neg_titles)
    train_labels.extend([0] * 90)
    
    # Create classifier with SciBERT (best for academic titles)
    classifier = OptimizedBERTClassifier(
        model_name='scibert',  # or try 'roberta-base', 'deberta-v3', etc.
        max_length=64,
        model_path='models/optimized_bert_model.pt',
        use_custom_head=True,
        dropout_rate=0.3
    )
    
    # Train with advanced settings
    print("\nTraining model with optimized settings...")
    classifier.train(
        train_titles, 
        train_labels,
        val_titles=train_titles[:40],  # Use subset for validation
        val_labels=train_labels[:40],
        epochs=5,
        batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        scheduler_type='cosine',
        early_stopping_patience=3,
        gradient_accumulation_steps=2,
        save_model=True,
        freeze_layers=0
    )
    
    # Make predictions
    test_titles = train_titles[:20]
    predictions = classifier.predict(test_titles)
    probabilities = classifier.predict_proba(test_titles)
    
    print("\n" + "="*70)
    print("Sample Predictions")
    print("="*70)
    for i in range(min(10, len(test_titles))):
        print(f"\nTitle: {test_titles[i][:60]}...")
        print(f"Predicted: {'Correct' if predictions[i] == 1 else 'Incorrect'}")
        print(f"Confidence: {probabilities[i][predictions[i]]:.3f}")


if __name__ == "__main__":
    main()

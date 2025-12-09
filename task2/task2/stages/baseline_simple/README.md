# Simple Baseline Implementation (朴素实现)

This is the **most basic implementation** of the academic title classification system without any feature engineering or optimization. It serves as a clean baseline to compare against the optimized version in `task2/task2/`.

## Overview

This implementation includes three classifiers with **no optimization**:

1. **Naive Bayes**: Simple TF-IDF features (unigrams + bigrams only)
2. **Word2Vec + SVM**: Basic word embeddings (100d) + default RBF kernel
3. **BERT**: Pre-trained `bert-base-uncased`, 3 epochs fine-tuning

## Quick Start

### Prerequisites

```bash
pip install torch transformers scikit-learn gensim pandas openpyxl matplotlib tqdm
```

### Run the Pipeline

```bash
cd /home/u2023312337/task2/task2/baseline_simple
python main.py
```

This will:
1. Load training data (positive.txt, negative.txt)
2. Load test data (testSet-1000.xlsx)
3. Train all three classifiers
4. Evaluate on test set
5. Generate visualizations

### Expected Runtime

- Naive Bayes: ~2-3 minutes
- Word2Vec + SVM: ~5-10 minutes
- BERT: ~30-60 minutes (depending on GPU)
- **Total: ~40-75 minutes**

## File Structure

```
baseline_simple/
├── main.py                 # Main pipeline
├── data_loader.py          # Simple data loading
├── naive_bayes.py          # Naive Bayes classifier
├── word2vec_svm.py         # Word2Vec + SVM classifier
├── bert_classifier.py      # BERT classifier
├── evaluator.py            # Evaluation metrics
├── visualizer.py           # Visualization utilities
├── models/                 # Saved models (created after training)
├── output/                 # Results and plots (created after training)
└── README.md               # This file
```

## What Makes This a "Simple Baseline"?

### Naive Bayes
- **No feature engineering**: Only basic TF-IDF
- **No statistical features**: No length, punctuation, or pattern features
- **No character n-grams**: Only word-level features
- **No optimization**: Standard MultinomialNB with default alpha=1.0

### Word2Vec + SVM
- **No feature engineering**: Only averaged word embeddings
- **No additional features**: No statistical or structural features
- **Simple averaging**: Just mean of word vectors, no weighting
- **Default SVM**: RBF kernel with default C and gamma

### BERT
- **No optimization**: Just 3 epochs of fine-tuning
- **No advanced techniques**: No focal loss, adversarial training, or early stopping
- **Standard model**: `bert-base-uncased`, not domain models like SciBERT
- **Short sequences**: max_length=64 tokens

## Expected Results

Based on the simple baseline approach:

- **Naive Bayes**: ~70-75% accuracy
- **Word2Vec + SVM**: ~75-80% accuracy
- **BERT**: ~85-88% accuracy

These results serve as the baseline for comparison with optimized versions.

## Output Files

### Models
- `models/naive_bayes.pkl` - Naive Bayes model
- `models/word2vec_svm_w2v.model` - Word2Vec model
- `models/word2vec_svm_svm.pkl` - SVM model
- `models/bert_model.pt` - BERT model

### Visualizations
- `output/model_comparison.png` - Bar chart comparing all models
- `output/confusion_matrices.png` - Confusion matrices for all models
- `output/tsne_Naive_Bayes.png` - t-SNE visualization
- `output/tsne_Word2Vec_SVM.png` - t-SNE visualization
- `output/tsne_BERT.png` - t-SNE visualization

## Comparison with Optimized Version

The code in `task2/task2/` represents the **optimized final version** with:

**Feature Engineering**:
- Character n-grams (3-5) for Naive Bayes → captures format patterns
- 22 statistical features → length, punctuation, special patterns
- Multi-level TF-IDF → word + char level features

**Algorithm Optimization**:
- ComplementNB instead of MultinomialNB → **79.20% vs 73.46%** (+5.74%)
- Hyperparameter tuning (alpha=0.5)
- Advanced BERT training (SciBERT, DeBERTa, Focal Loss) → **89-91%**

**Advanced Methods**:
- LLM experiments (GPT, Claude) → **92%+** accuracy
- Ensemble techniques
- Comprehensive evaluation framework

| Model | This (Baseline) | Optimized (task2/task2/) | Improvement |
|-------|-----------------|--------------------------|-------------|
| Naive Bayes | ~73% | **79.20%** | +6.2% |
| Word2Vec+SVM | ~80% | **82.99%** | +3.0% |
| BERT | ~87% | **89-91%** | +2-4% |

## Troubleshooting

### Out of Memory (BERT)
```python
# Edit main.py, reduce batch size:
bert_classifier.train(train_titles, train_labels, epochs=3, batch_size=8)  # was 16
```

### No GPU Available
BERT will automatically use CPU if CUDA is not available, but training will be slower (2-3 hours).

### Missing Dependencies
```bash
pip install torch transformers scikit-learn gensim pandas openpyxl matplotlib tqdm
```

## Relationship to task2/task2/

```
baseline_simple/          ← Simple baseline (THIS)
    ↓ evolution
task2/task2/              ← Optimized version with:
├── stages/               ← Organized by functional modules
│   ├── Stage1_Foundation
│   ├── Stage2_Traditional_Models
│   ├── Stage3_NaiveBayes_Optimization
│   ├── Stage4_BERT_Optimization
│   └── Stage5_LLM_Framework
└── [extensive documentation and experiments]
```

**Note**: The `task2/task2/stages/` directory organizes the **optimized code** by functional modules, not implementation stages. This `baseline_simple/` directory is the **starting point** before all those optimizations.

## Testing

Verify the implementation works:

```bash
cd /home/u2023312337/task2/baseline_simple
python test_implementation.py
```

All tests should pass ✓

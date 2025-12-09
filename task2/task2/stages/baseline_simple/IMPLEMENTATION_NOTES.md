# Stage 1 Implementation Summary

## Overview

Created a **simple baseline implementation** (Stage 1) of the academic title classification system as requested. This implementation has **no feature engineering or optimization** - it's the most basic version that addresses the project requirements.

## What Was Implemented

### Three Classifiers

1. **Naive Bayes** (`naive_bayes.py`)
   - Simple TF-IDF vectorization (5000 features)
   - Unigrams + bigrams only
   - Standard MultinomialNB with default parameters
   - No statistical features, no character n-grams

2. **Word2Vec + SVM** (`word2vec_svm.py`)
   - Basic Word2Vec embeddings (100 dimensions)
   - Simple averaging of word vectors
   - RBF kernel SVM with default parameters
   - No additional feature engineering

3. **BERT** (`bert_classifier.py`)
   - Pre-trained `bert-base-uncased`
   - Simple fine-tuning (3 epochs)
   - Max sequence length: 64 tokens
   - No optimization techniques (focal loss, adversarial training, etc.)

### Supporting Modules

- **Data Loader** (`data_loader.py`): Simple loading of positive/negative/test files
- **Evaluator** (`evaluator.py`): Standard metrics (accuracy, precision, recall, F1)
- **Visualizer** (`visualizer.py`): Model comparison charts, confusion matrices, t-SNE plots
- **Main Pipeline** (`main.py`): Orchestrates the entire workflow

## Key Design Decisions

### What Was Intentionally Kept Simple

1. **No Feature Engineering**
   - Naive Bayes: Only TF-IDF, no statistical features
   - Word2Vec: Only averaged embeddings, no weighting or additional features
   - BERT: Standard fine-tuning without optimization

2. **No Hyperparameter Tuning**
   - All models use default or simple parameters
   - No grid search or optimization

3. **No Advanced Techniques**
   - No ensemble methods
   - No data augmentation
   - No advanced loss functions
   - No early stopping or learning rate scheduling

4. **Minimal Preprocessing**
   - Just basic lowercasing and tokenization
   - No special handling of punctuation, numbers, etc.

### Why This Approach

This establishes a **clean baseline** that:
- Clearly shows what the basic implementation achieves
- Makes it easy to measure improvement from optimizations
- Follows the project requirement for a "朴素的实现" (simple/naive implementation)
- Serves as Stage 1 before adding optimizations in subsequent stages

## Usage

```bash
cd /home/u2023312337/task2/stage1
python main.py
```

## Expected Results

Based on the simple baseline approach:
- **Naive Bayes**: ~70-75% accuracy
- **Word2Vec + SVM**: ~75-80% accuracy
- **BERT**: ~85-88% accuracy

## Comparison with task2/task2/ (Optimized Version)

| Aspect | Stage 1 (This) | task2/task2/ (Optimized) |
|--------|----------------|---------------------------|
| Naive Bayes | TF-IDF only (~73%) | Multi-level features + ComplementNB (79.20%) |
| Features | None | 22 statistical features + char n-grams |
| BERT | 3 epochs, basic | Multiple experiments, SciBERT, DeBERTa |
| Loss | Standard | Focal Loss, Weighted CE |
| Documentation | Simple README | Extensive guides, optimization summaries |

## Next Steps (Evolution Path)

This Stage 1 implementation will be the foundation. Future stages will add:

**Stage 2**: Feature Engineering
- Add character-level n-grams
- Add statistical features (length, punctuation, patterns)
- Feature combination strategies

**Stage 3**: Algorithm Optimization
- ComplementNB vs MultinomialNB
- Hyperparameter tuning
- Advanced SVM kernels and parameters

**Stage 4**: BERT Optimization
- Domain-specific models (SciBERT)
- Advanced training (Focal Loss, adversarial)
- Early stopping and learning rate scheduling

**Stage 5**: Advanced Methods
- Ensemble techniques
- LLM experiments (GPT/Claude)
- Model stacking

## Files Created

```
/home/u2023312337/task2/stage1/
├── main.py                  # Main pipeline
├── data_loader.py           # Data loading
├── naive_bayes.py           # Naive Bayes classifier
├── word2vec_svm.py          # Word2Vec + SVM classifier
├── bert_classifier.py       # BERT classifier
├── evaluator.py             # Evaluation metrics
├── visualizer.py            # Visualization
├── README.md                # User guide
└── IMPLEMENTATION_NOTES.md  # This file
```

## Notes

- All code is well-commented and follows Python best practices
- Models can be saved and loaded for later use
- Visualizations include t-SNE for all three methods
- Error handling included for missing data files
- GPU automatically detected for BERT training

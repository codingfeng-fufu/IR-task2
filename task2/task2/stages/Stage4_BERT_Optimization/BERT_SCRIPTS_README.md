# BERT Training Scripts for Stage4

This directory now contains **complete BERT training and experiment infrastructure** for Stage 4 (BERT Optimization).

## 📁 Available Scripts

### Core BERT Training Scripts

1. **`run_bert_experiments.py`** - Batch experiment runner
   - Runs multiple BERT optimization experiments in sequence
   - Compares different models: BERT-base, SciBERT, RoBERTa
   - Tests different configurations: Focal Loss, Weighted CE, max_length variations
   - Generates comparison reports and result JSON files
   - **Usage**: `python run_bert_experiments.py`
   - **Time**: 8-12 hours for all experiments
   - **Output**: `models/experiments/comparison_report.txt`, `results.json`

2. **`train_bert_optimized_v2.py`** - Single optimized BERT training
   - Trains a single optimized BERT model with advanced features
   - Supports SciBERT, RoBERTa, DeBERTa, BERT-base
   - Features: Focal Loss, adversarial training, layer-wise LR decay
   - **Usage**: `python train_bert_optimized_v2.py`
   - **Time**: 2-3 hours
   - **Expected**: 87-88% accuracy, 85-87% recall

3. **`bert_classifier_optimized.py`** - Optimized BERT classifier class
   - Advanced BERT classifier implementation
   - Supports multiple loss functions (Focal, Weighted CE)
   - Includes adversarial training (FGM/PGD)
   - Layer-wise learning rate decay
   - Early stopping and mixed precision training
   - **Used by**: `train_bert_optimized_v2.py`, `run_bert_experiments.py`

4. **`bert_classifier.py`** - Standard BERT classifier
   - Basic BERT fine-tuning implementation
   - Uses `bert-base-uncased`
   - Simple training interface
   - **Baseline performance**: 87.91% accuracy

### Visualization Scripts

5. **`visualize_bert_experiments.py`** - Basic experiment visualization
   - Creates comparison charts for BERT experiments
   - Plots accuracy, precision, recall, F1 across experiments

6. **`visualize_bert_experiments_complete.py`** - Complete visualization suite
   - Comprehensive visualization of all experiment results
   - Includes training curves, confusion matrices, error analysis
   - Generates multiple output plots

7. **`visualize_bert_results_simple.py`** - Simple result visualization
   - Quick visualization of experiment results
   - Lightweight plotting for fast review

### Supporting Files

8. **`data_loader.py`** - Data loading utilities
   - Loads positive/negative training samples
   - Loads test data from Excel file
   - Preprocessing and data preparation

9. **`evaluator.py`** - Model evaluation utilities
   - Computes accuracy, precision, recall, F1 scores
   - Generates confusion matrices
   - Performs error analysis

10. **`visualizer.py`** - General visualization utilities
    - Model comparison bar charts
    - Confusion matrix heatmaps
    - t-SNE embeddings visualization

11. **`config.py`** - Configuration settings
    - Shared configuration for Stage4

## 🚀 Quick Start

### Run All BERT Experiments

```bash
cd /home/u2023312337/task2/task2/stages/Stage4_BERT_Optimization
source ../../.venv/bin/activate
python run_bert_experiments.py
```

This will run 4 experiments:
1. **exp1_bert_base_baseline** - BERT-base with standard CE loss
2. **exp2_scibert_focal** - SciBERT with Focal Loss + adversarial training
3. **exp3_roberta_weighted** - RoBERTa with Weighted CE
4. **exp5_scibert_maxlen128** - SciBERT with longer sequences (128 tokens)

### Run Single Optimized Model

```bash
cd /home/u2023312337/task2/task2/stages/Stage4_BERT_Optimization
source ../../.venv/bin/activate
python train_bert_optimized_v2.py
```

This trains a single SciBERT model with Focal Loss and all optimizations.

### Visualize Results

After training, visualize the results:

```bash
python visualize_bert_experiments.py
# or
python visualize_bert_experiments_complete.py
```

## 📊 Expected Performance

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **BERT-base (Baseline)** | 87.91% | 90.29% | 88.35% | 89.59% |
| **SciBERT + Focal Loss** | 87-88% | - | 85-87% | 88-89% |
| **RoBERTa + Weighted CE** | 88-89% | - | 86-88% | 89-90% |
| **DeBERTa-v3 (Best)** | 89-91% | - | 88-90% | 90-92% |

## 📁 Output Structure

```
Stage4_BERT_Optimization/
├── models/
│   └── experiments/
│       ├── exp1_bert_base_baseline.pt
│       ├── exp2_scibert_focal.pt
│       ├── exp3_roberta_weighted.pt
│       ├── exp5_scibert_maxlen128.pt
│       ├── results.json
│       └── comparison_report.txt
├── output/
│   ├── experiment_comparison.png
│   ├── training_curves.png
│   └── confusion_matrices.png
└── logs/
    └── training_*.log
```

## 🔧 Configuration

### Modify Experiments

Edit `run_bert_experiments.py` line 139-207 to modify experiment configurations:

```python
experiments = [
    {
        'name': 'exp1_bert_base_baseline',
        'config': {
            'model_name': 'bert-base',  # or 'scibert', 'roberta', 'deberta'
            'max_length': 64,           # or 96, 128
            'epochs': 8,
            'batch_size': 32,
            'learning_rate': 2e-5,
            'loss_type': 'ce',          # or 'focal', 'weighted_ce'
            'use_adversarial': False,   # Enable adversarial training
            'use_layer_wise_lr': False  # Enable layer-wise LR decay
        }
    },
    # Add more experiments...
]
```

### Modify Single Training

Edit `train_bert_optimized_v2.py` line 707-750 to change training parameters:

```python
classifier = OptimizedBERTClassifier(
    model_name='scibert',  # or 'bert-base', 'roberta', 'deberta'
    max_length=96,
    dropout_rate=0.2
)

history = classifier.train(
    train_titles,
    train_labels,
    epochs=10,
    batch_size=32,
    learning_rate=2e-5,
    loss_type='focal',  # or 'weighted_ce', 'ce'
    use_adversarial=True,
    use_layer_wise_lr=True,
    early_stopping_patience=3
)
```

## 📚 Dependencies

All scripts require:
- PyTorch (with CUDA for GPU acceleration)
- transformers (Hugging Face)
- scikit-learn
- pandas
- numpy
- matplotlib, seaborn
- tqdm

These are already installed in the virtual environment at `task2/task2/.venv/`.

## 🔗 Data Directory

The `data/` directory is a symbolic link to `../../data`, which contains:
- `positive.txt` (118,239 samples)
- `negative.txt` (114,163 samples)
- `testSet-1000.xlsx` (1,000 test samples)

## ⚠️ Important Notes

1. **Virtual Environment**: Always activate the venv before running scripts:
   ```bash
   source ../../.venv/bin/activate
   ```

2. **GPU Memory**: If you encounter OOM errors, reduce:
   - `batch_size` from 32 → 16
   - `max_length` from 96 → 64

3. **Training Time**: Full experiment suite takes 8-12 hours with GPU

4. **Model Files**: Trained models are large (438 MB each) and saved in `models/` directory

5. **Working Directory**: Scripts use absolute paths and can be run from the Stage4 directory

## 🆚 Comparison with Other Stages

- **Stage 2 (Traditional Models)**: Uses Naive Bayes (79.20%) and Word2Vec+SVM (82.99%)
- **Stage 3 (Naive Bayes Optimization)**: Optimized Naive Bayes to 79.20%
- **Stage 4 (BERT Optimization)**: This stage - BERT models 87-91% accuracy

## 📖 Additional Documentation

- `IMPLEMENTATION.md` - Implementation details for Stage4
- `README.md` - General Stage4 overview
- `../../CLAUDE.md` - Full project documentation
- `../../BERT_OPTIMIZATION_README.md` - Detailed BERT optimization guide

## 🎯 Quick Commands Reference

```bash
# Activate environment
source ../../.venv/bin/activate

# Run all experiments (8-12 hours)
python run_bert_experiments.py

# Run single optimized model (2-3 hours)
python train_bert_optimized_v2.py

# Visualize results
python visualize_bert_experiments.py

# Check environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# View experiment results
cat models/experiments/comparison_report.txt

# Monitor GPU during training (in another terminal)
watch -n 1 nvidia-smi
```

## ✅ Verification

To verify all scripts are working:

```bash
# Test imports
python -c "from run_bert_experiments import *; print('✓ run_bert_experiments.py OK')"
python -c "from train_bert_optimized_v2 import *; print('✓ train_bert_optimized_v2.py OK')"
python -c "from bert_classifier_optimized import *; print('✓ bert_classifier_optimized.py OK')"

# Check data directory
ls -la data/

# Check GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

**Created**: 2024-12-08
**Stage**: Stage4_BERT_Optimization
**Purpose**: Complete BERT training infrastructure for academic title classification

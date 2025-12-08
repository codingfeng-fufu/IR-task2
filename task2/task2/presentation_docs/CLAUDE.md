# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Academic Title Classification System** for identifying incorrectly extracted paper titles from the CiteSeer database. The system implements three machine learning approaches to classify whether a given title is correctly extracted (positive) or incorrectly extracted (negative):

1. **Naive Bayes (Optimized)** - Enhanced version with word-level + char-level TF-IDF features, statistical features, and ComplementNB - **79.20% accuracy**
2. **Word2Vec + SVM** - Word embeddings averaged with Support Vector Machine classification - **82.99% accuracy**
3. **BERT** - Fine-tuned transformer model for sequence classification - **87.91% accuracy**
4. **LLMs (Stage 5)** - In-context learning with large language models (DeepSeek, GLM-4, Qwen, etc.) - **~84-86% accuracy**

The project is written primarily in **Chinese comments** with English code and documentation.

**Recent Optimizations**:
- Naive Bayes: improved from 73.46% to 79.20% accuracy (+5.74 points)
- BERT: advanced optimization experiments with SciBERT, DeBERTa, Focal Loss, and adversarial training
- LLM Framework (Stage 5): Flexible configuration-driven system for testing multiple commercial LLMs

## Project Evolution (5 Stages)

The project evolved through 5 distinct development stages, each organized in the `stages/` directory:

1. **Stage 1 - Foundation** (Oct 2024): Data loading, evaluation framework, visualization
2. **Stage 2 - Traditional Models** (Nov 15, 2024): Initial NB, Word2Vec+SVM, BERT implementations
3. **Stage 3 - NB Optimization** (Nov 25, 2024): Enhanced Naive Bayes with multi-level features (+5.74% accuracy)
4. **Stage 4 - BERT Optimization** (Nov 16-28, 2024): SciBERT, DeBERTa, Focal Loss, adversarial training
5. **Stage 5 - LLM Framework** (Dec 1-2, 2024): In-context learning experiments with commercial LLMs

See `stages/README.md` for detailed organization, or `VERSION_EVOLUTION.md` for complete technical evolution.

## Key Commands

### Running the Full Pipeline

```bash
# Activate virtual environment first
cd /home/u2023312337/task2/task2
source .venv/bin/activate

# Run complete pipeline (all three models)
python main_pipeline.py
```

The main pipeline will:
- Load data from `data/positive.txt`, `data/negative.txt`, and `data/testSet-1000.xlsx`
- Train all three classifiers
- Evaluate on test set
- Generate visualizations in `output/` directory
- Save trained models to `models/` directory

**Configuration options** in `main_pipeline.py` (around line 287-298):
```python
USE_SAMPLE_DATA = False      # Use sample data if files missing
MAX_TRAIN_SAMPLES = None     # Limit training set (None = all)
TRAIN_ONLY_BERT = False      # Skip NB and Word2Vec
BERT_EPOCHS = 5              # BERT training epochs
OUTPUT_DIR = 'output'        # Output directory
```

### LLM In-Context Learning Experiments

**Stage 5** of the project includes Large Language Model experiments using in-context learning.

```bash
# Setup configuration (first time)
cp llm_config_template.json llm_config.json
vim llm_config.json  # Fill in API keys

# Run single model experiment
python run_llm_experiment.py --model deepseek
python run_llm_experiment.py --model glm-4.6
python run_llm_experiment.py --model qwen3

# Run all enabled models
python run_llm_experiment.py --all

# Quick test with smaller sample
python run_llm_experiment.py --model deepseek --sample 100

# Custom output directory
python run_llm_experiment.py --model deepseek --output my_results/

# Estimate costs before running
python calculate_llm_cost.py
```

**Supported models** (configured in `llm_config.json`):
- DeepSeek (国内，性价比高)
- Qwen/通义千问 (阿里云)
- GLM-4 (智谱AI)
- Kimi (Moonshot，长文本)
- GPT-3.5/GPT-4 (OpenAI)
- Claude-3 (Anthropic)

See `LLM_EXPERIMENT_GUIDE.md` for detailed setup and usage instructions.

### Training Individual Models

```bash
# Train only BERT (fast mode)
# Edit main_pipeline.py: set TRAIN_ONLY_BERT = True
python main_pipeline.py

# Run BERT optimization experiments (SciBERT, DeBERTa, Focal Loss, etc.)
python run_bert_experiments.py
# Time: 8-12 hours for 5 experiments
# Output: models/experiments/comparison_report.txt

# Single optimized BERT training (SciBERT + Focal Loss)
python train_bert_optimized_v2.py
# Time: 2-3 hours
# Expected: 87-88% accuracy, 85-87% recall

# Quick BERT test (3 epochs only)
./run_quick.sh
# Choose option 3
```

### Evaluating Saved Models

```bash
# Evaluate previously trained models (without retraining)
python evaluate_saved.py

# Compare original vs optimized Naive Bayes
python test_optimized_nb.py
```

### Environment Check

```bash
# Verify environment setup (checks packages, CUDA, data files)
python check_environment.py
```

## Architecture

### Data Flow

1. **Data Loading** (`data_loader.py`)
   - Loads positive/negative training samples from text files (232,402 total samples)
   - Loads test data from Excel file (`testSet-1000.xlsx`, 1000 samples)
   - Preprocessing: lowercase conversion, special character removal, whitespace normalization

2. **Model Training** (individual classifier files)
   - Each classifier implements standard interface: `train()`, `predict()`, `save_model()`, `load_model()`, `get_feature_vectors()`
   - Models are saved to `models/` directory with different formats:
     - Naive Bayes: `.pkl` (joblib) - 11-44 MB
     - Word2Vec+SVM: `_w2v.model` (gensim) + `_svm.pkl` (joblib) - 139 MB total
     - BERT: `.pt` (PyTorch state dict) - 438 MB

3. **Evaluation** (`evaluator.py`)
   - Computes accuracy, precision, recall, F1 (binary, macro, micro)
   - Generates confusion matrices
   - Performs error analysis (FP/FN breakdown with example samples)
   - Compares multiple models side-by-side

4. **Visualization** (`visualizer.py`)
   - Model comparison bar charts
   - Confusion matrix heatmaps
   - t-SNE embeddings visualization
   - All plots use English labels (no Chinese fonts to avoid rendering issues)

### Model Implementations

#### Naive Bayes Optimized (`naive_bayes_classifier_optimized.py`)

**Multi-level features** (total: 15,022 dimensions):
- Word-level TF-IDF (10,000 features, 1-3 grams, max_df=0.95)
- Character-level TF-IDF (5,000 features, 3-5 grams)
- 22 statistical features:
  - Length features (3): word count, char count, avg word length
  - Punctuation features (5): dots, commas, colons, semicolons, digits
  - Capitalization features (2): uppercase count, uppercase ratio
  - Vocabulary diversity (1): unique word ratio
  - Special pattern detection (9): "abstract", "reference", "page", "vol", "copyright", year patterns, page numbers
  - Format anomaly detection (2): consecutive dots, dot count

**Algorithm**: `ComplementNB` with alpha=0.5 (better for text classification than MultinomialNB)

**Performance**: 79.20% accuracy (vs 73.46% original) - see `OPTIMIZATION_SUMMARY.md` for details

**Original version** available in `naive_bayes_classifier.py` for comparison.

#### Word2Vec + SVM (`word2vec_svm_classifier.py`)

- Trains Word2Vec embeddings (100d, window=5, min_count=2)
- Averages word vectors for sentence representation
- Optional statistical feature engineering (`add_features=True`)
- Supports both LinearSVC (fast) and RBF kernel SVC (accurate)
- Uses `CalibratedClassifierCV` for probability outputs
- **Performance**: 82.99% accuracy

#### BERT (`bert_classifier.py`, `bert_classifier_optimized.py`)

**Standard version** (`bert_classifier.py`):
- Fine-tunes `bert-base-uncased` for binary classification
- Max sequence length: 64 tokens
- Training: AdamW optimizer, linear warmup schedule
- GPU-accelerated if CUDA available
- **Performance**: 87.91% accuracy (baseline)

**Optimized version** (`bert_classifier_optimized.py`):
- Supports multiple pre-trained models: SciBERT, RoBERTa, DeBERTa
- Advanced loss functions: Focal Loss, Weighted Cross-Entropy
- Adversarial training (FGM/PGD)
- Early stopping, gradient accumulation, mixed precision
- Configurable via `train_bert_optimized_v2.py`

**Batch experiment runner** (`run_bert_experiments.py`):
- Runs 5 experiments: BERT-base baseline, SciBERT+Focal, RoBERTa+Weighted CE, DeBERTa-v3, SciBERT+Max128
- Generates comparison report in `models/experiments/`
- Expected best performance: 89-91% accuracy with DeBERTa-v3

### Directory Structure

```
task2/task2/
├── data/                          # Training and test data
│   ├── positive.txt              # Correctly extracted titles (118,239 samples)
│   ├── negative.txt              # Incorrectly extracted titles (114,163 samples)
│   └── testSet-1000.xlsx         # Test dataset (1000 samples, columns: "title given by manchine", "Y/N")
│
├── models/                        # Saved model weights
│   ├── naive_bayes_optimized_model.pkl     # 44 MB
│   ├── naive_bayes_original_model.pkl      # 11 MB
│   ├── word2vec_svm_model_w2v.model        # 25 MB
│   ├── word2vec_svm_model_svm.pkl          # 114 MB
│   ├── best_bert_model.pt                  # 438 MB
│   └── experiments/                         # BERT optimization experiments
│       ├── comparison_report.txt
│       ├── results.json
│       └── *.pt                             # Multiple experimental models
│
├── output/                        # Generated visualizations and results
│   ├── model_comparison.png       # Performance comparison bar chart
│   ├── confusion_matrices.png     # Confusion matrix heatmap
│   ├── tsne_Naive_Bayes.png      # t-SNE visualization
│   ├── tsne_Word2Vec_SVM.png
│   ├── tsne_BERT.png
│   ├── evaluation_results.txt     # Detailed metrics
│   ├── predictions.json           # Model predictions
│   └── llm_experiments/           # LLM experiment results (Stage 5)
│       ├── *_report.txt           # Text reports
│       ├── *.json                 # Detailed results
│       └── checkpoints/           # Checkpoint files
│
├── stages/                        # Project files organized by development stage
│   ├── Stage1_Foundation/         # Data loader, evaluator, visualizer
│   ├── Stage2_Traditional_Models/ # Initial NB, Word2Vec, BERT implementations
│   ├── Stage3_NaiveBayes_Optimization/  # NB optimization files
│   ├── Stage4_BERT_Optimization/  # BERT optimization experiments
│   ├── Stage5_LLM_Framework/      # LLM in-context learning experiments
│   ├── Main_Scripts/              # Main pipeline and evaluation scripts
│   └── Utils/                     # Utility scripts
│
├── logs/                          # Training logs
│
├── .venv/                         # Python virtual environment
│
├── main_pipeline.py               # Main entry point - orchestrates entire workflow
├── data_loader.py                 # Data loading utilities
│
├── naive_bayes_classifier.py      # Original Naive Bayes (73.46% accuracy)
├── naive_bayes_classifier_optimized.py  # Optimized version (79.20% accuracy)
│
├── word2vec_svm_classifier.py     # Word2Vec + SVM classifier
│
├── bert_classifier.py             # Standard BERT classifier (87.91% baseline)
├── bert_classifier_optimized.py   # Optimized BERT with advanced features
├── train_bert_optimized_v2.py     # Single optimized BERT training script
├── run_bert_experiments.py        # Batch experiment runner for BERT
│
├── llm_in_context_classifier.py   # LLM in-context learning classifier (Stage 5)
├── llm_multi_experiment.py        # Multi-model LLM comparison (Stage 5)
├── run_llm_experiment.py          # Main LLM experiment script (Stage 5)
├── calculate_llm_cost.py          # LLM cost estimation tool
├── llm_config_template.json       # LLM configuration template
│
├── evaluator.py                   # Model evaluation utilities
├── visualizer.py                  # Result visualization
│
├── evaluate_saved.py              # Evaluate saved models without retraining
├── evaluate_all_stages.py         # Evaluate all stage models comprehensively
├── test_optimized_nb.py           # Compare original vs optimized Naive Bayes
├── check_environment.py           # Environment validation
├── predownload_models.py          # Pre-download Hugging Face models
├── run_quick.sh                   # Convenience script for quick experiments
│
├── CLAUDE.md                      # This file
├── VERSION_EVOLUTION.md           # Complete version evolution history
├── EVOLUTION_ROADMAP.md           # Quick reference roadmap
├── LLM_EXPERIMENT_GUIDE.md        # LLM experiment setup and usage guide
├── LLM_COST_SUMMARY.md            # LLM cost analysis and estimates
└── OPTIMIZATION_SUMMARY.md        # Naive Bayes optimization details
```

## Dependencies

The project uses a Python virtual environment in `.venv/`. Key packages:

- **PyTorch** - BERT model training
- **transformers** (Hugging Face) - Pre-trained BERT, SciBERT, RoBERTa, DeBERTa models
- **scikit-learn** - Naive Bayes, SVM, metrics
- **gensim** - Word2Vec embeddings
- **pandas** - Excel file reading
- **openpyxl** - Excel file support
- **numpy** - Numerical operations
- **matplotlib**, **seaborn** - Visualization
- **tqdm** - Progress bars
- **openai** - OpenAI-compatible LLM API client (Stage 5)
- **anthropic** - Anthropic Claude API client (Stage 5)

Activate environment:
```bash
cd /home/u2023312337/task2/task2
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

## Important Implementation Details

### Label Convention
- `1` = Positive (correctly extracted title)
- `0` = Negative (incorrectly extracted title)

### Test Data Format
Excel file columns:
- `title given by manchine` - The title text (note: typo in column name is intentional from source data)
- `Y/N` - Label where Y=1 (correct), N=0 (incorrect)

### Model Interface
All classifiers must implement:
```python
def train(titles: List[str], labels: List[int]) -> None
def predict(titles: List[str]) -> np.ndarray
def save_model() -> None
def load_model() -> None
def get_feature_vectors(titles: List[str]) -> np.ndarray  # For t-SNE visualization
```

### GPU Acceleration
BERT training automatically detects and uses CUDA if available. Check with:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

For GPU memory issues (OOM errors), reduce `batch_size` (32→16) or `max_length` (96→64) in training scripts.

### Working Directory
Scripts use absolute paths and can be run from any directory:
```bash
# Works from anywhere
python /home/u2023312337/task2/task2/main_pipeline.py
python /home/u2023312337/task2/task2/train_bert_optimized_v2.py
```

However, `run_quick.sh` must be run from the task2 directory:
```bash
cd /home/u2023312337/task2/task2
./run_quick.sh
```

## Common Tasks

### Adding a New Classifier

1. Create new file `{name}_classifier.py` implementing the required interface:
   ```python
   class MyClassifier:
       def train(self, titles: List[str], labels: List[int]) -> None: ...
       def predict(self, titles: List[str]) -> np.ndarray: ...
       def save_model(self) -> None: ...
       def load_model(self) -> None: ...
       def get_feature_vectors(self, titles: List[str]) -> np.ndarray: ...
   ```

2. Import in `main_pipeline.py`:
   ```python
   from my_classifier import MyClassifier
   ```

3. Add to training section in `main()` function (around line 318-332)

4. Update `classifiers` dict with your model instance

5. Run full pipeline to test

### Modifying Training Parameters

**Naive Bayes** (`naive_bayes_classifier_optimized.py` constructor):
- `max_features_word` - Word-level TF-IDF features (default: 10000)
- `max_features_char` - Char-level TF-IDF features (default: 5000)
- `word_ngram_range` - Word n-gram range (default: (1,3))
- `char_ngram_range` - Char n-gram range (default: (3,5))
- `alpha` - Smoothing parameter (default: 0.5)

**Word2Vec+SVM** (`word2vec_svm_classifier.py` constructor):
- `vector_size` - Word embedding dimension (default: 100)
- `window` - Context window size (default: 5)
- `use_linear_svm` - Use LinearSVC instead of RBF (default: False)
- `add_features` - Add statistical features (default: True)

**BERT** (`main_pipeline.py:123-147` in `train_bert()` function):
- `epochs` - Training epochs (default: 5)
- `batch_size` - Batch size (default: 32)
- `learning_rate` - Learning rate (default: 2e-5)
- `warmup_steps` - Warmup steps (default: 500)

**BERT Optimized** (`train_bert_optimized_v2.py:707-750` in `main()` function):
- `model_name` - Pre-trained model: 'bert-base-uncased', 'allenai/scibert_scivocab_uncased', 'roberta-base', 'microsoft/deberta-v3-base'
- `max_length` - Max sequence length (default: 96)
- `loss_type` - Loss function: 'focal', 'weighted_ce', 'ce' (default: 'focal')
- `use_adversarial` - Enable adversarial training (default: True)
- `use_early_stopping` - Enable early stopping (default: True)

### Debugging Failed Predictions

Use error analysis in `evaluator.py`:
```python
from evaluator import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(test_labels, predictions, "Model Name", verbose=True)

# Calculate and print error analysis
error_analysis = evaluator.calculate_error_analysis(test_labels, predictions, test_titles)
evaluator.print_error_analysis(error_analysis, max_examples=20)
```

This will show:
- False positive examples (predicted correct, actually incorrect)
- False negative examples (predicted incorrect, actually correct)
- Common patterns in misclassified samples

### Generating Only Specific Visualizations

Edit `generate_visualizations()` function in `main_pipeline.py:193-245` to comment out unwanted plots:
```python
# Comment out t-SNE if you don't need it (saves time)
# for model_name, classifier in classifiers.items():
#     ...
```

### Running BERT Experiments

**Quick test** (verify environment works, 30 minutes):
```bash
cd /home/u2023312337/task2/task2
./run_quick.sh
# Choose option 3 (Quick test - 3 epochs)
```

**Single optimized training** (SciBERT + Focal Loss, 2-3 hours):
```bash
python /home/u2023312337/task2/task2/train_bert_optimized_v2.py
```

**Batch experiments** (5 experiments, 8-12 hours):
```bash
python /home/u2023312337/task2/task2/run_bert_experiments.py

# Check results
cat /home/u2023312337/task2/task2/models/experiments/comparison_report.txt
```

## Performance Benchmarks

| Model | Accuracy | Precision | Recall | F1 | Training Time | Cost |
|-------|----------|-----------|--------|-----|---------------|------|
| **Naive Bayes (Original)** | 73.46% | 73.59% | 84.86% | 78.82% | ~2 min | Free |
| **Naive Bayes (Optimized)** | 79.20% | 76.96% | 91.73% | 83.69% | ~3 min | Free |
| **Word2Vec + SVM** | 82.99% | 85.84% | 85.58% | 85.74% | ~10 min | Free |
| **BERT (Baseline)** | 87.91% | 90.29% | 88.35% | 89.59% | ~1 hour | Free |
| **SciBERT + Focal Loss** | 87-88% | - | 85-87% | 88-89% | ~2-3 hours | Free |
| **DeBERTa-v3 + Advanced** | 89-91% | - | 88-90% | 90-92% | ~3-5 hours | Free |
| **LLM: DeepSeek (8-shot)** | ~85% | - | - | ~85% | 0.5s/sample | ¥0.30/976 samples |
| **LLM: Qwen Turbo (8-shot)** | ~84% | - | - | ~84% | 0.6s/sample | ¥0.09/976 samples |
| **LLM: GLM-4 Plus (8-shot)** | ~86% | - | - | ~86% | 0.7s/sample | ¥12/976 samples |

All times are for full training on 232K samples with GPU acceleration (CUDA). LLM inference costs are for the test set of 976 samples.

**Note**: To compare all models systematically including stage-specific implementations, use:
```bash
python evaluate_all_stages.py
```

## Notes

- The codebase uses **mixed Chinese/English**: Chinese in print statements and comments, English in code and variable names
- Data files are large (7-8 MB text files) and gitignored
- Model files are also gitignored (438 MB for BERT)
- All visualization output uses English labels to avoid font issues
- The project has been run successfully on Linux with Python 3.11
- Git status shows modified model files (.pt, .pkl) - these are tracked with Git LFS or should be gitignored

## Git Workflow

This is part of a git repository. Current branch: `master`

Modified files in current status:
- `main_pipeline.py` - Main orchestration script
- Model files in `models/` directory (tracked changes)
- Output files in `output/` directory

When making changes:
```bash
git status                    # Check current status
git add main_pipeline.py      # Add specific files
git commit -m "message"       # Commit changes
git log --oneline -10         # View recent commits
```

Note: Large model files (.pt, .pkl) should be gitignored or use Git LFS.

## Troubleshooting

**Missing data files**: The pipeline will automatically use sample data if `data/positive.txt`, `data/negative.txt`, or `data/testSet-1000.xlsx` are missing.

**GPU memory issues**: Reduce `batch_size` from 32 to 16, or `max_length` from 96 to 64 in BERT training scripts.

**Module not found**: Activate virtual environment first with `source .venv/bin/activate`

**Chinese encoding issues**: The project handles Chinese text in comments and print statements. If you see encoding errors, ensure terminal supports UTF-8.

**Training interrupted**: For BERT, if early stopping is enabled, the best model is saved as `*_best.pt` and can be loaded later.

**LLM API errors** (Stage 5):
- Invalid API key: Check `llm_config.json` has correct API keys
- Rate limit exceeded: Increase `delay_between_calls` in config or use `--sample` to test with fewer samples
- Network timeout: Use domestic models (DeepSeek, GLM, Qwen) instead of international ones
- Response parsing fails: Check model output format, may need to adjust prompt in `run_llm_experiment.py`

## Reference Documentation

### Core Documentation
- `CLAUDE.md` - This file (comprehensive project guide)
- `VERSION_EVOLUTION.md` - Complete version evolution history with technical details
- `EVOLUTION_ROADMAP.md` - Quick reference roadmap and visual evolution path
- `OPTIMIZATION_SUMMARY.md` - Naive Bayes optimization technical details

### Stage-Specific Guides
- `stages/README.md` - Project organization by development stage
- `stages/RUN_GUIDE.md` - How to run stage-specific code
- `stages/TRAINING_GUIDE.md` - Training guidelines for each stage
- `stages/TEST_RESULTS.md` - Test results across all stages
- `stages/IMPLEMENTATION_GUIDE.md` - Implementation details for each stage

### LLM Experiments (Stage 5)
- `LLM_EXPERIMENT_GUIDE.md` - Complete LLM experiment setup and usage
- `LLM_COST_SUMMARY.md` - Cost analysis and API pricing comparison
- `llm_config_template.json` - Configuration template for LLM experiments

### Quick Navigation
- **Want to understand the project evolution?** → Start with `EVOLUTION_ROADMAP.md`
- **Need detailed technical analysis?** → Read `VERSION_EVOLUTION.md`
- **Setting up LLM experiments?** → Follow `LLM_EXPERIMENT_GUIDE.md`
- **Exploring stage implementations?** → See `stages/README.md`

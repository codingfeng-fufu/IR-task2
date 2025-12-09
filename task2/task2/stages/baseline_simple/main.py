"""Main pipeline for stage1 - Simple baseline implementation

This is the most basic implementation without any feature engineering or optimization.
It serves as the starting point (Stage 1) before adding optimizations.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to access data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_training_data, load_test_data
from naive_bayes import SimpleNaiveBayes
from word2vec_svm import SimpleWord2VecSVM
from bert_classifier import SimpleBERT
from evaluator import evaluate_model, compare_models
from visualizer import plot_model_comparison, plot_confusion_matrices, plot_tsne


def print_header():
    """Print program header."""
    print("\n" + "="*80)
    print(" " * 20 + "Stage 1: Simple Baseline Implementation")
    print(" " * 15 + "Academic Title Classification System")
    print("="*80)
    print("\nThis is the most basic implementation without optimization:")
    print("  - Naive Bayes: Simple TF-IDF features")
    print("  - Word2Vec + SVM: Basic word embeddings + RBF kernel")
    print("  - BERT: Pre-trained bert-base-uncased, fine-tuned")
    print("\n" + "="*80 + "\n")


def main():
    """Main pipeline execution."""
    print_header()

    # Configuration
    DATA_DIR = '../task2/data'  # Relative to stage1 directory
    OUTPUT_DIR = 'output'
    MODELS_DIR = 'models'

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ============================================================
    # Step 1: Load Data
    # ============================================================
    print("[Step 1/6] Loading Data")
    print("-" * 80)

    train_titles, train_labels = load_training_data(
        f"{DATA_DIR}/positive.txt",
        f"{DATA_DIR}/negative.txt"
    )

    test_titles, test_labels = load_test_data(
        f"{DATA_DIR}/testSet-1000.xlsx"
    )

    if len(train_titles) == 0 or len(test_titles) == 0:
        print("\nError: Failed to load data. Please check data files.")
        return

    # ============================================================
    # Step 2: Train Naive Bayes
    # ============================================================
    print("\n[Step 2/6] Training Naive Bayes Classifier")
    print("-" * 80)

    nb_classifier = SimpleNaiveBayes()
    nb_classifier.train(train_titles, train_labels)
    nb_classifier.save_model(f"{MODELS_DIR}/naive_bayes.pkl")

    # ============================================================
    # Step 3: Train Word2Vec + SVM
    # ============================================================
    print("\n[Step 3/6] Training Word2Vec + SVM Classifier")
    print("-" * 80)

    w2v_classifier = SimpleWord2VecSVM(vector_size=100, window=5)
    w2v_classifier.train(train_titles, train_labels)
    w2v_classifier.save_model(f"{MODELS_DIR}/word2vec_svm")

    # ============================================================
    # Step 4: Train BERT
    # ============================================================
    print("\n[Step 4/6] Training BERT Classifier")
    print("-" * 80)

    bert_classifier = SimpleBERT(model_name='bert-base-uncased', max_length=64)
    bert_classifier.train(train_titles, train_labels, epochs=3, batch_size=16)
    bert_classifier.save_model(f"{MODELS_DIR}/bert_model.pt")

    # ============================================================
    # Step 5: Evaluate All Models
    # ============================================================
    print("\n[Step 5/6] Evaluating Models")
    print("-" * 80)

    # Get predictions from all models
    nb_predictions = nb_classifier.predict(test_titles)
    w2v_predictions = w2v_classifier.predict(test_titles)
    bert_predictions = bert_classifier.predict(test_titles)

    # Evaluate each model
    nb_results = evaluate_model(test_labels, nb_predictions, "Naive Bayes")
    w2v_results = evaluate_model(test_labels, w2v_predictions, "Word2Vec + SVM")
    bert_results = evaluate_model(test_labels, bert_predictions, "BERT")

    # Compare all models
    all_results = [nb_results, w2v_results, bert_results]
    compare_models(all_results)

    # ============================================================
    # Step 6: Visualize Results
    # ============================================================
    print("\n[Step 6/6] Generating Visualizations")
    print("-" * 80)

    # Plot model comparison
    plot_model_comparison(all_results, OUTPUT_DIR)

    # Plot confusion matrices
    plot_confusion_matrices(all_results, OUTPUT_DIR)

    # Generate t-SNE visualizations
    print("\nGenerating t-SNE visualizations (this may take a few minutes)...")

    # Naive Bayes t-SNE
    nb_vectors = nb_classifier.get_feature_vectors(test_titles)
    plot_tsne(nb_vectors, test_labels, "Naive_Bayes", OUTPUT_DIR)

    # Word2Vec + SVM t-SNE
    w2v_vectors = w2v_classifier.get_feature_vectors(test_titles)
    plot_tsne(w2v_vectors, test_labels, "Word2Vec_SVM", OUTPUT_DIR)

    # BERT t-SNE
    bert_vectors = bert_classifier.get_feature_vectors(test_titles)
    plot_tsne(bert_vectors, test_labels, "BERT", OUTPUT_DIR)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*80)
    print(" " * 30 + "Pipeline Completed!")
    print("="*80)
    print(f"\nModels saved to: {MODELS_DIR}/")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print(f"  - {OUTPUT_DIR}/model_comparison.png")
    print(f"  - {OUTPUT_DIR}/confusion_matrices.png")
    print(f"  - {OUTPUT_DIR}/tsne_Naive_Bayes.png")
    print(f"  - {OUTPUT_DIR}/tsne_Word2Vec_SVM.png")
    print(f"  - {OUTPUT_DIR}/tsne_BERT.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

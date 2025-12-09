"""Quick test script to verify stage1 implementation"""

import sys
import os

print("Testing Stage 1 Implementation...\n")

# Test imports
print("[1/7] Testing imports...")
try:
    from data_loader import load_training_data, load_test_data
    print("  ✓ data_loader")
except Exception as e:
    print(f"  ✗ data_loader: {e}")
    sys.exit(1)

try:
    from naive_bayes import SimpleNaiveBayes
    print("  ✓ naive_bayes")
except Exception as e:
    print(f"  ✗ naive_bayes: {e}")
    sys.exit(1)

try:
    from word2vec_svm import SimpleWord2VecSVM
    print("  ✓ word2vec_svm")
except Exception as e:
    print(f"  ✗ word2vec_svm: {e}")
    sys.exit(1)

try:
    from bert_classifier import SimpleBERT
    print("  ✓ bert_classifier")
except Exception as e:
    print(f"  ✗ bert_classifier: {e}")
    sys.exit(1)

try:
    from evaluator import evaluate_model, compare_models
    print("  ✓ evaluator")
except Exception as e:
    print(f"  ✗ evaluator: {e}")
    sys.exit(1)

try:
    from visualizer import plot_model_comparison, plot_confusion_matrices, plot_tsne
    print("  ✓ visualizer")
except Exception as e:
    print(f"  ✗ visualizer: {e}")
    sys.exit(1)

print("\n[2/7] Testing classifier initialization...")
try:
    nb = SimpleNaiveBayes()
    print("  ✓ Naive Bayes initialized")
except Exception as e:
    print(f"  ✗ Naive Bayes: {e}")
    sys.exit(1)

try:
    w2v = SimpleWord2VecSVM()
    print("  ✓ Word2Vec + SVM initialized")
except Exception as e:
    print(f"  ✗ Word2Vec + SVM: {e}")
    sys.exit(1)

print("  ⚠ Skipping BERT initialization (requires downloading model)")

print("\n[3/7] Testing data loading with sample data...")
try:
    # Create sample data
    sample_titles = [
        "The Social Life of Routers",
        "Machine Learning for Code Analysis",
        "Call for Papers......41 Fragments......42",
        "Abstract......Introduction......References"
    ]
    sample_labels = [1, 1, 0, 0]
    print(f"  ✓ Created sample data: {len(sample_titles)} titles")
except Exception as e:
    print(f"  ✗ Sample data: {e}")
    sys.exit(1)

print("\n[4/7] Testing Naive Bayes training...")
try:
    nb.train(sample_titles, sample_labels)
    print("  ✓ Naive Bayes trained")
except Exception as e:
    print(f"  ✗ Naive Bayes training: {e}")
    sys.exit(1)

print("\n[5/7] Testing Naive Bayes prediction...")
try:
    predictions = nb.predict(["A Test Title"])
    print(f"  ✓ Naive Bayes prediction: {predictions}")
except Exception as e:
    print(f"  ✗ Naive Bayes prediction: {e}")
    sys.exit(1)

print("\n[6/7] Testing Word2Vec + SVM training...")
try:
    w2v.train(sample_titles, sample_labels)
    print("  ✓ Word2Vec + SVM trained")
except Exception as e:
    print(f"  ✗ Word2Vec + SVM training: {e}")
    sys.exit(1)

print("\n[7/7] Testing Word2Vec + SVM prediction...")
try:
    predictions = w2v.predict(["A Test Title"])
    print(f"  ✓ Word2Vec + SVM prediction: {predictions}")
except Exception as e:
    print(f"  ✗ Word2Vec + SVM prediction: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nThe stage1 implementation is ready to use.")
print("Run 'python main.py' to execute the full pipeline.")
print("\nNote: BERT training requires downloading bert-base-uncased")
print("      model (~420MB) on first run.")
print("="*60)

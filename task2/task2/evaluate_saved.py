"""
evaluate_saved_models.py
========================
Load pre-trained models and generate evaluation results and visualizations
NO TRAINING - only loads existing models from models/ directory
"""

import sys
import os
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import DataLoader, create_sample_data
from naive_bayes_classifier import NaiveBayesClassifier
from word2vec_svm_classifier import Word2VecSVMClassifier
from bert_classifier import BERTClassifier
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer


def print_header():
    """Print program header"""
    print("\n" + "="*80)
    print(" " * 20 + "Model Evaluation & Visualization")
    print(" " * 15 + "(Using Pre-trained Models)")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load pre-trained models from models/ directory")
    print("  2. Evaluate models on test data")
    print("  3. Generate visualization charts")
    print("  4. Save all results to output/ directory")
    print("\n" + "="*80 + "\n")


def load_test_data(use_sample_data: bool = False):
    """
    Load test dataset only (no training data needed)

    Args:
        use_sample_data: Whether to use sample data

    Returns:
        (test_titles, test_labels)
    """
    print("[Step 1/4] Loading Test Dataset")
    print("-" * 80)

    if use_sample_data:
        print("Using sample data...")
        _, _, test_titles, test_labels = create_sample_data()
    else:
        # Try loading actual files
        _, _, test_titles, test_labels = DataLoader.prepare_dataset(
            'data/positive.txt',
            'data/negative.txt',
            'data/testSet-1000.xlsx'
        )

        # If files don't exist, use sample data
        if len(test_titles) == 0:
            print("\n⚠️  Data files not found, switching to sample data...")
            _, _, test_titles, test_labels = create_sample_data()

    print(f"\nTest data loading complete!")
    print(f"  Test set: {len(test_titles)} samples")

    return test_titles, test_labels


def load_saved_models(bert_model_path='models/best_bert_model.pt'):
    """
    Load all pre-trained models from models/ directory
    
    Args:
        bert_model_path: Path to BERT model file
    
    Returns:
        Dictionary of loaded classifiers
    """
    print("\n[Step 2/4] Loading Pre-trained Models")
    print("-" * 80)
    
    classifiers = {}
    
    # 1. Load Naive Bayes
    print("\n1. Loading Naive Bayes model...")
    nb_path = 'models/naive_bayes_model.pkl'
    if not os.path.exists(nb_path):
        print(f"   ⚠️  Model file not found: {nb_path}")
        print(f"   Please train the model first using main.py")
        classifiers['Naive Bayes'] = None
    else:
        nb_classifier = NaiveBayesClassifier(model_path=nb_path)
        if nb_classifier.load_model():
            classifiers['Naive Bayes'] = nb_classifier
            print(f"   ✓ Naive Bayes model loaded successfully")
        else:
            classifiers['Naive Bayes'] = None
    
    # 2. Load Word2Vec+SVM
    print("\n2. Loading Word2Vec+SVM model...")
    w2v_path = 'models/word2vec_svm_model'
    w2v_w2v_file = f"{w2v_path}_w2v.model"
    w2v_svm_file = f"{w2v_path}_svm.pkl"
    
    if not os.path.exists(w2v_w2v_file) or not os.path.exists(w2v_svm_file):
        print(f"   ⚠️  Model files not found:")
        if not os.path.exists(w2v_w2v_file):
            print(f"      Missing: {w2v_w2v_file}")
        if not os.path.exists(w2v_svm_file):
            print(f"      Missing: {w2v_svm_file}")
        print(f"   Please train the model first using main.py")
        classifiers['Word2Vec+SVM'] = None
    else:
        w2v_classifier = Word2VecSVMClassifier(model_path=w2v_path)
        if w2v_classifier.load_model():
            classifiers['Word2Vec+SVM'] = w2v_classifier
            print(f"   ✓ Word2Vec+SVM model loaded successfully")
        else:
            classifiers['Word2Vec+SVM'] = None
    
    # 3. Load BERT model
    print("\n3. Loading BERT model...")
    if not os.path.exists(bert_model_path):
        print(f"   ⚠️  Model file not found: {bert_model_path}")
        print(f"   Please train the model first using main.py")
        classifiers['BERT'] = None
    else:
        try:
            bert_classifier = BERTClassifier(
                model_name='bert-base-uncased',
                max_length=64,
                model_path=bert_model_path
            )
            if bert_classifier.load_model():
                classifiers['BERT'] = bert_classifier
                print(f"   ✓ BERT model loaded successfully")
            else:
                classifiers['BERT'] = None
        except Exception as e:
            print(f"   ⚠️  Failed to load BERT model: {str(e)}")
            classifiers['BERT'] = None
    
    # Check if any models loaded successfully
    loaded_models = [name for name, clf in classifiers.items() if clf is not None]
    
    if len(loaded_models) == 0:
        print("\n" + "="*80)
        print("❌ ERROR: No models could be loaded!")
        print("="*80)
        print("\nPlease train models first by running:")
        print("  python main.py")
        print("\nThis will create model files in the models/ directory.")
        sys.exit(1)
    
    print("\n" + "-"*80)
    print(f"✓ Successfully loaded {len(loaded_models)} model(s): {', '.join(loaded_models)}")
    
    return classifiers


def evaluate_models(classifiers: Dict, test_titles: List[str], test_labels: List[int]):
    """Evaluate all loaded models"""
    print("\n[Step 3/4] Evaluating Models")
    print("-" * 80)
    
    evaluator = ModelEvaluator()
    results = []
    predictions_dict = {}
    
    for model_name, classifier in classifiers.items():
        if classifier is None:
            continue
        
        print(f"\nEvaluating {model_name}...")
        
        # Make predictions
        predictions = classifier.predict(test_titles)
        predictions_dict[model_name] = predictions
        
        # Evaluate
        result = evaluator.evaluate_model(
            test_labels,
            predictions,
            model_name,
            verbose=True
        )
        results.append(result)
    
    # Compare models
    if len(results) > 1:
        evaluator.compare_models(results)
    
    return results, predictions_dict


def generate_visualizations(
    classifiers: Dict,
    results: List[Dict],
    test_titles: List[str],
    test_labels: List[int],
    output_dir: str = 'output'
):
    """Generate visualization charts"""
    print("\n[Step 4/4] Generating Visualizations")
    print("-" * 80)
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}/")
    
    visualizer = ResultVisualizer()
    
    # 1. Performance comparison chart
    print("\n1. Generating model comparison chart...")
    visualizer.plot_comparison(
        results,
        save_path=os.path.join(output_dir, 'model_comparison.png')
    )
    
    # 2. Confusion matrices
    print("2. Generating confusion matrices...")
    visualizer.plot_confusion_matrices(
        results,
        save_path=os.path.join(output_dir, 'confusion_matrices.png')
    )
    
    # 3. t-SNE visualization (generate for each model)
    print("3. Generating t-SNE visualizations...")
    for model_name, classifier in classifiers.items():
        if classifier is None:
            continue
        
        try:
            # Get feature vectors
            print(f"   Extracting features for {model_name}...")
            feature_vectors = classifier.get_feature_vectors(test_titles)
            
            # Generate t-SNE visualization
            safe_name = model_name.replace(' ', '_').replace('+', '_')
            visualizer.visualize_embeddings_tsne(
                feature_vectors,
                test_labels,
                model_name,
                save_path=os.path.join(output_dir, f'tsne_{safe_name}.png'),
                perplexity=30,
                n_iter=1000
            )
        except Exception as e:
            print(f"   ⚠️  Failed to generate t-SNE for {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()


def save_results(results: List[Dict], predictions_dict: Dict, output_dir: str = 'output'):
    """Save results to files"""
    import json
    
    print("\nSaving Results")
    print("-" * 80)
    
    # Save evaluation results
    results_file = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" Model Evaluation Results\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall:    {result['recall']:.4f}\n")
            f.write(f"  F1 Score:  {result['f1']:.4f}\n")
            f.write(f"  F1 Macro:  {result['f1_macro']:.4f}\n")
            f.write(f"  F1 Micro:  {result['f1_micro']:.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"✓ Evaluation results saved to: {results_file}")
    
    # Save predictions
    predictions_file = os.path.join(output_dir, 'predictions.json')
    predictions_json = {
        model: pred.tolist() if hasattr(pred, 'tolist') else pred
        for model, pred in predictions_dict.items()
    }
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_json, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Predictions saved to: {predictions_file}")


def main():
    """Main function: Load models and generate results"""
    
    # Print header
    print_header()
    
    # Configuration
    USE_SAMPLE_DATA = False  # Whether to use sample data
    OUTPUT_DIR = 'output'    # Output directory
    BERT_MODEL_PATH = 'models/best_bert_model.pt'  # BERT model path

    try:
        # 1. Load test data
        test_titles, test_labels = load_test_data(USE_SAMPLE_DATA)
        
        # 2. Load saved models
        classifiers = load_saved_models(bert_model_path=BERT_MODEL_PATH)
        
        # 3. Evaluate models
        results, predictions_dict = evaluate_models(classifiers, test_titles, test_labels)
        
        # 4. Generate visualizations
        generate_visualizations(
            classifiers,
            results,
            test_titles,
            test_labels,
            output_dir=OUTPUT_DIR
        )
        
        # 5. Save results
        save_results(results, predictions_dict, output_dir=OUTPUT_DIR)
        
        # Complete
        print("\n" + "="*80)
        print(" Evaluation Complete!")
        print("="*80)
        print(f"\nAll results saved to '{OUTPUT_DIR}/' directory")
        print("\nGenerated files:")
        print("  1. model_comparison.png - Model performance comparison chart")
        print("  2. confusion_matrices.png - Confusion matrices")
        print("  3. tsne_*.png - t-SNE visualization charts")
        print("  4. evaluation_results.txt - Detailed evaluation results")
        print("  5. predictions.json - Prediction results")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
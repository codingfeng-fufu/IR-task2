"""
run_optimized_classifier.py
===========================
Complete script to run the optimized BERT classifier
with your actual data and compare different models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bert_classifier_optimized import OptimizedBERTClassifier
from data_loader import DataLoader, create_sample_data
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
import numpy as np


def train_and_evaluate_model(
    model_name,
    train_titles,
    train_labels,
    val_titles,
    val_labels,
    test_titles,
    test_labels,
    epochs=10,
    batch_size=32,
    use_custom_head=True,
    freeze_layers=0
):
    """
    Train and evaluate a single model
    
    Args:
        model_name: Name of the model to use
        train_titles: Training titles
        train_labels: Training labels
        val_titles: Validation titles
        val_labels: Validation labels
        test_titles: Test titles
        test_labels: Test labels
        epochs: Number of epochs
        batch_size: Batch size
        use_custom_head: Whether to use custom classification head
        freeze_layers: Number of layers to freeze
        
    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*70)
    print(f"Training Model: {model_name}")
    print("="*70)
    
    # Create classifier
    classifier = OptimizedBERTClassifier(
        model_name=model_name,
        max_length=64,
        model_path=f'models/{model_name.replace("/", "_")}_optimized.pt',
        use_custom_head=use_custom_head,
        dropout_rate=0.3
    )
    
    # Train model
    classifier.train(
        train_titles=train_titles,
        train_labels=train_labels,
        val_titles=val_titles,
        val_labels=val_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        scheduler_type='cosine',
        early_stopping_patience=3,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        save_model=True,
        freeze_layers=freeze_layers
    )
    
    # Make predictions on test set
    print("\n" + "="*70)
    print(f"Evaluating {model_name} on Test Set")
    print("="*70)
    
    predictions = classifier.predict(test_titles, batch_size=batch_size)
    probabilities = classifier.predict_proba(test_titles, batch_size=batch_size)
    
    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(
        test_labels,
        predictions,
        model_name=f"{model_name} (Optimized)",
        verbose=True
    )
    
    # Get feature vectors for visualization
    feature_vectors = classifier.get_feature_vectors(test_titles, batch_size=batch_size)
    
    return {
        'results': results,
        'predictions': predictions,
        'probabilities': probabilities,
        'feature_vectors': feature_vectors,
        'classifier': classifier
    }


def main():
    """Main function: Run optimized BERT classifier experiments"""
    
    print("="*70)
    print(" Optimized BERT Classifier - Full Experiment")
    print("="*70)
    
    # ===== Load Data =====
    print("\n【Step 1】Loading Data...")
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'positive_titles.txt',
        'negative_titles.txt',
        'test_titles.txt'
    )
    
    # If no actual files, use sample data
    if len(train_titles) == 0:
        print("\n⚠️  No data files found, using sample data for demonstration")
        train_titles, train_labels, test_titles, test_labels = create_sample_data()
    
    # Create validation set (20% of training data)
    val_size = len(train_titles) // 5
    val_titles = train_titles[:val_size]
    val_labels = train_labels[:val_size]
    train_titles = train_titles[val_size:]
    train_labels = train_labels[val_size:]
    
    print(f"\nDataset Split:")
    print(f"  - Training: {len(train_titles)} samples")
    print(f"  - Validation: {len(val_titles)} samples")
    print(f"  - Test: {len(test_titles)} samples")
    
    # ===== Choose Models to Compare =====
    # You can modify this list to test different models
    models_to_test = [
        ('scibert', True, 0),      # SciBERT with custom head (RECOMMENDED)
        ('roberta-base', True, 0), # RoBERTa with custom head
        # ('deberta-v3', True, 0), # Uncomment to test DeBERTa (slower but better)
        # ('albert-base', False, 4), # Uncomment to test ALBERT (faster)
    ]
    
    print("\n【Step 2】Models to Test:")
    for i, (model_name, use_head, freeze) in enumerate(models_to_test, 1):
        print(f"  {i}. {model_name} (custom_head={use_head}, freeze_layers={freeze})")
    
    # ===== Train and Evaluate Each Model =====
    all_results = []
    all_feature_vectors = []
    
    for model_name, use_custom_head, freeze_layers in models_to_test:
        try:
            model_output = train_and_evaluate_model(
                model_name=model_name,
                train_titles=train_titles,
                train_labels=train_labels,
                val_titles=val_titles,
                val_labels=val_labels,
                test_titles=test_titles,
                test_labels=test_labels,
                epochs=10,  # Adjust as needed
                batch_size=32,
                use_custom_head=use_custom_head,
                freeze_layers=freeze_layers
            )
            
            all_results.append(model_output['results'])
            all_feature_vectors.append({
                'vectors': model_output['feature_vectors'],
                'model_name': model_name
            })
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # ===== Compare Models =====
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("【Step 3】Comparing All Models")
        print("="*70)
        
        evaluator = ModelEvaluator()
        evaluator.compare_models(all_results)
        
        # ===== Generate Visualizations =====
        print("\n" + "="*70)
        print("【Step 4】Generating Visualizations")
        print("="*70)
        
        visualizer = ResultVisualizer()
        
        # Comparison chart
        visualizer.plot_comparison(
            all_results,
            save_path='outputs/optimized_model_comparison.png'
        )
        
        # Confusion matrices
        visualizer.plot_confusion_matrices(
            all_results,
            save_path='outputs/optimized_confusion_matrices.png'
        )
        
        # t-SNE visualizations
        for fv_data in all_feature_vectors:
            visualizer.visualize_embeddings_tsne(
                vectors=fv_data['vectors'],
                labels=test_labels,
                title=fv_data['model_name'],
                save_path=f'outputs/tsne_{fv_data["model_name"].replace("/", "_")}.png'
            )
        
        print("\n✓ All visualizations saved to outputs/ directory")
    
    # ===== Final Summary =====
    print("\n" + "="*70)
    print(" EXPERIMENT COMPLETE!")
    print("="*70)
    
    print("\n【Best Model】")
    if all_results:
        best_model = max(all_results, key=lambda x: x['f1'])
        print(f"  Model: {best_model['model']}")
        print(f"  F1 Score: {best_model['f1']:.4f}")
        print(f"  Accuracy: {best_model['accuracy']:.4f}")
        print(f"  Precision: {best_model['precision']:.4f}")
        print(f"  Recall: {best_model['recall']:.4f}")
    
    print("\n【Outputs】")
    print("  - Models saved in: models/")
    print("  - Visualizations in: outputs/")
    
    print("\n【Next Steps】")
    print("  1. Review the comparison charts in outputs/")
    print("  2. Load the best model for production use")
    print("  3. Fine-tune hyperparameters if needed")
    print("  4. Consider testing additional models (DeBERTa, etc.)")


def quick_test_single_model():
    """
    Quick test function to try a single model
    Useful for rapid experimentation
    """
    print("="*70)
    print(" Quick Test - Single Model")
    print("="*70)
    
    # Load data
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'positive_titles.txt',
        'negative_titles.txt',
        'test_titles.txt'
    )
    
    if len(train_titles) == 0:
        train_titles, train_labels, test_titles, test_labels = create_sample_data()
    
    # Use subset for quick testing
    train_titles = train_titles[:200]
    train_labels = train_labels[:200]
    test_titles = test_titles[:50]
    test_labels = test_labels[:50]
    
    # Create validation set
    val_titles = train_titles[:40]
    val_labels = train_labels[:40]
    
    # Test SciBERT
    classifier = OptimizedBERTClassifier(
        model_name='scibert',
        max_length=64,
        model_path='models/scibert_quick_test.pt',
        use_custom_head=True,
        dropout_rate=0.3
    )
    
    # Quick training
    classifier.train(
        train_titles=train_titles,
        train_labels=train_labels,
        val_titles=val_titles,
        val_labels=val_labels,
        epochs=3,  # Just 3 epochs for quick test
        batch_size=16,
        learning_rate=2e-5,
        early_stopping_patience=2
    )
    
    # Evaluate
    predictions = classifier.predict(test_titles)
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_model(
        test_labels,
        predictions,
        model_name="SciBERT (Quick Test)"
    )
    
    print("\n✓ Quick test complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run optimized BERT classifier')
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'quick'],
        help='Run mode: full experiment or quick test'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if args.mode == 'quick':
        quick_test_single_model()
    else:
        main()

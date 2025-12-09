"""Simple evaluation module for stage1"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict
import numpy as np


def evaluate_model(y_true: List[int], y_pred: np.ndarray, model_name: str) -> Dict:
    """
    Evaluate model performance with standard metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n=== Evaluation Results for {model_name} ===")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    # Macro and Micro averages
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"F1 Macro:  {f1_macro:.4f}")
    print(f"F1 Micro:  {f1_micro:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'confusion_matrix': cm
    }

    return results


def compare_models(results_list: List[Dict]):
    """
    Compare multiple models.

    Args:
        results_list: List of result dictionaries from evaluate_model
    """
    print("\n" + "="*80)
    print(" " * 25 + "Model Comparison")
    print("="*80)

    # Print header
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)

    # Print each model's results
    for result in results_list:
        print(f"{result['model_name']:<20} "
              f"{result['accuracy']:>10.4f} "
              f"{result['precision']:>10.4f} "
              f"{result['recall']:>10.4f} "
              f"{result['f1']:>10.4f}")

    print("="*80)

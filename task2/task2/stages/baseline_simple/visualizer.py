"""Simple visualization module for stage1"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from typing import List, Dict
import os


def plot_model_comparison(results_list: List[Dict], output_dir: str = 'output'):
    """
    Plot comparison of model performance.

    Args:
        results_list: List of result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    models = [r['model_name'] for r in results_list]
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results_list]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved model comparison plot to {output_dir}/model_comparison.png")
    plt.close()


def plot_confusion_matrices(results_list: List[Dict], output_dir: str = 'output'):
    """
    Plot confusion matrices for all models.

    Args:
        results_list: List of result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, result in zip(axes, results_list):
        cm = result['confusion_matrix']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        ax.set_title(result['model_name'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrices to {output_dir}/confusion_matrices.png")
    plt.close()


def plot_tsne(feature_vectors: np.ndarray, labels: List[int], model_name: str, output_dir: str = 'output'):
    """
    Plot t-SNE visualization of feature vectors.

    Args:
        feature_vectors: Feature vectors from model
        labels: True labels
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating t-SNE visualization for {model_name}...")

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(feature_vectors)

    # Plot
    plt.figure(figsize=(10, 8))

    # Plot positive samples (green)
    positive_mask = np.array(labels) == 1
    plt.scatter(embeddings_2d[positive_mask, 0],
               embeddings_2d[positive_mask, 1],
               c='green', label='Positive (Correct)', alpha=0.6, s=20)

    # Plot negative samples (red)
    negative_mask = np.array(labels) == 0
    plt.scatter(embeddings_2d[negative_mask, 0],
               embeddings_2d[negative_mask, 1],
               c='red', label='Negative (Wrong)', alpha=0.6, s=20)

    plt.title(f't-SNE Visualization - {model_name}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    filename = f"{output_dir}/tsne_{model_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to {filename}")
    plt.close()

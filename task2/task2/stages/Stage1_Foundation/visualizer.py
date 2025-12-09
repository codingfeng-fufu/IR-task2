"""
visualizer.py
=============
Visualization Module
Generate various charts to display classification results and model performance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# Set standard English fonts - NO CHINESE FONTS
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
print("Using font: DejaVu Sans")


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ResultVisualizer:
    """
    Result Visualizer
    Generate various charts for analyzing and displaying model performance
    """

    @staticmethod
    def plot_comparison(results: List[Dict], save_path='model_comparison.png'):
        """
        Plot performance comparison of multiple models

        Args:
            results: List of evaluation results
            save_path: Save path
        """
        print("\nGenerating model performance comparison chart...")

        # 动态确定可用的指标（兼容不同格式）
        all_possible_metrics = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_micro']
        metric_names = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1 Score',
            'f1_macro': 'F1 Macro',
            'f1_micro': 'F1 Micro'
        }

        # 确定实际可用的指标（所有结果都有的指标）
        available_metrics = []
        for metric in all_possible_metrics:
            if all(metric in r for r in results):
                available_metrics.append(metric)

        if not available_metrics:
            raise ValueError("No common metrics found in results")

        metrics = available_metrics
        models = [r['model'] for r in results]
        data = {metric: [r[metric] for r in results] for metric in metrics}

        # 根据可用指标数量调整子图布局
        n_metrics = len(metrics)
        if n_metrics <= 3:
            n_rows, n_cols = 1, n_metrics
            figsize = (6 * n_metrics, 5)
        elif n_metrics == 4:
            n_rows, n_cols = 2, 2
            figsize = (12, 10)
        else:
            n_rows, n_cols = 2, 3
            figsize = (18, 10)

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle('Model Performance Comparison',
                     fontsize=16, fontweight='bold', y=0.98)

        # 确保 axes 是一维数组
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_metrics > 1 else [axes]

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(
                range(len(models)),
                data[metric],
                color=colors[:len(models)],
                alpha=0.8,
                edgecolor='black',
                linewidth=1.5
            )

            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(metric_names[metric], fontsize=13, fontweight='bold', pad=10)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.02,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

        # 隐藏多余的子图
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to: {save_path}")
        plt.close()


    @staticmethod
    def plot_confusion_matrices(results: List[Dict], save_path='confusion_matrices.png'):
        """Plot confusion matrix heatmaps for all models"""
        print("\nGenerating confusion matrix charts...")

        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        fig.suptitle('Confusion Matrix Comparison',
                     fontsize=16, fontweight='bold', y=1.02)

        for idx, result in enumerate(results):
            cm = result['confusion_matrix']
            ax = axes[idx]

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='white',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'}
            )

            ax.set_title(f"{result['model']}\nConfusion Matrix",
                         fontsize=13, fontweight='bold', pad=10)
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix chart saved to: {save_path}")
        plt.close()

    @staticmethod
    def visualize_embeddings_tsne(
            vectors: np.ndarray, labels: List[int], title: str,
            save_path='tsne_visualization.png', perplexity=30, n_iter=1000
    ):
        """Visualize high-dimensional vectors in 2D space using t-SNE"""
        print(f"\nGenerating t-SNE visualization: {title}...")

        tsne = TSNE(
            n_components=2, random_state=42,
            perplexity=min(perplexity, len(vectors) - 1),
            n_iter=n_iter, verbose=0
        )
        vectors_2d = tsne.fit_transform(vectors)

        fig, ax = plt.subplots(figsize=(12, 9))

        labels = np.array(labels)
        neg_mask = labels == 0
        pos_mask = labels == 1

        # Red circles for negative (incorrect titles)
        ax.scatter(
            vectors_2d[neg_mask, 0], vectors_2d[neg_mask, 1],
            c='#e74c3c', label='Negative',
            alpha=0.6, s=80, edgecolors='black', linewidth=0.5, marker='o'
        )

        # Green triangles for positive (correct titles)
        ax.scatter(
            vectors_2d[pos_mask, 0], vectors_2d[pos_mask, 1],
            c='#2ecc71', label='Positive',
            alpha=0.6, s=80, edgecolors='black', linewidth=0.5, marker='^'
        )

        ax.set_title(
            f't-SNE Visualization: {title}',
            fontsize=15, fontweight='bold', pad=15
        )
        ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right', framealpha=0.9, edgecolor='black', shadow=True)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to: {save_path}")
        plt.close()


def main():
    """Main function: Demonstrate visualizer usage"""
    from data_loader import create_sample_data

    print("=" * 70)
    print(" Visualization Module Demo")
    print("=" * 70)

    _, _, test_titles, test_labels = create_sample_data()

    results = [
        {
            'model': 'Naive Bayes',
            'accuracy': 0.92, 'precision': 0.91, 'recall': 0.93,
            'f1': 0.92, 'f1_macro': 0.92, 'f1_micro': 0.92,
            'confusion_matrix': np.array([[42, 3], [5, 40]])
        },
        {
            'model': 'Word2Vec+SVM',
            'accuracy': 0.89, 'precision': 0.88, 'recall': 0.90,
            'f1': 0.89, 'f1_macro': 0.89, 'f1_micro': 0.89,
            'confusion_matrix': np.array([[40, 5], [5, 40]])
        },
        {
            'model': 'BERT',
            'accuracy': 0.96, 'precision': 0.95, 'recall': 0.97,
            'f1': 0.96, 'f1_macro': 0.96, 'f1_micro': 0.96,
            'confusion_matrix': np.array([[44, 1], [2, 43]])
        }
    ]

    visualizer = ResultVisualizer()
    visualizer.plot_comparison(results, save_path='demo_comparison.png')
    visualizer.plot_confusion_matrices(results, save_path='demo_confusion_matrices.png')

    np.random.seed(42)
    vectors = np.random.randn(len(test_labels), 100)
    visualizer.visualize_embeddings_tsne(
        vectors, test_labels, "Demo Model", save_path='demo_tsne.png'
    )

    print("\nAll visualization charts generated successfully!")


if __name__ == "__main__":
    main()
"""
visualizer.py
=============
可视化模块
生成各种图表用于展示分类结果和模型性能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ResultVisualizer:
    """
    结果可视化器
    生成各种图表用于分析和展示模型性能
    """

    @staticmethod
    def plot_comparison(results: List[Dict], save_path='model_comparison.png'):
        """
        绘制多个模型的性能对比图

        参数:
            results: 评估结果列表
            save_path: 保存路径
        """
        print("\n生成模型性能对比图...")

        metrics = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_micro']
        metric_names = {
            'accuracy': 'Accuracy\n(准确率)',
            'precision': 'Precision\n(精确率)',
            'recall': 'Recall\n(召回率)',
            'f1': 'F1 Score\n(F1分数)',
            'f1_macro': 'F1 Macro\n(F1宏平均)',
            'f1_micro': 'F1 Micro\n(F1微平均)'
        }

        models = [r['model'] for r in results]
        data = {metric: [r[metric] for r in results] for metric in metrics}

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('模型性能对比 / Model Performance Comparison',
                     fontsize=16, fontweight='bold', y=0.98)

        axes = axes.flatten()
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

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 对比图已保存至: {save_path}")
        plt.close()

    @staticmethod
    def plot_confusion_matrices(results: List[Dict], save_path='confusion_matrices.png'):
        """绘制所有模型的混淆矩阵热力图"""
        print("\n生成混淆矩阵图...")

        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        fig.suptitle('混淆矩阵对比 / Confusion Matrix Comparison',
                     fontsize=16, fontweight='bold', y=1.02)

        for idx, result in enumerate(results):
            cm = result['confusion_matrix']
            ax = axes[idx]

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative\n(错误)', 'Positive\n(正确)'],
                yticklabels=['Negative\n(错误)', 'Positive\n(正确)'],
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='white',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'}
            )

            ax.set_title(f"{result['model']}\nConfusion Matrix",
                         fontsize=13, fontweight='bold', pad=10)
            ax.set_ylabel('True Label / 真实标签', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Label / 预测标签', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵图已保存至: {save_path}")
        plt.close()

    @staticmethod
    def visualize_embeddings_tsne(
            vectors: np.ndarray, labels: List[int], title: str,
            save_path='tsne_visualization.png', perplexity=30, n_iter=1000
    ):
        """
        使用t-SNE将高维向量可视化到2D空间

        t-SNE (t-distributed Stochastic Neighbor Embedding):
        - 降维算法，将高维数据（如100维词向量或768维BERT向量）映射到2D/3D
        - 保持数据点之间的相对距离关系
        - 用于可视化特征空间的聚类效果
        """
        print(f"\n生成t-SNE可视化图: {title}...")

        # ====================
        # 第1步: 创建t-SNE降维器
        # ====================
        tsne = TSNE(
            n_components=2,                                  # 降到2维（用于2D平面绘图）
            random_state=42,                                 # 随机种子（保证可复现）
            perplexity=min(perplexity, len(vectors) - 1),  # 困惑度（平衡局部和全局结构）
            max_iter=n_iter,                                 # 最大迭代次数（1000次通常足够收敛）
            verbose=0                                        # 不打印训练日志
        )
        # 执行降维：将 [n_samples, n_features] 转为 [n_samples, 2]
        vectors_2d = tsne.fit_transform(vectors)

        # ====================
        # 第2步: 创建画布
        # ====================
        fig, ax = plt.subplots(figsize=(12, 9))  # 创建12x9英寸的图表

        # ====================
        # 第3步: 分离不同类别的数据点
        # ====================
        labels = np.array(labels)  # 转为numpy数组便于布尔索引
        neg_mask = labels == 0     # 负样本（错误标题）的布尔掩码
        pos_mask = labels == 1     # 正样本（正确标题）的布尔掩码

        # ====================
        # 第4步: 绘制负样本（错误标题）散点图
        # ====================
        ax.scatter(
            vectors_2d[neg_mask, 0],     # X坐标（t-SNE第1维）
            vectors_2d[neg_mask, 1],     # Y坐标（t-SNE第2维）
            c='#e74c3c',                  # 红色
            label='Negative (错误标题)', # 图例标签
            alpha=0.6,                    # 透明度60%（避免重叠遮挡）
            s=80,                         # 点大小
            edgecolors='black',           # 黑色边框（使点更清晰）
            linewidth=0.5,                # 边框宽度
            marker='o'                    # 圆形标记
        )

        # ====================
        # 第5步: 绘制正样本（正确标题）散点图
        # ====================
        ax.scatter(
            vectors_2d[pos_mask, 0],     # X坐标
            vectors_2d[pos_mask, 1],     # Y坐标
            c='#2ecc71',                  # 绿色
            label='Positive (正确标题)', # 图例标签
            alpha=0.6,                    # 透明度60%
            s=80,                         # 点大小
            edgecolors='black',           # 黑色边框
            linewidth=0.5,                # 边框宽度
            marker='^'                    # 三角形标记（与负样本区分）
        )

        ax.set_title(
            f't-SNE Visualization: {title}\nt-SNE可视化',
            fontsize=15, fontweight='bold', pad=15
        )
        ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right', framealpha=0.9, edgecolor='black', shadow=True)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ t-SNE可视化图已保存至: {save_path}")
        plt.close()


def main():
    """主函数:演示可视化器的使用"""
    from data_loader import create_sample_data

    print("=" * 70)
    print(" 可视化模块演示")
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

    print("\n所有可视化图表生成完成!")


if __name__ == "__main__":
    main()
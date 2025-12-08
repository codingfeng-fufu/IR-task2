"""
visualize_bert_experiments_complete.py
=======================================
完整的BERT实验可视化脚本
- 加载训练好的模型
- 提取特征向量
- 生成t-SNE可视化
- 生成模型对比图和混淆矩阵

使用方法:
    cd /home/u2023312337/task2/task2
    source .venv/bin/activate
    python visualize_bert_experiments_complete.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from typing import List, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
from data_loader import DataLoader as TitleDataLoader
from train_bert_optimized_v2 import OptimizedBERTClassifier

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_experiment_results(results_path: str) -> List[Dict]:
    """加载实验结果JSON"""
    print(f"加载实验结果: {results_path}")
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"✓ 加载了 {len(results)} 个实验\n")
    return results


def load_model_and_get_predictions(
    experiment_name: str,
    config: Dict,
    test_titles: List[str],
    test_labels: List[int],
    script_dir: str
) -> tuple:
    """
    加载模型并获取预测结果和特征向量

    Returns:
        (predictions, feature_vectors) or (None, None) if failed
    """
    print(f"\n{'='*80}")
    print(f"处理实验: {experiment_name}")
    print(f"{'='*80}")

    # 尝试加载_best模型
    model_path = os.path.join(script_dir, f"models/experiments/{experiment_name}_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(script_dir, f"models/experiments/{experiment_name}.pt")

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None, None

    print(f"模型文件: {os.path.basename(model_path)}")
    print(f"配置:")
    print(f"  - Model: {config.get('model_name', 'unknown')}")
    print(f"  - Max Length: {config.get('max_length', 96)}")
    print(f"  - Loss Type: {config.get('loss_type', 'ce')}")

    # 创建分类器
    try:
        classifier = OptimizedBERTClassifier(
            model_name=config.get('model_name', 'scibert'),
            max_length=config.get('max_length', 96),
            model_path=model_path,
            dropout_rate=config.get('dropout_rate', 0.2)
        )

        # 加载模型
        classifier.load_model()
        print("✓ 模型加载成功")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # 预测
    print(f"\n预测测试集 ({len(test_titles)} 样本)...")
    try:
        predictions = classifier.predict(test_titles)
        print(f"✓ 预测完成")

        # 计算准确率验证
        accuracy = np.mean(predictions == np.array(test_labels))
        print(f"  实际准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return None, None

    # 提取特征向量
    print(f"\n提取特征向量...")
    try:
        feature_vectors = classifier.get_feature_vectors(test_titles, batch_size=32)
        print(f"✓ 特征提取完成: {feature_vectors.shape}")

    except Exception as e:
        print(f"❌ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return predictions, None

    return predictions, feature_vectors


def plot_tsne_visualization(
    vectors: np.ndarray,
    labels: List[int],
    experiment_name: str,
    config: Dict,
    save_path: str,
    perplexity: int = 30,
    n_iter: int = 1000
):
    """生成t-SNE可视化"""
    print(f"\n生成t-SNE可视化: {experiment_name}...")

    # 执行t-SNE降维
    print(f"  执行t-SNE降维 (perplexity={perplexity}, n_iter={n_iter})...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(perplexity, len(vectors) - 1),
        n_iter=n_iter,
        verbose=0
    )
    vectors_2d = tsne.fit_transform(vectors)
    print(f"  ✓ t-SNE完成")

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 10))

    labels = np.array(labels)
    neg_mask = labels == 0
    pos_mask = labels == 1

    # 负样本（红色圆圈）
    scatter_neg = ax.scatter(
        vectors_2d[neg_mask, 0], vectors_2d[neg_mask, 1],
        c='#e74c3c', label='Negative (Incorrect Title)',
        alpha=0.6, s=100, edgecolors='black', linewidth=0.5, marker='o'
    )

    # 正样本（绿色三角）
    scatter_pos = ax.scatter(
        vectors_2d[pos_mask, 0], vectors_2d[pos_mask, 1],
        c='#2ecc71', label='Positive (Correct Title)',
        alpha=0.6, s=100, edgecolors='black', linewidth=0.5, marker='^'
    )

    # 标题
    model_name = config.get('model_name', 'unknown').upper()
    max_len = config.get('max_length', 96)
    loss_type = config.get('loss_type', 'ce').upper()

    title = f't-SNE Visualization: {experiment_name}\n'
    title += f'Model: {model_name} | Max Length: {max_len} | Loss: {loss_type}'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Component 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Component 2', fontsize=14, fontweight='bold')

    # 图例
    ax.legend(
        fontsize=12, loc='upper right',
        framealpha=0.95, edgecolor='black',
        shadow=True, fancybox=True
    )

    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ t-SNE保存: {save_path}")
    plt.close()


def plot_model_comparison(results_data: List[Dict], save_path: str):
    """生成模型性能对比图"""
    print(f"\n生成模型性能对比图...")

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score'
    }

    models = [r['model'] for r in results_data]
    data = {metric: [r[metric] for r in results_data] for metric in metrics}

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BERT Experiments Performance Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    axes = axes.flatten()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(
            range(len(models)),
            data[metric],
            color=colors[:len(models)],
            alpha=0.85,
            edgecolor='black',
            linewidth=2
        )

        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title(metric_names[metric], fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha='right', fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.02,
                f'{height:.4f}\n({height*100:.2f}%)',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图保存: {save_path}")
    plt.close()


def plot_confusion_matrices(results_data: List[Dict], save_path: str):
    """生成混淆矩阵可视化"""
    print(f"\n生成混淆矩阵...")

    n_models = len(results_data)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    fig.suptitle('Confusion Matrix Comparison - BERT Experiments',
                 fontsize=18, fontweight='bold', y=1.02)

    for idx, result in enumerate(results_data):
        cm = result['confusion_matrix']
        ax = axes[idx]

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'},
            linewidths=2.5, linecolor='white',
            annot_kws={'fontsize': 16, 'fontweight': 'bold'}
        )

        ax.set_title(
            f"{result['model']}\nAcc: {result['accuracy']:.4f} | F1: {result['f1']:.4f}",
            fontsize=14, fontweight='bold', pad=12
        )
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵保存: {save_path}")
    plt.close()


def main():
    """主函数"""
    print("\n" + "="*100)
    print(" BERT实验完整可视化（含t-SNE）")
    print("="*100)

    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义路径
    results_path = os.path.join(script_dir, 'models/experiments/results.json')
    output_dir = os.path.join(script_dir, 'output/bert_experiments')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}\n")

    # 检查结果文件
    if not os.path.exists(results_path):
        print(f"❌ 实验结果文件不存在: {results_path}")
        print("请先运行 run_bert_experiments.py")
        return

    # 加载实验结果
    experiments = load_experiment_results(results_path)

    # 加载测试数据
    print("加载测试数据...")
    _, _, test_titles, test_labels = TitleDataLoader.prepare_dataset(
        os.path.join(script_dir, 'data/positive.txt'),
        os.path.join(script_dir, 'data/negative.txt'),
        os.path.join(script_dir, 'data/testSet-1000.xlsx')
    )
    print(f"✓ 测试集: {len(test_titles)} 个样本")
    print(f"  正样本: {sum(test_labels)} ({sum(test_labels)/len(test_labels)*100:.1f}%)")
    print(f"  负样本: {len(test_labels)-sum(test_labels)} ({(len(test_labels)-sum(test_labels))/len(test_labels)*100:.1f}%)")

    # 过滤有效实验（排除失败的）
    valid_experiments = [
        exp for exp in experiments
        if exp['status'] == 'success' and exp['accuracy'] > 0.5
    ]

    print(f"\n有效实验数量: {len(valid_experiments)}/{len(experiments)}")
    for exp in valid_experiments:
        print(f"  ✓ {exp['experiment_name']}")

    # 存储结果用于后续可视化
    all_results = []

    # 处理每个实验
    for exp in valid_experiments:
        exp_name = exp['experiment_name']

        # 加载模型并获取预测和特征
        predictions, features = load_model_and_get_predictions(
            exp_name,
            exp['config'],
            test_titles,
            test_labels,
            script_dir
        )

        if predictions is None:
            print(f"⏭  跳过 {exp_name}\n")
            continue

        # 计算混淆矩阵
        cm = confusion_matrix(test_labels, predictions)

        # 保存结果
        all_results.append({
            'model': exp_name,
            'accuracy': exp['accuracy'],
            'precision': exp['precision'],
            'recall': exp['recall'],
            'f1': exp['f1'],
            'confusion_matrix': cm,
            'predictions': predictions
        })

        # 生成t-SNE可视化（如果特征提取成功）
        if features is not None:
            tsne_path = os.path.join(output_dir, f"tsne_{exp_name}.png")
            plot_tsne_visualization(
                features,
                test_labels,
                exp_name,
                exp['config'],
                tsne_path,
                perplexity=30,
                n_iter=1000
            )
        else:
            print(f"⚠️  跳过t-SNE: {exp_name} (特征提取失败)")

    # 生成汇总可视化
    if len(all_results) > 0:
        print("\n" + "="*100)
        print(" 生成汇总可视化")
        print("="*100)

        # 模型对比图
        plot_model_comparison(
            all_results,
            os.path.join(output_dir, "model_comparison.png")
        )

        # 混淆矩阵
        plot_confusion_matrices(
            all_results,
            os.path.join(output_dir, "confusion_matrices.png")
        )

    # 生成总结报告
    print("\n" + "="*100)
    print(" 生成总结报告")
    print("="*100)

    report_path = os.path.join(output_dir, "visualization_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(" BERT实验可视化报告\n")
        f.write("="*100 + "\n\n")

        f.write(f"测试集大小: {len(test_titles)} 样本\n")
        f.write(f"  正样本: {sum(test_labels)} ({sum(test_labels)/len(test_labels)*100:.1f}%)\n")
        f.write(f"  负样本: {len(test_labels)-sum(test_labels)} ({(len(test_labels)-sum(test_labels))/len(test_labels)*100:.1f}%)\n\n")

        f.write("成功生成可视化的实验:\n")
        f.write("-"*100 + "\n")
        for result in all_results:
            f.write(f"\n{result['model']}:\n")
            f.write(f"  Accuracy:  {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)\n")
            f.write(f"  Recall:    {result['recall']:.4f} ({result['recall']*100:.2f}%)\n")
            f.write(f"  F1 Score:  {result['f1']:.4f} ({result['f1']*100:.2f}%)\n")
            f.write(f"  混淆矩阵:\n")
            f.write(f"    TN={result['confusion_matrix'][0,0]:4d}  FP={result['confusion_matrix'][0,1]:4d}\n")
            f.write(f"    FN={result['confusion_matrix'][1,0]:4d}  TP={result['confusion_matrix'][1,1]:4d}\n")

        if len(all_results) > 0:
            best_f1 = max(all_results, key=lambda x: x['f1'])
            best_recall = max(all_results, key=lambda x: x['recall'])

            f.write("\n\n" + "="*100 + "\n")
            f.write(" 最佳模型\n")
            f.write("="*100 + "\n")
            f.write(f"\n最高F1分数: {best_f1['model']}\n")
            f.write(f"  F1: {best_f1['f1']:.4f} ({best_f1['f1']*100:.2f}%)\n")
            f.write(f"  Accuracy: {best_f1['accuracy']:.4f}\n")
            f.write(f"  Recall: {best_f1['recall']:.4f}\n")

            f.write(f"\n最高Recall: {best_recall['model']}\n")
            f.write(f"  Recall: {best_recall['recall']:.4f} ({best_recall['recall']*100:.2f}%)\n")
            f.write(f"  F1: {best_recall['f1']:.4f}\n")
            f.write(f"  Accuracy: {best_recall['accuracy']:.4f}\n")

    print(f"✓ 报告保存: {report_path}")

    # 列出所有生成的文件
    print("\n" + "="*100)
    print(" ✅ 可视化完成!")
    print("="*100)
    print(f"\n输出目录: {output_dir}\n")
    print("生成的文件:")

    files = sorted([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
    for file in files:
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  - {file:<50} ({file_size:>8.1f} KB)")

    print("\n" + "="*100)


if __name__ == "__main__":
    main()

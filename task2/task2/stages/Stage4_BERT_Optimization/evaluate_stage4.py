"""
evaluate_stage4.py
==================
评估Stage4已训练好的BERT模型并生成输出文件
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE

# 导入必要的类
from data_loader import DataLoader as TitleDataLoader
from train_bert_optimized_v2 import OptimizedBERTClassifier
from evaluator import ModelEvaluator

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_evaluate(model_path, model_name, config, test_titles, test_labels):
    """加载模型并评估"""
    print(f"\n{'='*80}")
    print(f"评估模型: {model_name}")
    print(f"{'='*80}")

    # 先读取checkpoint以获取正确的模型名称
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_name' in checkpoint:
            actual_model_name = checkpoint['model_name']
            print(f"从checkpoint读取模型名称: {actual_model_name}")
            config['model_name'] = actual_model_name
    except Exception as e:
        print(f"⚠️  无法读取checkpoint中的模型名称，使用配置中的名称: {e}")

    # 创建分类器实例
    classifier = OptimizedBERTClassifier(
        model_name=config.get('model_name', 'scibert'),
        max_length=config.get('max_length', 96),
        model_path=model_path,
        dropout_rate=config.get('dropout_rate', 0.2)
    )

    # 加载模型
    try:
        classifier.load_model()
        print(f"✓ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None

    # 预测
    start_time = time.time()
    predictions = classifier.predict(test_titles)
    pred_time = time.time() - start_time

    # 计算指标
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)

    # 获取特征向量（用于t-SNE）
    try:
        features = classifier.get_feature_vectors(test_titles)
    except:
        features = None

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred_time': pred_time,
        'predictions': predictions,
        'features': features,
        'config': config
    }

    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"精确率: {precision:.4f} ({precision*100:.2f}%)")
    print(f"召回率: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1分数: {f1:.4f} ({f1*100:.2f}%)")
    print(f"预测时间: {pred_time:.2f}秒")

    return results


def generate_comparison_plot(all_results, output_path):
    """生成模型对比图"""
    print(f"\n生成模型对比图...")

    # 准备数据
    model_names = [r['model_name'] for r in all_results]
    metrics = {
        'Accuracy': [r['accuracy'] for r in all_results],
        'Precision': [r['precision'] for r in all_results],
        'Recall': [r['recall'] for r in all_results],
        'F1 Score': [r['f1'] for r in all_results]
    }

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(model_names))
    width = 0.2

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i], alpha=0.8)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Stage4 BERT Optimization - Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 模型对比图已保存: {output_path}")


def generate_confusion_matrices(all_results, test_labels, output_path):
    """生成混淆矩阵图"""
    print(f"\n生成混淆矩阵图...")

    n_models = len(all_results)

    # 根据模型数量调整布局
    if n_models <= 3:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, result in enumerate(all_results):
        if idx >= len(axes):
            break

        cm = confusion_matrix(test_labels, result['predictions'])

        # 绘制混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=axes[idx], cbar=True, square=True)

        axes[idx].set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.4f}",
                           fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)

    # 隐藏多余的子图
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Stage4 BERT Optimization - Confusion Matrices',
                fontsize=14, fontweight='bold', y=0.98 if n_models <= 3 else 0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 混淆矩阵已保存: {output_path}")


def generate_tsne_plots(all_results, test_labels, output_dir):
    """生成t-SNE可视化图"""
    print(f"\n生成t-SNE可视化图...")

    for result in all_results:
        if result['features'] is None:
            print(f"⊘ {result['model_name']}: 无特征向量，跳过t-SNE")
            continue

        model_name = result['model_name']
        features = result['features']

        print(f"  处理 {model_name}...")

        # 采样（如果数据太大）
        if len(features) > 2000:
            indices = np.random.choice(len(features), 2000, replace=False)
            features_sample = features[indices]
            labels_sample = np.array(test_labels)[indices]
        else:
            features_sample = features
            labels_sample = np.array(test_labels)

        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features_sample)

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['#e74c3c', '#3498db']
        labels_text = ['Incorrect Title (0)', 'Correct Title (1)']

        for label_val in [0, 1]:
            mask = labels_sample == label_val
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=colors[label_val], label=labels_text[label_val],
                      alpha=0.6, s=20, edgecolors='w', linewidth=0.5)

        ax.set_title(f't-SNE Visualization - {model_name}\n'
                    f'Accuracy: {result["accuracy"]:.4f}, F1: {result["f1"]:.4f}',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=11)
        ax.set_ylabel('t-SNE Component 2', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存文件名
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        output_path = os.path.join(output_dir, f'tsne_{safe_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ {model_name} t-SNE图已保存")


def generate_text_report(all_results, output_path):
    """生成文本报告"""
    print(f"\n生成文本报告...")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("                Stage4 BERT Optimization - 训练结果\n")
        f.write("="*80 + "\n\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for result in all_results:
            f.write("-"*80 + "\n")
            f.write(f"模型: {result['model_name']}\n")
            f.write("-"*80 + "\n")
            f.write(f"配置: {json.dumps(result['config'], ensure_ascii=False, indent=2)}\n")
            f.write(f"准确率: {result['accuracy']:.4f}\n")
            f.write(f"精确率: {result['precision']:.4f}\n")
            f.write(f"召回率: {result['recall']:.4f}\n")
            f.write(f"F1分数: {result['f1']:.4f}\n")
            f.write(f"预测时间: {result['pred_time']:.2f}秒\n")
            f.write("\n")

        # 找出最佳模型
        best_f1 = max(all_results, key=lambda x: x['f1'])
        best_acc = max(all_results, key=lambda x: x['accuracy'])
        best_recall = max(all_results, key=lambda x: x['recall'])

        f.write("="*80 + "\n")
        f.write("                            最佳模型\n")
        f.write("="*80 + "\n\n")
        f.write(f"最高F1分数: {best_f1['model_name']}\n")
        f.write(f"  - F1: {best_f1['f1']:.4f}\n")
        f.write(f"  - Accuracy: {best_f1['accuracy']:.4f}\n")
        f.write(f"  - Recall: {best_f1['recall']:.4f}\n\n")

        f.write(f"最高准确率: {best_acc['model_name']}\n")
        f.write(f"  - Accuracy: {best_acc['accuracy']:.4f}\n")
        f.write(f"  - F1: {best_acc['f1']:.4f}\n\n")

        f.write(f"最高召回率: {best_recall['model_name']}\n")
        f.write(f"  - Recall: {best_recall['recall']:.4f}\n")
        f.write(f"  - F1: {best_recall['f1']:.4f}\n\n")

    print(f"✓ 文本报告已保存: {output_path}")


def main():
    """主函数"""

    print("\n" + "="*80)
    print("           Stage4 BERT Optimization - 模型评估")
    print("="*80)

    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 创建输出目录
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 加载测试数据
    print("\n加载测试数据...")
    _, _, test_titles, test_labels = TitleDataLoader.prepare_dataset(
        os.path.join(script_dir, 'data/positive.txt'),
        os.path.join(script_dir, 'data/negative.txt'),
        os.path.join(script_dir, 'data/testSet-1000.xlsx')
    )

    print(f"✓ 测试集: {len(test_titles)} 样本")

    # 定义要评估的模型（使用主目录的模型文件，这些是训练时保存的）
    # 注意：exp3_roberta_weighted训练失败（F1=0），已排除
    main_models_dir = '/home/u2023312337/task2/task2/models/experiments'
    models_to_evaluate = [
        {
            'name': 'BERT-base Baseline',
            'path': os.path.join(main_models_dir, 'exp1_bert_base_baseline_best.pt'),
            'config': {
                'model_name': 'bert-base',
                'max_length': 64,
                'loss_type': 'ce'
            }
        },
        {
            'name': 'SciBERT + Focal Loss',
            'path': os.path.join(main_models_dir, 'exp2_scibert_focal_best.pt'),
            'config': {
                'model_name': 'scibert',
                'max_length': 96,
                'loss_type': 'focal'
            }
        },
        {
            'name': 'SciBERT MaxLen128',
            'path': os.path.join(main_models_dir, 'exp5_scibert_maxlen128_best.pt'),
            'config': {
                'model_name': 'scibert',
                'max_length': 128,
                'loss_type': 'focal'
            }
        }
    ]

    # 评估所有模型
    all_results = []
    for model_info in models_to_evaluate:
        if not os.path.exists(model_info['path']):
            print(f"\n⊘ 模型文件不存在: {model_info['path']}")
            continue

        result = load_model_and_evaluate(
            model_info['path'],
            model_info['name'],
            model_info['config'],
            test_titles,
            test_labels
        )

        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\n❌ 没有成功评估的模型!")
        return

    print(f"\n✓ 成功评估 {len(all_results)} 个模型")

    # 生成输出文件
    print("\n" + "="*80)
    print("           生成输出文件")
    print("="*80)

    # 1. 模型对比图
    generate_comparison_plot(
        all_results,
        os.path.join(output_dir, 'model_comparison.png')
    )

    # 2. 混淆矩阵
    generate_confusion_matrices(
        all_results,
        test_labels,
        os.path.join(output_dir, 'confusion_matrices.png')
    )

    # 3. t-SNE可视化
    generate_tsne_plots(all_results, test_labels, output_dir)

    # 4. 文本报告
    generate_text_report(
        all_results,
        os.path.join(output_dir, 'training_results.txt')
    )

    print("\n" + "="*80)
    print("           评估完成!")
    print("="*80)
    print(f"\n所有输出文件已保存到: {output_dir}")
    print("\n生成的文件:")
    print("  - model_comparison.png      模型对比图")
    print("  - confusion_matrices.png    混淆矩阵")
    print("  - tsne_*.png                t-SNE可视化图")
    print("  - training_results.txt      详细结果报告")


if __name__ == "__main__":
    main()

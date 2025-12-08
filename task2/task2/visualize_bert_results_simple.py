"""
visualize_bert_results_simple.py
==================================
基于已有的实验结果JSON生成可视化（无需重新加载模型）
- 模型性能对比图
- 混淆矩阵（基于results.json中的指标推导）
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_experiment_results(results_path: str) -> List[Dict]:
    """加载实验结果JSON"""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def calculate_confusion_matrix_from_metrics(accuracy, precision, recall, n_samples):
    """
    根据评估指标反推混淆矩阵

    假设二分类平衡数据集: P ≈ N ≈ n_samples/2
    """
    # 假设正负样本各占一半
    P = n_samples // 2  # 实际正样本数
    N = n_samples - P   # 实际负样本数

    # 从recall计算TP
    TP = int(recall * P)
    FN = P - TP

    # 从precision计算FP
    # precision = TP / (TP + FP)
    # FP = TP / precision - TP
    if precision > 0:
        FP = int(TP / precision - TP)
    else:
        FP = 0

    TN = N - FP

    # 确保数值合理
    FP = max(0, FP)
    TN = max(0, TN)

    cm = np.array([[TN, FP], [FN, TP]])
    return cm


def plot_model_comparison(results: List[Dict], save_path: str):
    """生成模型性能对比图"""
    print("\n生成模型性能对比图...")

    # 过滤有效结果
    valid_results = [r for r in results if r['status'] == 'success' and r['accuracy'] > 0.5]

    if len(valid_results) == 0:
        print("❌ 没有有效的实验结果")
        return

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score'
    }

    models = [r['experiment_name'] for r in valid_results]
    data = {metric: [r[metric] for r in valid_results] for metric in metrics}

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
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=11)
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
                fontsize=11,
                fontweight='bold'
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {save_path}")
    plt.close()


def plot_confusion_matrices(results: List[Dict], save_path: str, n_test_samples=1000):
    """生成混淆矩阵可视化"""
    print("\n生成混淆矩阵...")

    # 过滤有效结果
    valid_results = [r for r in results if r['status'] == 'success' and r['accuracy'] > 0.5]

    if len(valid_results) == 0:
        print("❌ 没有有效的实验结果")
        return

    n_models = len(valid_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    fig.suptitle('Confusion Matrix Comparison - BERT Experiments',
                 fontsize=18, fontweight='bold', y=1.02)

    for idx, result in enumerate(valid_results):
        # 根据指标计算混淆矩阵
        cm = calculate_confusion_matrix_from_metrics(
            result['accuracy'],
            result['precision'],
            result['recall'],
            n_test_samples
        )

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
            f"{result['experiment_name']}\nAcc: {result['accuracy']:.4f} | F1: {result['f1']:.4f}",
            fontsize=14, fontweight='bold', pad=12
        )
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存: {save_path}")
    plt.close()


def plot_training_curves(results: List[Dict], output_dir: str):
    """为每个实验绘制训练曲线"""
    print("\n生成训练曲线...")

    valid_results = [r for r in results if r['status'] == 'success' and r['accuracy'] > 0.5]

    for result in valid_results:
        exp_name = result['experiment_name']
        history = result.get('training_history', {})

        if not history:
            print(f"⏭  {exp_name}: 无训练历史")
            continue

        print(f"  - {exp_name}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Training History: {exp_name}', fontsize=16, fontweight='bold')

        # Loss曲线
        ax = axes[0, 0]
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
            ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Loss', fontweight='bold')
            ax.set_title('Loss Curve', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        # Accuracy曲线
        ax = axes[0, 1]
        if 'train_acc' in history and 'val_acc' in history:
            epochs = range(1, len(history['train_acc']) + 1)
            ax.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2)
            ax.plot(epochs, history['val_acc'], 'r-s', label='Val Acc', linewidth=2)
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.set_title('Accuracy Curve', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        # F1曲线
        ax = axes[1, 0]
        if 'train_f1' in history and 'val_f1' in history:
            epochs = range(1, len(history['train_f1']) + 1)
            ax.plot(epochs, history['train_f1'], 'b-o', label='Train F1', linewidth=2)
            ax.plot(epochs, history['val_f1'], 'r-s', label='Val F1', linewidth=2)
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('F1 Score', fontweight='bold')
            ax.set_title('F1 Score Curve', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        # Recall曲线
        ax = axes[1, 1]
        if 'val_recall' in history:
            epochs = range(1, len(history['val_recall']) + 1)
            ax.plot(epochs, history['val_recall'], 'g-^', label='Val Recall', linewidth=2, markersize=8)
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Recall', fontweight='bold')
            ax.set_title('Recall Curve (Validation)', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"training_curve_{exp_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print("✓ 训练曲线生成完成")


def generate_summary_report(results: List[Dict], output_path: str):
    """生成总结报告"""
    print("\n生成总结报告...")

    valid_results = [r for r in results if r['status'] == 'success' and r['accuracy'] > 0.5]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(" BERT实验总结报告\n")
        f.write("="*100 + "\n\n")

        f.write("实验配置对比:\n")
        f.write("-"*100 + "\n")
        for result in valid_results:
            f.write(f"\n实验: {result['experiment_name']}\n")
            config = result['config']
            f.write(f"  模型: {config['model_name']}\n")
            f.write(f"  Max Length: {config['max_length']}\n")
            f.write(f"  Loss Type: {config['loss_type']}\n")
            f.write(f"  Epochs: {config['epochs']}\n")
            f.write(f"  Batch Size: {config['batch_size']}\n")
            f.write(f"  Learning Rate: {config['learning_rate']}\n")
            f.write(f"  Layer-wise LR: {'✓' if config.get('use_layer_wise_lr') else '✗'}\n")
            f.write(f"  Adversarial Training: {'✓' if config.get('use_adversarial') else '✗'}\n")

        f.write("\n\n")
        f.write("性能指标对比:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'实验名称':<30} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}\n")
        f.write("-"*100 + "\n")

        for result in valid_results:
            f.write(f"{result['experiment_name']:<30} "
                   f"{result['accuracy']:>12.4f} "
                   f"{result['precision']:>12.4f} "
                   f"{result['recall']:>12.4f} "
                   f"{result['f1']:>12.4f}\n")

        # 找出最佳模型
        best_f1 = max(valid_results, key=lambda x: x['f1'])
        best_recall = max(valid_results, key=lambda x: x['recall'])
        best_acc = max(valid_results, key=lambda x: x['accuracy'])

        f.write("\n\n")
        f.write("="*100 + "\n")
        f.write(" 最佳模型\n")
        f.write("="*100 + "\n")

        f.write(f"\n最高F1分数: {best_f1['experiment_name']}\n")
        f.write(f"  F1: {best_f1['f1']:.4f} ({best_f1['f1']*100:.2f}%)\n")
        f.write(f"  Accuracy: {best_f1['accuracy']:.4f} ({best_f1['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision: {best_f1['precision']:.4f} ({best_f1['precision']*100:.2f}%)\n")
        f.write(f"  Recall: {best_f1['recall']:.4f} ({best_f1['recall']*100:.2f}%)\n")

        f.write(f"\n最高Recall: {best_recall['experiment_name']}\n")
        f.write(f"  Recall: {best_recall['recall']:.4f} ({best_recall['recall']*100:.2f}%)\n")
        f.write(f"  F1: {best_recall['f1']:.4f} ({best_recall['f1']*100:.2f}%)\n")
        f.write(f"  Accuracy: {best_recall['accuracy']:.4f} ({best_recall['accuracy']*100:.2f}%)\n")

        f.write(f"\n最高Accuracy: {best_acc['experiment_name']}\n")
        f.write(f"  Accuracy: {best_acc['accuracy']:.4f} ({best_acc['accuracy']*100:.2f}%)\n")
        f.write(f"  F1: {best_acc['f1']:.4f} ({best_acc['f1']*100:.2f}%)\n")
        f.write(f"  Recall: {best_acc['recall']:.4f} ({best_acc['recall']*100:.2f}%)\n")

    print(f"✓ 总结报告已保存: {output_path}")


def main():
    """主函数"""
    print("\n" + "="*100)
    print(" BERT实验结果可视化（简化版）")
    print("="*100)

    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义路径
    results_path = os.path.join(script_dir, 'models/experiments/results.json')
    output_dir = os.path.join(script_dir, 'output/bert_experiments')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 检查结果文件
    if not os.path.exists(results_path):
        print(f"❌ 实验结果文件不存在: {results_path}")
        print("请先运行 run_bert_experiments.py")
        return

    # 加载实验结果
    print(f"\n加载实验结果: {results_path}")
    results = load_experiment_results(results_path)
    print(f"✓ 加载了 {len(results)} 个实验")

    # 显示实验摘要
    print("\n实验摘要:")
    print("-"*100)
    for r in results:
        status = "✓ 成功" if r['status'] == 'success' and r['accuracy'] > 0.5 else "❌ 失败"
        print(f"  {r['experiment_name']:<30} {status}  "
              f"Acc: {r['accuracy']:.4f}  F1: {r['f1']:.4f}  Recall: {r['recall']:.4f}")

    # 1. 生成模型对比图
    plot_model_comparison(
        results,
        os.path.join(output_dir, "model_comparison.png")
    )

    # 2. 生成混淆矩阵
    plot_confusion_matrices(
        results,
        os.path.join(output_dir, "confusion_matrices.png"),
        n_test_samples=1000
    )

    # 3. 生成训练曲线
    plot_training_curves(results, output_dir)

    # 4. 生成总结报告
    generate_summary_report(
        results,
        os.path.join(output_dir, "summary_report.txt")
    )

    print("\n" + "="*100)
    print(" ✅ 所有可视化生成完成!")
    print("="*100)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    for file in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file} ({file_size:.1f} KB)")


if __name__ == "__main__":
    main()

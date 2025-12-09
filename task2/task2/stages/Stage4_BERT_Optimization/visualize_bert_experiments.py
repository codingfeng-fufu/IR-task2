"""
visualize_bert_experiments.py
==============================
为已训练的BERT实验模型生成可视化结果
- t-SNE特征空间可视化
- 模型性能对比图
- 混淆矩阵
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from typing import List, Dict
from tqdm import tqdm

# 导入项目模块
from data_loader import DataLoader as TitleDataLoader
from train_bert_optimized_v2 import OptimizedBERTClassifier
from visualizer import ResultVisualizer

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)


def load_experiment_results(results_path: str) -> List[Dict]:
    """加载实验结果JSON"""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def load_model_and_extract_features(
    experiment_name: str,
    config: Dict,
    test_titles: List[str],
    script_dir: str,
    device: str = 'cuda'
) -> np.ndarray:
    """
    加载模型并提取测试集特征向量

    Args:
        experiment_name: 实验名称
        config: 模型配置
        test_titles: 测试集标题
        script_dir: 脚本目录
        device: 设备

    Returns:
        特征向量数组 (n_samples, hidden_dim)
    """
    print(f"\n加载模型: {experiment_name}")

    # 使用_best.pt模型（最佳验证集性能）
    model_path = os.path.join(script_dir, f"models/experiments/{experiment_name}_best.pt")
    if not os.path.exists(model_path):
        # 如果没有_best.pt，使用普通模型
        model_path = os.path.join(script_dir, f"models/experiments/{experiment_name}.pt")

    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        return None

    # 创建分类器
    classifier = OptimizedBERTClassifier(
        model_name=config.get('model_name', 'scibert'),
        max_length=config.get('max_length', 96),
        model_path=model_path,
        dropout_rate=config.get('dropout_rate', 0.2)
    )

    # 加载模型
    try:
        classifier.load_model()
        print(f"✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    # 提取特征
    print(f"提取特征向量 (共 {len(test_titles)} 个样本)...")
    features = classifier.get_feature_vectors(test_titles)
    print(f"✓ 特征提取完成: {features.shape}")

    return features


def generate_tsne_visualizations(
    experiments: List[Dict],
    test_titles: List[str],
    test_labels: List[int],
    script_dir: str,
    output_dir: str
):
    """
    为每个成功的实验生成t-SNE可视化

    Args:
        experiments: 实验结果列表
        test_titles: 测试集标题
        test_labels: 测试集标签
        script_dir: 脚本目录
        output_dir: 输出目录
    """
    print("\n" + "="*100)
    print(" 生成 t-SNE 可视化")
    print("="*100)

    visualizer = ResultVisualizer()

    for exp in experiments:
        exp_name = exp['experiment_name']

        # 跳过失败的实验
        if exp['status'] != 'success':
            print(f"\n⏭  跳过失败的实验: {exp_name}")
            continue

        # 跳过RoBERTa崩溃实验
        if 'roberta' in exp_name.lower() and exp['accuracy'] < 0.5:
            print(f"\n⏭  跳过训练崩溃的实验: {exp_name}")
            continue

        # 提取特征
        features = load_model_and_extract_features(
            exp_name,
            exp['config'],
            test_titles,
            script_dir
        )

        if features is None:
            print(f"⏭  跳过: {exp_name}")
            continue

        # 生成t-SNE可视化
        save_path = os.path.join(output_dir, f"tsne_{exp_name}.png")
        model_display_name = f"{exp['config']['model_name']} ({exp_name})"

        visualizer.visualize_embeddings_tsne(
            vectors=features,
            labels=test_labels,
            title=model_display_name,
            save_path=save_path,
            perplexity=30,
            n_iter=1000
        )


def generate_model_comparison(
    experiments: List[Dict],
    output_dir: str
):
    """
    生成模型性能对比图

    Args:
        experiments: 实验结果列表
        output_dir: 输出目录
    """
    print("\n" + "="*100)
    print(" 生成模型性能对比图")
    print("="*100)

    # 过滤成功的实验（排除训练崩溃的RoBERTa）
    valid_experiments = [
        exp for exp in experiments
        if exp['status'] == 'success' and exp['accuracy'] > 0.5
    ]

    if len(valid_experiments) == 0:
        print("❌ 没有有效的实验结果")
        return

    # 准备数据
    results = []
    for exp in valid_experiments:
        results.append({
            'model': exp['experiment_name'],
            'accuracy': exp['accuracy'],
            'precision': exp['precision'],
            'recall': exp['recall'],
            'f1': exp['f1'],
            'f1_macro': exp['f1'],  # 二分类时macro和micro相同
            'f1_micro': exp['f1'],
            'confusion_matrix': None  # 将在下一步生成
        })

    # 生成对比图
    visualizer = ResultVisualizer()
    save_path = os.path.join(output_dir, "bert_experiments_comparison.png")
    visualizer.plot_comparison(results, save_path=save_path)


def generate_confusion_matrices(
    experiments: List[Dict],
    test_titles: List[str],
    test_labels: List[int],
    script_dir: str,
    output_dir: str
):
    """
    生成混淆矩阵可视化

    Args:
        experiments: 实验结果列表
        test_titles: 测试集标题
        test_labels: 测试集标签
        script_dir: 脚本目录
        output_dir: 输出目录
    """
    print("\n" + "="*100)
    print(" 生成混淆矩阵")
    print("="*100)

    # 过滤成功的实验
    valid_experiments = [
        exp for exp in experiments
        if exp['status'] == 'success' and exp['accuracy'] > 0.5
    ]

    if len(valid_experiments) == 0:
        print("❌ 没有有效的实验结果")
        return

    results = []

    for exp in valid_experiments:
        exp_name = exp['experiment_name']
        print(f"\n处理: {exp_name}")

        # 加载模型
        model_path = os.path.join(script_dir, f"models/experiments/{exp_name}_best.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(script_dir, f"models/experiments/{exp_name}.pt")

        if not os.path.exists(model_path):
            print(f"⚠️  模型文件不存在: {model_path}")
            continue

        # 创建分类器
        classifier = OptimizedBERTClassifier(
            model_name=exp['config'].get('model_name', 'scibert'),
            max_length=exp['config'].get('max_length', 96),
            model_path=model_path,
            dropout_rate=exp['config'].get('dropout_rate', 0.2)
        )

        try:
            classifier.load_model()
            print(f"✓ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            continue

        # 预测
        print(f"预测测试集...")
        predictions = classifier.predict(test_titles)

        # 计算混淆矩阵
        cm = confusion_matrix(test_labels, predictions)
        print(f"✓ 混淆矩阵:\n{cm}")

        results.append({
            'model': exp_name,
            'accuracy': exp['accuracy'],
            'precision': exp['precision'],
            'recall': exp['recall'],
            'f1': exp['f1'],
            'f1_macro': exp['f1'],
            'f1_micro': exp['f1'],
            'confusion_matrix': cm
        })

    # 生成混淆矩阵可视化
    if len(results) > 0:
        visualizer = ResultVisualizer()
        save_path = os.path.join(output_dir, "bert_experiments_confusion_matrices.png")
        visualizer.plot_confusion_matrices(results, save_path=save_path)


def main():
    """主函数"""
    print("\n" + "="*100)
    print(" BERT实验结果可视化")
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
    experiments = load_experiment_results(results_path)
    print(f"✓ 加载了 {len(experiments)} 个实验")

    # 加载测试数据
    print("\n加载测试数据...")
    _, _, test_titles, test_labels = TitleDataLoader.prepare_dataset(
        os.path.join(script_dir, 'data/positive.txt'),
        os.path.join(script_dir, 'data/negative.txt'),
        os.path.join(script_dir, 'data/testSet-1000.xlsx')
    )
    print(f"✓ 测试集: {len(test_titles)} 个样本")

    # 1. 生成t-SNE可视化
    generate_tsne_visualizations(
        experiments,
        test_titles,
        test_labels,
        script_dir,
        output_dir
    )

    # 2. 生成模型对比图
    generate_model_comparison(experiments, output_dir)

    # 3. 生成混淆矩阵
    generate_confusion_matrices(
        experiments,
        test_titles,
        test_labels,
        script_dir,
        output_dir
    )

    print("\n" + "="*100)
    print(" ✅ 所有可视化生成完成!")
    print("="*100)
    print(f"\n输出目录: {output_dir}")
    print("\n生成的文件:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png'):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file} ({file_size:.1f} KB)")


if __name__ == "__main__":
    main()

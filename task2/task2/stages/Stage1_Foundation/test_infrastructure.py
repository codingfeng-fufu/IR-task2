#!/usr/bin/env python3
"""Stage1 Foundation - 基础设施测试脚本

本阶段不包含模型训练，主要测试基础设施功能：
- 数据加载
- 评估模块
- 可视化功能
- 环境检查

用法:
    # 完整测试
    python test_infrastructure.py

    # 仅测试数据加载
    python test_infrastructure.py --test data

    # 仅测试可视化
    python test_infrastructure.py --test viz
"""

import os
import sys
import argparse
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_data_path, get_model_path, get_output_path
from data_loader import DataLoader
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer


def print_banner():
    """打印横幅."""
    print("\n" + "="*80)
    print(" " * 25 + "Stage1 Foundation - 基础设施测试")
    print("="*80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def test_data_loading():
    """测试数据加载功能."""
    print("[测试 1/4] 数据加载模块")
    print("-" * 80)

    try:
        # 加载训练数据
        train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
            get_data_path('positive.txt'),
            get_data_path('negative.txt'),
            get_data_path('testSet-1000.xlsx')
        )

        print(f"✓ 训练数据: {len(train_titles)} 个样本")
        print(f"✓ 测试数据: {len(test_titles)} 个样本")
        print(f"✓ 正样本比例: {sum(train_labels)/len(train_labels):.2%}")

        # 显示示例
        print(f"\n示例标题:")
        for i in range(min(3, len(test_titles))):
            label = "正样本" if test_labels[i] == 1 else "负样本"
            print(f"  [{label}] {test_titles[i][:60]}...")

        print("\n✓ 数据加载测试通过!\n")
        return train_titles, train_labels, test_titles, test_labels

    except Exception as e:
        print(f"\n✗ 数据加载测试失败: {e}\n")
        return None, None, None, None


def test_evaluator(test_labels):
    """测试评估模块."""
    print("[测试 2/4] 评估模块")
    print("-" * 80)

    try:
        evaluator = ModelEvaluator()

        # 生成模拟预测（70%准确率）
        np.random.seed(42)
        predictions = test_labels.copy()
        num_errors = int(len(predictions) * 0.3)
        error_indices = np.random.choice(len(predictions), num_errors, replace=False)
        predictions[error_indices] = 1 - predictions[error_indices]

        # 评估
        result = evaluator.evaluate_model(
            test_labels,
            predictions,
            "Test Model",
            verbose=False
        )

        print(f"✓ 模拟模型性能:")
        print(f"  准确率: {result['accuracy']:.4f}")
        print(f"  精确率: {result['precision']:.4f}")
        print(f"  召回率: {result['recall']:.4f}")
        print(f"  F1分数: {result['f1']:.4f}")
        print(f"  混淆矩阵形状: {result['confusion_matrix'].shape}")

        # 测试多模型对比
        result2 = result.copy()
        result2['model_name'] = 'Test Model 2'
        result2['accuracy'] = 0.65

        print(f"\n✓ 测试多模型对比:")
        ModelEvaluator.compare_models([result, result2])

        print("\n✓ 评估模块测试通过!\n")
        return [result, result2]

    except Exception as e:
        print(f"\n✗ 评估模块测试失败: {e}\n")
        return None


def test_visualizer(results):
    """测试可视化功能."""
    print("[测试 3/4] 可视化模块")
    print("-" * 80)

    try:
        visualizer = ResultVisualizer()
        output_dir = get_output_path()

        # 测试性能对比图
        print("生成性能对比图...")
        visualizer.plot_comparison(
            results,
            save_path=os.path.join(output_dir, 'test_comparison.png')
        )
        print(f"✓ 保存到: {output_dir}/test_comparison.png")

        # 测试混淆矩阵
        print("生成混淆矩阵...")
        visualizer.plot_confusion_matrices(
            results,
            save_path=os.path.join(output_dir, 'test_confusion_matrices.png')
        )
        print(f"✓ 保存到: {output_dir}/test_confusion_matrices.png")

        # 测试t-SNE（使用小样本）
        print("生成t-SNE可视化（小样本测试）...")
        np.random.seed(42)
        sample_vectors = np.random.randn(100, 50)  # 100个样本，50维特征
        sample_labels = np.random.randint(0, 2, 100)

        visualizer.plot_tsne(
            sample_vectors,
            sample_labels,
            model_name="Test_tSNE",
            save_path=os.path.join(output_dir, 'test_tsne.png')
        )
        print(f"✓ 保存到: {output_dir}/test_tsne.png")

        print("\n✓ 可视化模块测试通过!\n")
        return True

    except Exception as e:
        print(f"\n✗ 可视化模块测试失败: {e}\n")
        return False


def test_environment():
    """测试环境配置."""
    print("[测试 4/4] 环境检查")
    print("-" * 80)

    try:
        import torch
        import transformers
        import sklearn
        import gensim
        import pandas
        import matplotlib
        import seaborn

        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Transformers: {transformers.__version__}")
        print(f"✓ Scikit-learn: {sklearn.__version__}")
        print(f"✓ Gensim: {gensim.__version__}")
        print(f"✓ Pandas: {pandas.__version__}")
        print(f"✓ Matplotlib: {matplotlib.__version__}")
        print(f"✓ Seaborn: {seaborn.__version__}")

        # CUDA检查
        if torch.cuda.is_available():
            print(f"\n✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print(f"\n⚠ CUDA不可用，将使用CPU")

        # 路径检查
        print(f"\n路径配置:")
        print(f"  数据目录: {get_data_path()}")
        print(f"  模型目录: {get_model_path()}")
        print(f"  输出目录: {get_output_path()}")

        print("\n✓ 环境检查通过!\n")
        return True

    except ImportError as e:
        print(f"\n✗ 环境检查失败: {e}")
        print("请确保已安装所有依赖包\n")
        return False


def main():
    """主函数."""
    parser = argparse.ArgumentParser(
        description='Stage1 Foundation 基础设施测试',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--test',
        type=str,
        choices=['all', 'data', 'eval', 'viz', 'env'],
        default='all',
        help='选择要测试的模块（all=全部, data=数据加载, eval=评估, viz=可视化, env=环境）'
    )

    args = parser.parse_args()

    # 打印横幅
    print_banner()

    success_count = 0
    total_tests = 0

    # 数据加载测试
    train_titles, train_labels, test_titles, test_labels = None, None, None, None
    if args.test in ['all', 'data']:
        total_tests += 1
        train_titles, train_labels, test_titles, test_labels = test_data_loading()
        if test_labels is not None:
            success_count += 1

    # 评估模块测试
    results = None
    if args.test in ['all', 'eval']:
        total_tests += 1
        if test_labels is None:
            # 如果没有真实数据，生成模拟数据
            test_labels = np.random.randint(0, 2, 1000)
        results = test_evaluator(test_labels)
        if results is not None:
            success_count += 1

    # 可视化测试
    if args.test in ['all', 'viz']:
        total_tests += 1
        if results is None:
            # 生成模拟结果
            results = [
                {
                    'model_name': 'Model1',
                    'accuracy': 0.70,
                    'precision': 0.72,
                    'recall': 0.68,
                    'f1': 0.70,
                    'confusion_matrix': np.array([[45, 15], [10, 30]])
                },
                {
                    'model_name': 'Model2',
                    'accuracy': 0.65,
                    'precision': 0.67,
                    'recall': 0.63,
                    'f1': 0.65,
                    'confusion_matrix': np.array([[40, 20], [15, 25]])
                }
            ]
        if test_visualizer(results):
            success_count += 1

    # 环境检查
    if args.test in ['all', 'env']:
        total_tests += 1
        if test_environment():
            success_count += 1

    # 打印总结
    print("="*80)
    print(" " * 30 + "测试完成")
    print("="*80)
    print(f"\n通过: {success_count}/{total_tests} 项测试")

    if success_count == total_tests:
        print("\n✓ 所有测试通过！Stage1基础设施工作正常。")
    else:
        print(f"\n⚠ {total_tests - success_count} 项测试失败，请检查错误信息。")

    print(f"\n输出文件位置: {get_output_path()}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

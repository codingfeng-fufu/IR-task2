#!/usr/bin/env python3
"""训练脚本 - Baseline Simple 版本

用法:
    # 训练所有模型（默认）
    python train.py

    # 仅训练特定模型
    python train.py --model nb           # 朴素贝叶斯
    python train.py --model w2v          # Word2Vec+SVM
    python train.py --model bert         # BERT

    # 快速测试模式（减少训练数据）
    python train.py --quick --max-samples 5000

    # BERT快速测试（仅1个epoch）
    python train.py --model bert --quick --epochs 1
"""

import os
import sys
import argparse
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_training_data, load_test_data
from naive_bayes import SimpleNaiveBayes
from word2vec_svm import SimpleWord2VecSVM
from bert_classifier import SimpleBERT
from evaluator import evaluate_model, compare_models
from visualizer import plot_model_comparison, plot_confusion_matrices, plot_tsne
from config import get_data_path, get_model_path, get_output_path


def print_banner():
    """打印横幅."""
    print("\n" + "="*80)
    print(" " * 25 + "Baseline Simple - 训练脚本")
    print(" " * 20 + "Academic Title Classification")
    print("="*80)
    print(f"\n训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def load_data(max_samples=None):
    """加载训练和测试数据.

    Args:
        max_samples: 如果指定，限制训练样本数量（用于快速测试）

    Returns:
        train_titles, train_labels, test_titles, test_labels
    """
    print("[1] 加载数据")
    print("-" * 80)

    # 加载训练数据
    train_titles, train_labels = load_training_data(
        get_data_path('positive.txt'),
        get_data_path('negative.txt')
    )

    # 如果指定了最大样本数，进行随机采样
    if max_samples and max_samples < len(train_titles):
        import random
        indices = list(range(len(train_titles)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        train_titles = [train_titles[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        print(f"⚠️  快速测试模式: 使用 {len(train_titles)} 个训练样本")
    else:
        print(f"✓ 加载训练数据: {len(train_titles)} 个样本")

    # 加载测试数据
    test_titles, test_labels = load_test_data(
        get_data_path('testSet-1000.xlsx')
    )
    print(f"✓ 加载测试数据: {len(test_titles)} 个样本")
    print()

    return train_titles, train_labels, test_titles, test_labels


def train_naive_bayes(train_titles, train_labels, test_titles, test_labels):
    """训练朴素贝叶斯模型.

    Returns:
        classifier, predictions, results, training_time
    """
    print("[2] 训练朴素贝叶斯分类器")
    print("-" * 80)

    start_time = time.time()

    # 初始化和训练
    classifier = SimpleNaiveBayes()
    classifier.train(train_titles, train_labels)

    training_time = time.time() - start_time
    print(f"\n✓ 训练完成，耗时: {training_time:.2f} 秒")

    # 保存模型
    model_path = get_model_path('naive_bayes.pkl')
    classifier.save_model(model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 评估
    print("\n评估模型...")
    predictions = classifier.predict(test_titles)
    results = evaluate_model(test_labels, predictions, "Naive Bayes")

    print(f"\n📊 性能指标:")
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  精确率: {results['precision']:.4f}")
    print(f"  召回率: {results['recall']:.4f}")
    print(f"  F1分数: {results['f1']:.4f}")
    print()

    return classifier, predictions, results, training_time


def train_word2vec_svm(train_titles, train_labels, test_titles, test_labels):
    """训练Word2Vec+SVM模型.

    Returns:
        classifier, predictions, results, training_time
    """
    print("[3] 训练 Word2Vec + SVM 分类器")
    print("-" * 80)

    start_time = time.time()

    # 初始化和训练
    classifier = SimpleWord2VecSVM(vector_size=100, window=5)
    classifier.train(train_titles, train_labels)

    training_time = time.time() - start_time
    print(f"\n✓ 训练完成，耗时: {training_time:.2f} 秒 ({training_time/60:.1f} 分钟)")

    # 保存模型
    model_path_prefix = get_model_path('word2vec_svm')
    classifier.save_model(model_path_prefix)
    print(f"✓ 模型已保存: {model_path_prefix}_*")

    # 评估
    print("\n评估模型...")
    predictions = classifier.predict(test_titles)
    results = evaluate_model(test_labels, predictions, "Word2Vec + SVM")

    print(f"\n📊 性能指标:")
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  精确率: {results['precision']:.4f}")
    print(f"  召回率: {results['recall']:.4f}")
    print(f"  F1分数: {results['f1']:.4f}")
    print()

    return classifier, predictions, results, training_time


def train_bert(train_titles, train_labels, test_titles, test_labels,
               epochs=3, batch_size=16, max_length=64):
    """训练BERT模型.

    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        max_length: 最大序列长度

    Returns:
        classifier, predictions, results, training_time
    """
    print("[4] 训练 BERT 分类器")
    print("-" * 80)
    print(f"配置: epochs={epochs}, batch_size={batch_size}, max_length={max_length}")
    print()

    start_time = time.time()

    # 初始化和训练
    classifier = SimpleBERT(model_name='bert-base-uncased', max_length=max_length)
    classifier.train(train_titles, train_labels, epochs=epochs, batch_size=batch_size)

    training_time = time.time() - start_time
    print(f"\n✓ 训练完成，耗时: {training_time:.2f} 秒 ({training_time/60:.1f} 分钟)")

    # 保存模型
    model_path = get_model_path('bert.pt')
    classifier.save_model(model_path)
    print(f"✓ 模型已保存: {model_path}")

    # 评估
    print("\n评估模型...")
    predictions = classifier.predict(test_titles, batch_size=batch_size)
    results = evaluate_model(test_labels, predictions, "BERT")

    print(f"\n📊 性能指标:")
    print(f"  准确率: {results['accuracy']:.4f}")
    print(f"  精确率: {results['precision']:.4f}")
    print(f"  召回率: {results['recall']:.4f}")
    print(f"  F1分数: {results['f1']:.4f}")
    print()

    return classifier, predictions, results, training_time


def save_results_summary(all_results, output_dir):
    """保存结果摘要到文本文件.

    Args:
        all_results: 包含所有模型结果的字典列表
        output_dir: 输出目录
    """
    output_file = os.path.join(output_dir, 'training_results.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 25 + "Baseline Simple - 训练结果\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for result in all_results:
            f.write("-" * 80 + "\n")
            f.write(f"模型: {result['model_name']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"准确率: {result['accuracy']:.4f}\n")
            f.write(f"精确率: {result['precision']:.4f}\n")
            f.write(f"召回率: {result['recall']:.4f}\n")
            f.write(f"F1分数: {result['f1']:.4f}\n")
            if 'training_time' in result:
                f.write(f"训练时间: {result['training_time']:.2f} 秒\n")
            f.write("\n")

    print(f"✓ 结果已保存: {output_file}")


def main():
    """主函数."""
    parser = argparse.ArgumentParser(
        description='Baseline Simple 训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py                          # 训练所有模型
  python train.py --model nb               # 仅训练朴素贝叶斯
  python train.py --model bert --quick     # BERT快速测试
  python train.py --max-samples 10000      # 使用10K样本训练
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['nb', 'w2v', 'bert', 'all'],
        default='all',
        help='选择要训练的模型 (nb=朴素贝叶斯, w2v=Word2Vec+SVM, bert=BERT, all=全部)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速测试模式（减少训练样本和epochs）'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大训练样本数（用于快速测试）'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='BERT训练轮数（默认3）'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='BERT批次大小（默认16）'
    )

    args = parser.parse_args()

    # 打印横幅
    print_banner()

    # 快速测试模式配置
    if args.quick:
        if args.max_samples is None:
            args.max_samples = 5000
        if args.epochs == 3:  # 如果没有手动指定epochs
            args.epochs = 1
        print("⚡ 快速测试模式")
        print(f"  最大样本数: {args.max_samples}")
        print(f"  BERT epochs: {args.epochs}")
        print()

    # 加载数据
    train_titles, train_labels, test_titles, test_labels = load_data(
        max_samples=args.max_samples
    )

    # 创建输出目录
    from config import OUTPUT_DIR, MODELS_DIR
    output_dir = OUTPUT_DIR
    models_dir = MODELS_DIR

    # 存储所有结果
    all_results = []
    total_start_time = time.time()

    # 训练模型
    if args.model in ['nb', 'all']:
        _, _, results, train_time = train_naive_bayes(
            train_titles, train_labels, test_titles, test_labels
        )
        results['training_time'] = train_time
        all_results.append(results)

    if args.model in ['w2v', 'all']:
        _, _, results, train_time = train_word2vec_svm(
            train_titles, train_labels, test_titles, test_labels
        )
        results['training_time'] = train_time
        all_results.append(results)

    if args.model in ['bert', 'all']:
        _, _, results, train_time = train_bert(
            train_titles, train_labels, test_titles, test_labels,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        results['training_time'] = train_time
        all_results.append(results)

    total_time = time.time() - total_start_time

    # 对比所有模型（如果训练了多个）
    if len(all_results) > 1:
        print("\n" + "="*80)
        print(" " * 30 + "模型对比")
        print("="*80 + "\n")
        compare_models(all_results)

        # 生成可视化
        print("\n生成可视化...")
        plot_model_comparison(all_results, output_dir)
        plot_confusion_matrices(all_results, output_dir)
        print(f"✓ 可视化已保存到: {output_dir}/")

    # 生成 t-SNE 可视化（为每个训练的模型）
    print("\n" + "="*80)
    print(" " * 28 + "生成 t-SNE 可视化")
    print("="*80 + "\n")

    classifiers_dict = {}
    if args.model in ['nb', 'all']:
        nb_classifier = SimpleNaiveBayes()
        nb_classifier.load_model(get_model_path('naive_bayes.pkl'))
        classifiers_dict['Naive_Bayes'] = nb_classifier

    if args.model in ['w2v', 'all']:
        w2v_classifier = SimpleWord2VecSVM(vector_size=100, window=5)
        w2v_classifier.load_model(get_model_path('word2vec_svm'))
        classifiers_dict['Word2Vec_SVM'] = w2v_classifier

    if args.model in ['bert', 'all']:
        bert_classifier = SimpleBERT(model_name='bert-base-uncased', max_length=64)
        bert_classifier.load_model(get_model_path('bert.pt'))
        classifiers_dict['BERT'] = bert_classifier

    # 为每个模型生成 t-SNE 图
    for model_name, classifier in classifiers_dict.items():
        try:
            feature_vectors = classifier.get_feature_vectors(test_titles)
            plot_tsne(feature_vectors, test_labels, model_name, output_dir)
        except Exception as e:
            print(f"⚠️  为 {model_name} 生成 t-SNE 图失败: {str(e)}")

    # 保存结果摘要
    save_results_summary(all_results, output_dir)

    # 打印总结
    print("\n" + "="*80)
    print(" " * 30 + "训练完成！")
    print("="*80)
    print(f"\n总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"\n模型保存位置: {models_dir}/")
    print(f"结果保存位置: {output_dir}/")
    print("\n生成的文件:")
    if len(all_results) > 1:
        print(f"  - {output_dir}/model_comparison.png")
        print(f"  - {output_dir}/confusion_matrices.png")
    for model_name in classifiers_dict.keys():
        print(f"  - {output_dir}/tsne_{model_name}.png")
    print(f"  - {output_dir}/training_results.txt")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage2 Traditional Models - 训练脚本

本阶段实现三种基础模型:
- 朴素贝叶斯 (73.46%)
- Word2Vec + SVM (82.99%)
- BERT (87.91%)

用法:
    python train.py                      # 训练所有模型
    python train.py --model nb           # 仅朴素贝叶斯
    python train.py --model w2v          # 仅Word2Vec+SVM
    python train.py --model bert         # 仅BERT
    python train.py --quick              # 快速测试
"""

import os
import sys
import argparse
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# 首先导入Stage2自己的config（必须在导入Stage1之前）
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Stage2目录优先
import config as stage2_config

# 然后添加Stage1到路径以访问基础设施
sys.path.insert(0, os.path.join(current_dir, '..', 'Stage1_Foundation'))

from data_loader import DataLoader
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
from naive_bayes_classifier import NaiveBayesClassifier
from word2vec_svm_classifier import Word2VecSVMClassifier
from bert_classifier import BERTClassifier

# 使用Stage2的配置函数
get_data_path = stage2_config.get_data_path
get_model_path = stage2_config.get_model_path
get_output_path = stage2_config.get_output_path


def print_banner():
    print("\n" + "="*80)
    print(" " * 25 + "Stage2 Traditional Models - 训练")
    print("="*80)
    print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def load_data(max_samples=None):
    print("[1] 加载数据")
    print("-" * 80)

    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        get_data_path('positive.txt'),
        get_data_path('negative.txt'),
        get_data_path('testSet-1000.xlsx')
    )

    if max_samples and max_samples < len(train_titles):
        import random
        indices = list(range(len(train_titles)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        train_titles = [train_titles[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        print(f"⚠ 快速测试模式: {len(train_titles)} 样本")
    else:
        print(f"✓ 训练数据: {len(train_titles)} 样本")

    print(f"✓ 测试数据: {len(test_titles)} 样本\n")
    return train_titles, train_labels, test_titles, test_labels


def train_naive_bayes(train_titles, train_labels, test_titles, test_labels):
    print("[2] 训练朴素贝叶斯 (V1 - 基础版)")
    print("-" * 80)

    start_time = time.time()

    classifier = NaiveBayesClassifier(
        max_features=5000,
        ngram_range=(1, 2),
        model_path=get_model_path('naive_bayes_model.pkl')
    )
    classifier.train(train_titles, train_labels)

    train_time = time.time() - start_time
    print(f"\n✓ 训练完成: {train_time:.2f}秒")

    predictions = classifier.predict(test_titles)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "Naive Bayes V1", verbose=False)

    print(f"\n📊 性能 (预期 ~73%)")
    print(f"  准确率: {result['accuracy']:.4f}")
    print(f"  F1分数: {result['f1']:.4f}\n")

    result['training_time'] = train_time
    return result


def train_word2vec_svm(train_titles, train_labels, test_titles, test_labels):
    print("[3] 训练 Word2Vec + SVM")
    print("-" * 80)

    start_time = time.time()

    classifier = Word2VecSVMClassifier(
        vector_size=100,
        window=5,
        model_path=get_model_path('word2vec_svm_model')
    )
    classifier.train(train_titles, train_labels)

    train_time = time.time() - start_time
    print(f"\n✓ 训练完成: {train_time:.2f}秒 ({train_time/60:.1f}分钟)")

    predictions = classifier.predict(test_titles)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "Word2Vec + SVM", verbose=False)

    print(f"\n📊 性能 (预期 ~83%)")
    print(f"  准确率: {result['accuracy']:.4f}")
    print(f"  F1分数: {result['f1']:.4f}\n")

    result['training_time'] = train_time
    return result


def train_bert(train_titles, train_labels, test_titles, test_labels, epochs=3, batch_size=16):
    print("[4] 训练 BERT (基础版)")
    print("-" * 80)
    print(f"配置: epochs={epochs}, batch_size={batch_size}\n")

    start_time = time.time()

    classifier = BERTClassifier(
        model_name='bert-base-uncased',
        max_length=64,
        model_path=get_model_path('best_bert_model.pt')
    )
    classifier.train(train_titles, train_labels, epochs=epochs, batch_size=batch_size)

    train_time = time.time() - start_time
    print(f"\n✓ 训练完成: {train_time:.2f}秒 ({train_time/60:.1f}分钟)")

    predictions = classifier.predict(test_titles, batch_size=batch_size)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "BERT", verbose=False)

    print(f"\n📊 性能 (预期 ~88%)")
    print(f"  准确率: {result['accuracy']:.4f}")
    print(f"  F1分数: {result['f1']:.4f}\n")

    result['training_time'] = train_time
    return result


def main():
    parser = argparse.ArgumentParser(description='Stage2 Traditional Models 训练脚本')
    parser.add_argument('--model', type=str, choices=['nb', 'w2v', 'bert', 'all'], default='all')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=3, help='BERT训练轮数')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    if args.quick:
        args.max_samples = args.max_samples or 5000
        args.epochs = 1
        print("⚡ 快速测试模式")

    print_banner()

    train_titles, train_labels, test_titles, test_labels = load_data(args.max_samples)

    all_results = []
    total_start = time.time()

    if args.model in ['nb', 'all']:
        result = train_naive_bayes(train_titles, train_labels, test_titles, test_labels)
        all_results.append(result)

    if args.model in ['w2v', 'all']:
        result = train_word2vec_svm(train_titles, train_labels, test_titles, test_labels)
        all_results.append(result)

    if args.model in ['bert', 'all']:
        result = train_bert(train_titles, train_labels, test_titles, test_labels,
                           epochs=args.epochs, batch_size=args.batch_size)
        all_results.append(result)

    total_time = time.time() - total_start

    if len(all_results) > 1:
        print("\n" + "="*80)
        print(" " * 30 + "模型对比")
        print("="*80 + "\n")
        ModelEvaluator.compare_models(all_results)

        visualizer = ResultVisualizer()
        visualizer.plot_comparison(all_results, save_path=get_output_path('comparison.png'))
        visualizer.plot_confusion_matrices(all_results, save_path=get_output_path('confusion_matrices.png'))

    # 生成 t-SNE 可视化
    print("\n" + "="*80)
    print(" " * 28 + "生成 t-SNE 可视化")
    print("="*80 + "\n")

    classifiers_dict = {}
    if args.model in ['nb', 'all']:
        nb_classifier = NaiveBayesClassifier(
            max_features=5000,
            ngram_range=(1, 2),
            model_path=get_model_path('naive_bayes_model.pkl')
        )
        nb_classifier.load_model()
        classifiers_dict['Naive_Bayes'] = nb_classifier

    if args.model in ['w2v', 'all']:
        w2v_classifier = Word2VecSVMClassifier(
            vector_size=100,
            window=5,
            model_path=get_model_path('word2vec_svm_model')
        )
        w2v_classifier.load_model()
        classifiers_dict['Word2Vec_SVM'] = w2v_classifier

    if args.model in ['bert', 'all']:
        bert_classifier = BERTClassifier(
            model_name='bert-base-uncased',
            max_length=64,
            model_path=get_model_path('best_bert_model.pt')
        )
        bert_classifier.load_model()
        classifiers_dict['BERT'] = bert_classifier

    # 为每个模型生成 t-SNE 图
    visualizer = ResultVisualizer()
    for model_name, classifier in classifiers_dict.items():
        try:
            feature_vectors = classifier.get_feature_vectors(test_titles)
            visualizer.visualize_embeddings_tsne(
                feature_vectors,
                test_labels,
                model_name,
                save_path=get_output_path(f'tsne_{model_name}.png')
            )
        except Exception as e:
            print(f"⚠️  为 {model_name} 生成 t-SNE 图失败: {str(e)}")

    print("\n" + "="*80)
    print(" " * 30 + "训练完成")
    print("="*80)
    print(f"\n总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"模型保存: {get_model_path('')}")
    print(f"结果保存: {get_output_path('')}")
    print("\n生成的文件:")
    if len(all_results) > 1:
        print(f"  - comparison.png (模型对比)")
        print(f"  - confusion_matrices.png (混淆矩阵)")
    for model_name in classifiers_dict.keys():
        print(f"  - tsne_{model_name}.png (t-SNE可视化)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

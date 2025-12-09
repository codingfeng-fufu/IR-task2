#!/usr/bin/env python3
"""Stage3 NaiveBayes Optimization - 训练脚本

本阶段优化朴素贝叶斯模型并训练所有模型进行验证:
- 朴素贝叶斯 V1: 73.46%
- 朴素贝叶斯 V2 (优化): 79.20% (+5.74%)
- Word2Vec + SVM: 82.99%
- BERT: 87.91%

用法:
    python train.py                  # 训练所有模型
    python train.py --model nb       # 仅训练朴素贝叶斯(V1+V2)
    python train.py --model w2v      # 仅训练Word2Vec+SVM
    python train.py --model bert     # 仅训练BERT
    python train.py --quick          # 快速测试
"""

import os
import sys
import argparse
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 首先导入Stage3自己的config（必须在导入Stage1之前）
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Stage3目录优先
import config as stage3_config

# 然后添加Stage1到路径以访问基础设施
sys.path.insert(0, os.path.join(current_dir, '..', 'Stage1_Foundation'))

from data_loader import DataLoader
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
from naive_bayes_classifier_optimized import NaiveBayesClassifierOptimized

# 使用Stage3的配置函数
get_data_path = stage3_config.get_data_path
get_model_path = stage3_config.get_model_path
get_output_path = stage3_config.get_output_path

# 导入Stage2的模型用于训练和对比
try:
    sys.path.insert(0, os.path.join(current_dir, '..', 'Stage2_Traditional_Models'))
    from naive_bayes_classifier import NaiveBayesClassifier
    from word2vec_svm_classifier import Word2VecSVMClassifier
    from bert_classifier import BERTClassifier
    HAS_STAGE2_MODELS = True
except Exception as e:
    print(f"⚠️ 无法导入Stage2模型: {e}")
    HAS_STAGE2_MODELS = False


def print_banner():
    print("\n" + "="*80)
    print(" " * 18 + "Stage3 NaiveBayes Optimization + 全模型验证")
    print(" " * 20 + "朴素贝叶斯从73.46%提升至79.20% (+5.74%)")
    print("="*80 + "\n")


def train_naive_bayes_v1(train_titles, train_labels, test_titles, test_labels):
    """训练朴素贝叶斯V1作为对比基准."""
    print("[2] 训练朴素贝叶斯 V1 (基础版)")
    print("-" * 80)

    start_time = time.time()

    classifier = NaiveBayesClassifier(
        max_features=5000,
        ngram_range=(1, 2),
        model_path=get_model_path('naive_bayes_v1_model.pkl')
    )
    classifier.train(train_titles, train_labels)

    train_time = time.time() - start_time
    predictions = classifier.predict(test_titles)

    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "Naive Bayes V1", verbose=False)

    print(f"\n✓ 训练完成: {train_time:.2f}秒")
    print(f"📊 性能 (预期 ~73%)")
    print(f"  准确率: {result['accuracy']:.4f}")
    print(f"  F1分数: {result['f1']:.4f}\n")

    result['training_time'] = train_time
    return classifier, result


def train_naive_bayes_v2(train_titles, train_labels, test_titles, test_labels):
    """训练朴素贝叶斯V2优化版本."""
    print("[3] 训练朴素贝叶斯 V2 (优化版)")
    print("-" * 80)
    print("优化策略:")
    print("  1. 多层TF-IDF: 词级(10K) + 字符级(5K)")
    print("  2. 统计特征: 22维专门特征")
    print("  3. ComplementNB算法")
    print()

    start_time = time.time()

    classifier = NaiveBayesClassifierOptimized(
        max_features_word=10000,
        max_features_char=5000,
        word_ngram_range=(1, 3),
        char_ngram_range=(3, 5),
        alpha=0.5,
        model_path=get_model_path('naive_bayes_v2_optimized_model.pkl')
    )
    classifier.train(train_titles, train_labels)

    train_time = time.time() - start_time
    predictions = classifier.predict(test_titles)

    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "Naive Bayes V2 (Optimized)", verbose=False)

    print(f"\n✓ 训练完成: {train_time:.2f}秒")
    print(f"📊 性能 (预期 ~79%)")
    print(f"  准确率: {result['accuracy']:.4f}")
    print(f"  F1分数: {result['f1']:.4f}\n")

    result['training_time'] = train_time
    return classifier, result


def train_word2vec_svm(train_titles, train_labels, test_titles, test_labels):
    """训练Word2Vec+SVM模型."""
    print("[4] 训练 Word2Vec + SVM")
    print("-" * 80)

    start_time = time.time()

    classifier = Word2VecSVMClassifier(
        vector_size=100,
        window=5,
        use_linear_svm=False,
        add_features=True,
        model_path=get_model_path('word2vec_svm')
    )
    classifier.train(train_titles, train_labels)

    train_time = time.time() - start_time
    predictions = classifier.predict(test_titles)

    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "Word2Vec + SVM", verbose=False)

    print(f"\n✓ 训练完成: {train_time:.2f}秒 ({train_time/60:.1f}分钟)")
    print(f"📊 性能 (预期 ~83%)")
    print(f"  准确率: {result['accuracy']:.4f}")
    print(f"  F1分数: {result['f1']:.4f}\n")

    result['training_time'] = train_time
    return classifier, result


def train_bert(train_titles, train_labels, test_titles, test_labels, epochs=3, batch_size=16):
    """训练BERT模型."""
    print("[5] 训练 BERT")
    print("-" * 80)
    print(f"配置: epochs={epochs}, batch_size={batch_size}")
    print()

    start_time = time.time()

    classifier = BERTClassifier(
        model_name='bert-base-uncased',
        max_length=64,
        model_path=get_model_path('best_bert_model.pt')
    )
    classifier.train(
        train_titles,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=2e-5,
        warmup_steps=500
    )

    train_time = time.time() - start_time
    predictions = classifier.predict(test_titles, batch_size=batch_size)

    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "BERT", verbose=False)

    print(f"\n✓ 训练完成: {train_time:.2f}秒 ({train_time/60:.1f}分钟)")
    print(f"📊 性能 (预期 ~88%)")
    print(f"  准确率: {result['accuracy']:.4f}")
    print(f"  F1分数: {result['f1']:.4f}\n")

    result['training_time'] = train_time
    return classifier, result


def main():
    parser = argparse.ArgumentParser(
        description='Stage3 NaiveBayes Optimization + 全模型训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py                  # 训练所有模型
  python train.py --model nb       # 仅训练朴素贝叶斯
  python train.py --model w2v      # 仅训练Word2Vec+SVM
  python train.py --model bert     # 仅训练BERT
  python train.py --quick          # 快速测试
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['nb', 'w2v', 'bert', 'all'],
        default='all',
        help='选择要训练的模型'
    )
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    parser.add_argument('--max-samples', type=int, default=None, help='最大训练样本数')
    parser.add_argument('--epochs', type=int, default=3, help='BERT训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='BERT批次大小')
    args = parser.parse_args()

    if args.quick and args.max_samples is None:
        args.max_samples = 10000
        if args.epochs == 3:
            args.epochs = 1

    print_banner()

    # 检查Stage2模型是否可用
    if not HAS_STAGE2_MODELS and args.model in ['w2v', 'bert', 'all']:
        print("❌ 无法导入Stage2模型，请检查Stage2_Traditional_Models目录")
        sys.exit(1)

    # 加载数据
    print("[1] 加载数据")
    print("-" * 80)
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        get_data_path('positive.txt'),
        get_data_path('negative.txt'),
        get_data_path('testSet-1000.xlsx')
    )

    if args.max_samples and args.max_samples < len(train_titles):
        import random
        indices = list(range(len(train_titles)))
        random.shuffle(indices)
        indices = indices[:args.max_samples]
        train_titles = [train_titles[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        print(f"⚠ 快速测试: {len(train_titles)} 样本")

    print(f"✓ 训练数据: {len(train_titles)} 样本")
    print(f"✓ 测试数据: {len(test_titles)} 样本\n")

    # 训练模型
    all_results = []
    classifiers_dict = {}
    total_start = time.time()

    # 训练朴素贝叶斯
    if args.model in ['nb', 'all']:
        nb_v1_clf, nb_v1_result = train_naive_bayes_v1(train_titles, train_labels, test_titles, test_labels)
        all_results.append(nb_v1_result)
        classifiers_dict['Naive_Bayes_V1'] = nb_v1_clf

        nb_v2_clf, nb_v2_result = train_naive_bayes_v2(train_titles, train_labels, test_titles, test_labels)
        all_results.append(nb_v2_result)
        classifiers_dict['Naive_Bayes_V2_Optimized'] = nb_v2_clf

        # 显示朴素贝叶斯优化提升
        improvement = (nb_v2_result['accuracy'] - nb_v1_result['accuracy']) * 100
        print("\n" + "="*80)
        print(" " * 25 + "朴素贝叶斯优化效果")
        print("="*80)
        print(f"\n🎯 准确率提升: +{improvement:.2f}个百分点 ({nb_v1_result['accuracy']:.4f} → {nb_v2_result['accuracy']:.4f})")
        print(f"目标提升: +5.74个百分点 (达成率: {improvement/5.74*100:.1f}%)\n")

    # 训练Word2Vec+SVM
    if args.model in ['w2v', 'all']:
        w2v_clf, w2v_result = train_word2vec_svm(train_titles, train_labels, test_titles, test_labels)
        all_results.append(w2v_result)
        classifiers_dict['Word2Vec_SVM'] = w2v_clf

    # 训练BERT
    if args.model in ['bert', 'all']:
        bert_clf, bert_result = train_bert(
            train_titles, train_labels, test_titles, test_labels,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        all_results.append(bert_result)
        classifiers_dict['BERT'] = bert_clf

    total_time = time.time() - total_start

    # 对比所有模型
    if len(all_results) > 1:
        print("\n" + "="*80)
        print(" " * 30 + "模型对比")
        print("="*80 + "\n")
        ModelEvaluator.compare_models(all_results)

        # 生成对比可视化
        visualizer = ResultVisualizer()
        visualizer.plot_comparison(all_results, save_path=get_output_path('model_comparison.png'))
        visualizer.plot_confusion_matrices(all_results, save_path=get_output_path('confusion_matrices.png'))
        print(f"\n✓ 可视化已保存到: {get_output_path('')}")

    # 生成 t-SNE 可视化
    print("\n" + "="*80)
    print(" " * 28 + "生成 t-SNE 可视化")
    print("="*80 + "\n")

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

    # 保存结果摘要
    results_file = get_output_path('training_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" "*20 + "Stage3 NaiveBayes Optimization - 训练结果\n")
        f.write("="*80 + "\n\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for result in all_results:
            f.write("-"*80 + "\n")
            f.write(f"模型: {result['model']}\n")
            f.write("-"*80 + "\n")
            f.write(f"准确率: {result['accuracy']:.4f}\n")
            f.write(f"精确率: {result['precision']:.4f}\n")
            f.write(f"召回率: {result['recall']:.4f}\n")
            f.write(f"F1分数: {result['f1']:.4f}\n")
            if 'training_time' in result:
                f.write(f"训练时间: {result['training_time']:.2f}秒\n")
            f.write("\n")

    print("\n" + "="*80)
    print(" " * 30 + "训练完成")
    print("="*80)
    print(f"\n总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"模型保存: {get_model_path('')}")
    print(f"结果保存: {results_file}")
    print("\n生成的文件:")
    if len(all_results) > 1:
        print(f"  - model_comparison.png")
        print(f"  - confusion_matrices.png")
    for model_name in classifiers_dict.keys():
        print(f"  - tsne_{model_name}.png")
    print(f"  - training_results.txt")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

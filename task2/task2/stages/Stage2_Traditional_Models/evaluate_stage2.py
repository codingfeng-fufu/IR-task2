#!/usr/bin/env python3
"""
Stage2 评估脚本 - 使用已训练的模型生成输出
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
stage1_dir = os.path.join(current_dir, '..', 'Stage1_Foundation')

# 导入Stage1的工具类
sys.path.insert(0, stage1_dir)
from data_loader import DataLoader
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
from naive_bayes_classifier import NaiveBayesClassifier
from word2vec_svm_classifier import Word2VecSVMClassifier
from bert_classifier import BERTClassifier

# 使用Stage2自己的config
sys.path.insert(0, current_dir)
import config as stage2_config

get_data_path = stage2_config.get_data_path
get_model_path = stage2_config.get_model_path
get_output_path = stage2_config.get_output_path

print("="*80)
print(" " * 25 + "Stage2 Traditional Models - 评估")
print("="*80)

# 加载测试数据
print("\n[1] 加载测试数据")
print("-" * 80)

train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
    get_data_path('positive.txt'),
    get_data_path('negative.txt'),
    get_data_path('testSet-1000.xlsx')
)

print(f"✓ 测试集: {len(test_titles)} 样本\n")

# 评估所有模型
print("[2] 评估所有已训练模型")
print("-" * 80 + "\n")

all_results = []
classifiers_dict = {}

# 朴素贝叶斯
print("[1/3] 评估朴素贝叶斯...")
nb_classifier = NaiveBayesClassifier(
    max_features=5000,
    ngram_range=(1, 2),
    model_path=get_model_path('naive_bayes_model.pkl')
)
if nb_classifier.load_model():
    predictions = nb_classifier.predict(test_titles)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "Naive Bayes", verbose=False)
    print(f"✓ 准确率: {result['accuracy']:.4f}, F1: {result['f1']:.4f}\n")
    all_results.append(result)
    classifiers_dict['Naive_Bayes'] = nb_classifier

# Word2Vec + SVM
print("[2/3] 评估 Word2Vec + SVM...")
w2v_classifier = Word2VecSVMClassifier(
    vector_size=100,
    window=5,
    model_path=get_model_path('word2vec_svm_model')
)
if w2v_classifier.load_model():
    predictions = w2v_classifier.predict(test_titles)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "Word2Vec + SVM", verbose=False)
    print(f"✓ 准确率: {result['accuracy']:.4f}, F1: {result['f1']:.4f}\n")
    all_results.append(result)
    classifiers_dict['Word2Vec_SVM'] = w2v_classifier

# BERT
print("[3/3] 评估 BERT...")
bert_classifier = BERTClassifier(
    model_name='bert-base-uncased',
    max_length=64,
    model_path=get_model_path('best_bert_model.pt')
)
if bert_classifier.load_model():
    predictions = bert_classifier.predict(test_titles, batch_size=16)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate_model(test_labels, predictions, "BERT", verbose=False)
    print(f"✓ 准确率: {result['accuracy']:.4f}, F1: {result['f1']:.4f}\n")
    all_results.append(result)
    classifiers_dict['BERT'] = bert_classifier

# 生成对比
if len(all_results) > 1:
    print("\n" + "="*80)
    print(" " * 30 + "模型对比")
    print("="*80 + "\n")
    ModelEvaluator.compare_models(all_results)

    print("\n[3] 生成可视化")
    print("-" * 80 + "\n")

    visualizer = ResultVisualizer()

    comparison_path = get_output_path('comparison.png')
    visualizer.plot_comparison(all_results, save_path=comparison_path)
    print(f"✓ 模型对比图: {comparison_path}")

    confusion_path = get_output_path('confusion_matrices.png')
    visualizer.plot_confusion_matrices(all_results, save_path=confusion_path)
    print(f"✓ 混淆矩阵: {confusion_path}")

# 生成 t-SNE 可视化
print("\n[4] 生成 t-SNE 可视化")
print("-" * 80 + "\n")

visualizer = ResultVisualizer()
for model_name, classifier in classifiers_dict.items():
    try:
        feature_vectors = classifier.get_feature_vectors(test_titles)
        tsne_path = get_output_path(f'tsne_{model_name}.png')
        visualizer.visualize_embeddings_tsne(
            feature_vectors,
            test_labels,
            model_name,
            save_path=tsne_path
        )
        print(f"✓ {model_name}: {tsne_path}")
    except Exception as e:
        print(f"⚠️  {model_name} t-SNE 失败: {str(e)}")

# 保存文本结果
print("\n[5] 保存文本结果")
print("-" * 80 + "\n")

from datetime import datetime
results_txt_path = get_output_path('training_results.txt')
with open(results_txt_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write(" " * 20 + "Stage2 Traditional Models - 评估结果\n")
    f.write("="*80 + "\n\n")
    f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    for result in all_results:
        f.write("-"*80 + "\n")
        f.write(f"模型: {result['model']}\n")
        f.write("-"*80 + "\n")
        f.write(f"准确率: {result['accuracy']:.4f}\n")
        f.write(f"精确率: {result['precision']:.4f}\n")
        f.write(f"召回率: {result['recall']:.4f}\n")
        f.write(f"F1分数: {result['f1']:.4f}\n")
        f.write("\n")

print(f"✓ 文本结果: {results_txt_path}")

print("\n" + "="*80)
print(" " * 30 + "评估完成")
print("="*80)
print(f"\n结果保存在: {get_output_path('')}")
print("\n生成的文件:")
print(f"  - comparison.png")
print(f"  - confusion_matrices.png")
print(f"  - training_results.txt")
for model_name in classifiers_dict.keys():
    print(f"  - tsne_{model_name}.png")
print("\n" + "="*80 + "\n")

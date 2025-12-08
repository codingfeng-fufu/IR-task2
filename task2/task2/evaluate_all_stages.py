#!/usr/bin/env python3
"""
Task2 完整评估 - 所有阶段模型对比
=====================================
评估Stage1到Stage4的所有训练模型
"""

import os
import sys
from pathlib import Path

# 添加所有阶段路径
base_dir = Path(__file__).parent
for stage in ['Stage1_Foundation', 'Stage2_Traditional_Models',
              'Stage3_NaiveBayes_Optimization', 'Stage4_BERT_Optimization']:
    sys.path.insert(0, str(base_dir / 'stages' / stage))

from data_loader import DataLoader
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
import config

print("="*80)
print(" " * 25 + "Task2 完整评估")
print(" " * 20 + "所有阶段模型性能对比")
print("="*80)

# 加载测试数据
print("\n[1] 加载测试数据")
print("-" * 80)

train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
    config.get_data_path('positive.txt'),
    config.get_data_path('negative.txt'),
    config.get_data_path('testSet-1000.xlsx')
)

print(f"✓ 测试集: {len(test_titles)} 样本")
print(f"  - 正样本: {sum(test_labels)} ({sum(test_labels)/len(test_labels)*100:.1f}%)")
print(f"  - 负样本: {len(test_labels)-sum(test_labels)} ({(len(test_labels)-sum(test_labels))/len(test_labels)*100:.1f}%)")

# 定义所有模型
models_to_evaluate = [
    # Stage2/3 - 传统模型
    {
        'name': 'Naive Bayes (原始)',
        'stage': 'Stage3',
        'type': 'nb_original',
        'path': base_dir / 'stages/Stage3_NaiveBayes_Optimization/models/naive_bayes_original_model.pkl'
    },
    {
        'name': 'Naive Bayes (优化)',
        'stage': 'Stage3',
        'type': 'nb_optimized',
        'path': base_dir / 'stages/Stage3_NaiveBayes_Optimization/models/naive_bayes_optimized_model.pkl'
    },
    {
        'name': 'Word2Vec + SVM',
        'stage': 'Stage3',
        'type': 'word2vec_svm',
        'path': base_dir / 'stages/Stage3_NaiveBayes_Optimization/models/word2vec_svm_svm.pkl'
    },
    # Stage4 - BERT模型
    {
        'name': 'BERT Baseline',
        'stage': 'Stage4',
        'type': 'bert',
        'path': base_dir / 'stages/Stage4_BERT_Optimization/models/bert_model.pt'
    },
    {
        'name': 'SciBERT + Focal Loss',
        'stage': 'Stage4',
        'type': 'scibert',
        'path': base_dir / 'stages/Stage4_BERT_Optimization/models/scibert_optimized_model.pt'
    },
    {
        'name': 'DeBERTa-v3',
        'stage': 'Stage4',
        'type': 'deberta',
        'path': base_dir / 'stages/Stage4_BERT_Optimization/models/deberta_optimized_model.pt'
    }
]

# 评估所有模型
print("\n[2] 评估所有模型")
print("-" * 80 + "\n")

results = []
evaluator = ModelEvaluator()

for i, model_info in enumerate(models_to_evaluate, 1):
    print(f"\n[{i}/{len(models_to_evaluate)}] 评估: {model_info['name']}")
    print(f"{'='*70}")

    model_path = model_info['path']

    # 检查文件
    if not model_path.exists():
        print(f"⚠️  模型文件不存在: {model_path}")
        continue

    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"模型文件: {model_path.name}")
    print(f"模型大小: {model_size_mb:.1f} MB")
    print(f"所属阶段: {model_info['stage']}")

    try:
        # 根据模型类型加载
        if model_info['type'] == 'nb_original':
            from naive_bayes_classifier import NaiveBayesClassifier
            classifier = NaiveBayesClassifier(model_path=str(model_path))
            classifier.load_model()

        elif model_info['type'] == 'nb_optimized':
            sys.path.insert(0, str(base_dir / 'stages/Stage3_NaiveBayes_Optimization'))
            from naive_bayes_classifier_optimized import NaiveBayesClassifierOptimized
            classifier = NaiveBayesClassifierOptimized(model_path=str(model_path))
            classifier.load_model()

        elif model_info['type'] == 'word2vec_svm':
            from word2vec_svm_classifier import Word2VecSVMClassifier
            base_path = str(model_path).replace('_svm.pkl', '')
            classifier = Word2VecSVMClassifier(model_path=base_path)
            classifier.load_model()

        elif model_info['type'] in ['bert', 'scibert', 'deberta']:
            sys.path.insert(0, str(base_dir / 'stages/Stage4_BERT_Optimization'))

            if model_info['type'] == 'bert':
                from bert_classifier import BERTClassifier
                classifier = BERTClassifier(
                    model_name='bert-base-uncased',
                    max_length=64,
                    model_path=str(model_path)
                )
            else:
                from bert_classifier_optimized import OptimizedBERTClassifier
                if model_info['type'] == 'scibert':
                    model_name = 'allenai/scibert_scivocab_uncased'
                else:  # deberta
                    model_name = 'microsoft/deberta-v3-base'

                classifier = OptimizedBERTClassifier(
                    model_name=model_name,
                    model_path=str(model_path)
                )

            classifier.load_model()

        # 预测
        print("\n开始预测...")
        predictions = classifier.predict(test_titles)

        # 评估
        result = evaluator.evaluate_model(
            test_labels,
            predictions,
            model_info['name'],
            verbose=False
        )

        # 添加额外信息
        result['stage'] = model_info['stage']
        result['model_size_mb'] = model_size_mb
        results.append(result)

        print(f"\n✓ 评估完成")
        print(f"  准确率: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"  F1分数: {result['f1']:.4f}")

    except Exception as e:
        print(f"\n⚠️  评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# 生成对比报告
if not results:
    print("\n⚠️  没有成功评估任何模型")
    sys.exit(1)

print("\n" + "="*80)
print(" " * 25 + "所有模型性能对比")
print("="*80 + "\n")

# 按准确率排序
results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

# 打印表格
header = f"{'排名':<6} {'模型':<30} {'阶段':<10} {'准确率':>10} {'F1分数':>10} {'大小(MB)':>12}"
print(header)
print("-" * 80)

for rank, result in enumerate(results_sorted, 1):
    row = f"{rank:<6} "
    row += f"{result['model']:<30} "
    row += f"{result['stage']:<10} "
    row += f"{result['accuracy']:>9.4f} "
    row += f"{result['f1']:>9.4f} "
    row += f"{result['model_size_mb']:>11.1f}"
    print(row)

print("-" * 80)

# 最佳模型
best_model = results_sorted[0]
print(f"\n🏆 最佳模型: {best_model['model']}")
print(f"   准确率: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)")
print(f"   F1分数: {best_model['f1']:.4f}")
print(f"   所属阶段: {best_model['stage']}")

# 按阶段分组统计
print(f"\n{'='*80}")
print(" " * 25 + "各阶段最佳模型")
print("="*80 + "\n")

stages = {}
for result in results:
    stage = result['stage']
    if stage not in stages or result['accuracy'] > stages[stage]['accuracy']:
        stages[stage] = result

for stage in sorted(stages.keys()):
    result = stages[stage]
    print(f"{stage}:")
    print(f"  模型: {result['model']}")
    print(f"  准确率: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print(f"  F1分数: {result['f1']:.4f}")
    print()

# 保存结果
output_dir = base_dir / 'output'
output_dir.mkdir(exist_ok=True)

# JSON格式
import json
json_path = output_dir / 'all_stages_comparison.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"✓ JSON结果已保存: {json_path}")

# 文本报告
report_path = output_dir / 'all_stages_comparison_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write(" " * 25 + "Task2 完整评估报告\n")
    f.write(" " * 20 + "所有阶段模型性能对比\n")
    f.write("="*80 + "\n\n")

    f.write(f"评估日期: 2025-12-08\n")
    f.write(f"测试样本数: {len(test_titles)}\n")
    f.write(f"评估模型数: {len(results)}\n\n")

    f.write("## 完整排名\n\n")
    f.write(f"{'排名':<6} {'模型':<35} {'阶段':<12} {'准确率':>12} {'F1分数':>12}\n")
    f.write("-" * 85 + "\n")

    for rank, result in enumerate(results_sorted, 1):
        f.write(f"{rank:<6} ")
        f.write(f"{result['model']:<35} ")
        f.write(f"{result['stage']:<12} ")
        f.write(f"{result['accuracy']:>11.4f} ")
        f.write(f"{result['f1']:>11.4f}\n")

    f.write("\n## 最佳模型\n\n")
    f.write(f"**{best_model['model']}** ({best_model['stage']})\n")
    f.write(f"- 准确率: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)\n")
    f.write(f"- F1分数: {best_model['f1']:.4f}\n")
    f.write(f"- 模型大小: {best_model['model_size_mb']:.1f} MB\n\n")

    f.write("## 各阶段最佳\n\n")
    for stage in sorted(stages.keys()):
        result = stages[stage]
        f.write(f"### {stage}\n\n")
        f.write(f"模型: {result['model']}\n")
        f.write(f"- 准确率: {result['accuracy']:.4f}\n")
        f.write(f"- F1分数: {result['f1']:.4f}\n\n")

print(f"✓ 文本报告已保存: {report_path}")

# 生成可视化
print(f"\n{'='*80}")
print(" " * 30 + "生成可视化")
print("="*80 + "\n")

visualizer = ResultVisualizer()

# 模型对比图
comparison_path = output_dir / 'all_models_comparison.png'
visualizer.plot_comparison(results, save_path=str(comparison_path))
print(f"✓ 模型对比图: {comparison_path}")

# 混淆矩阵
confusion_path = output_dir / 'all_models_confusion_matrices.png'
visualizer.plot_confusion_matrices(results[:4], save_path=str(confusion_path))  # 只显示前4个
print(f"✓ 混淆矩阵 (前4个模型): {confusion_path}")

print("\n" + "="*80)
print(" " * 30 + "评估完成")
print("="*80)
print("\n生成的文件:")
print(f"  - all_stages_comparison.json")
print(f"  - all_stages_comparison_report.txt")
print(f"  - all_models_comparison.png")
print(f"  - all_models_confusion_matrices.png")
print("\n" + "="*80 + "\n")

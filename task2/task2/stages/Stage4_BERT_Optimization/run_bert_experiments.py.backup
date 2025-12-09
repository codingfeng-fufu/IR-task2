"""
run_bert_experiments.py
=======================
批量运行BERT优化实验,对比不同模型和配置

快速对比:
1. SciBERT vs RoBERTa vs DeBERTa vs BERT-base
2. Focal Loss vs Weighted CE vs Standard CE
3. 不同max_length: 64 vs 96 vs 128
4. 对抗训练 vs 无对抗训练
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List
from data_loader import DataLoader as TitleDataLoader
from train_bert_optimized_v2 import OptimizedBERTClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
import numpy as np


def run_single_experiment(
    experiment_name: str,
    train_titles: List[str],
    train_labels: List[int],
    test_titles: List[str],
    test_labels: List[int],
    config: Dict
) -> Dict:
    """运行单个实验"""

    print("\n" + "="*100)
    print(f" 🔬 实验: {experiment_name}")
    print("="*100)
    print(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")

    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 创建分类器
    classifier = OptimizedBERTClassifier(
        model_name=config.get('model_name', 'scibert'),
        max_length=config.get('max_length', 96),
        model_path=os.path.join(script_dir, f"models/experiments/{experiment_name}.pt"),
        dropout_rate=config.get('dropout_rate', 0.2)
    )

    # 训练
    try:
        history = classifier.train(
            train_titles,
            train_labels,
            val_ratio=config.get('val_ratio', 0.1),
            epochs=config.get('epochs', 10),
            batch_size=config.get('batch_size', 32),
            learning_rate=config.get('learning_rate', 2e-5),
            warmup_ratio=config.get('warmup_ratio', 0.1),
            scheduler_type=config.get('scheduler_type', 'cosine'),
            loss_type=config.get('loss_type', 'focal'),
            class_weight_positive=config.get('class_weight_positive', 1.3),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
            early_stopping_patience=config.get('early_stopping_patience', 3),
            use_layer_wise_lr=config.get('use_layer_wise_lr', True),
            layer_decay=config.get('layer_decay', 0.95),
            use_adversarial=config.get('use_adversarial', True),
            adv_epsilon=config.get('adv_epsilon', 1.0),
            use_mixed_precision=config.get('use_mixed_precision', True),
            save_model=True
        )
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'experiment_name': experiment_name,
            'status': 'failed',
            'error': str(e)
        }

    # 测试
    print(f"\n测试集评估...")
    predictions = classifier.predict(test_titles)
    probabilities = classifier.predict_proba(test_titles)

    # 计算指标
    results = {
        'experiment_name': experiment_name,
        'status': 'success',
        'config': config,
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions),
        'recall': recall_score(test_labels, predictions),
        'f1': f1_score(test_labels, predictions),
        'training_history': history
    }

    # 打印结果
    print(f"\n{'='*100}")
    print(f" 📊 {experiment_name} - 结果")
    print(f"{'='*100}")
    print(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%) ⭐")
    print(f"F1 Score:  {results['f1']:.4f} ({results['f1']*100:.2f}%)")

    return results


def main():
    """运行所有实验"""

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载数据
    print("\n" + "="*100)
    print(" 📦 加载数据")
    print("="*100)
    print(f"脚本目录: {script_dir}")

    train_titles, train_labels, test_titles, test_labels = TitleDataLoader.prepare_dataset(
        os.path.join(script_dir, 'data/positive.txt'),
        os.path.join(script_dir, 'data/negative.txt'),
        os.path.join(script_dir, 'data/testSet-1000.xlsx')
    )

    if len(train_titles) == 0:
        print("❌ 未找到数据文件!")
        return

    print(f"✓ 数据加载完成")
    print(f"  训练集: {len(train_titles)} 样本")
    print(f"  测试集: {len(test_titles)} 样本")

    # 定义实验
    experiments = [
        # 实验1: Baseline (BERT-base)
        {
            'name': 'exp1_bert_base_baseline',
            'config': {
                'model_name': 'bert-base',
                'max_length': 64,
                'epochs': 8,
                'batch_size': 32,
                'learning_rate': 2e-5,
                'loss_type': 'ce',  # 标准交叉熵
                'use_layer_wise_lr': False,
                'use_adversarial': False,
                'use_mixed_precision': True
            }
        },

        # 实验2: SciBERT (学术专用)
        {
            'name': 'exp2_scibert_focal',
            'config': {
                'model_name': 'scibert',
                'max_length': 96,
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 2e-5,
                'loss_type': 'focal',  # Focal Loss
                'focal_alpha': 0.25,
                'focal_gamma': 2.0,
                'use_layer_wise_lr': True,
                'layer_decay': 0.95,
                'use_adversarial': True,
                'use_mixed_precision': True
            }
        },

        # 实验3: RoBERTa
        {
            'name': 'exp3_roberta_weighted',
            'config': {
                'model_name': 'roberta',
                'max_length': 96,
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 2e-5,
                'loss_type': 'weighted_ce',  # 加权交叉熵
                'class_weight_positive': 1.3,
                'use_layer_wise_lr': True,
                'use_adversarial': True,
                'use_mixed_precision': True
            }
        },

        # 实验4: SciBERT + max_length=128
        {
            'name': 'exp5_scibert_maxlen128',
            'config': {
                'model_name': 'scibert',
                'max_length': 128,  # 更长序列
                'epochs': 10,
                'batch_size': 24,  # 减小batch size适应更长序列
                'learning_rate': 2e-5,
                'loss_type': 'focal',
                'use_layer_wise_lr': True,
                'use_adversarial': True,
                'use_mixed_precision': True
            }
        },
    ]

    # 运行实验
    all_results = []
    os.makedirs(os.path.join(script_dir, 'models/experiments'), exist_ok=True)

    for exp in experiments:
        result = run_single_experiment(
            exp['name'],
            train_titles,
            train_labels,
            test_titles,
            test_labels,
            exp['config']
        )
        all_results.append(result)

        # 保存中间结果
        with open(os.path.join(script_dir, 'models/experiments/results.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # 生成对比报告
    print("\n\n" + "="*100)
    print(" 🏆 实验对比报告")
    print("="*100)

    # 创建对比表格
    comparison_data = []
    for result in all_results:
        if result['status'] == 'success':
            comparison_data.append({
                '实验名称': result['experiment_name'],
                '模型': result['config']['model_name'],
                'Max Length': result['config']['max_length'],
                'Loss Type': result['config']['loss_type'],
                'Layer-wise LR': '✓' if result['config'].get('use_layer_wise_lr') else '✗',
                '对抗训练': '✓' if result['config'].get('use_adversarial') else '✗',
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1': f"{result['f1']:.4f}"
            })

    df = pd.DataFrame(comparison_data)
    print("\n")
    print(df.to_string(index=False))

    # 找出最佳模型
    successful_results = [r for r in all_results if r['status'] == 'success']
    if successful_results:
        best_f1 = max(successful_results, key=lambda x: x['f1'])
        best_recall = max(successful_results, key=lambda x: x['recall'])

        print(f"\n" + "="*100)
        print(f" 🏅 最佳模型")
        print(f"="*100)
        print(f"\n最高F1分数: {best_f1['experiment_name']}")
        print(f"  - F1: {best_f1['f1']:.4f} ({best_f1['f1']*100:.2f}%)")
        print(f"  - Accuracy: {best_f1['accuracy']:.4f} ({best_f1['accuracy']*100:.2f}%)")
        print(f"  - Recall: {best_f1['recall']:.4f} ({best_f1['recall']*100:.2f}%)")

        print(f"\n最高召回率: {best_recall['experiment_name']}")
        print(f"  - Recall: {best_recall['recall']:.4f} ({best_recall['recall']*100:.2f}%)")
        print(f"  - F1: {best_recall['f1']:.4f} ({best_recall['f1']*100:.2f}%)")
        print(f"  - Accuracy: {best_recall['accuracy']:.4f} ({best_recall['accuracy']*100:.2f}%)")

        # 对比原始baseline
        baseline_acc = 0.8525
        baseline_recall = 0.8116
        baseline_f1 = 0.8649

        print(f"\n" + "="*100)
        print(f" 📈 相比原始BERT的提升")
        print(f"="*100)
        print(f"\n原始BERT (bert-base-uncased, max_length=64):")
        print(f"  Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        print(f"  Recall:   {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
        print(f"  F1:       {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")

        print(f"\n最佳优化模型 ({best_f1['experiment_name']}):")
        print(f"  Accuracy: {best_f1['accuracy']:.4f} ({best_f1['accuracy']*100:.2f}%) "
              f"[{'+' if best_f1['accuracy'] > baseline_acc else ''}{(best_f1['accuracy']-baseline_acc)*100:.2f}%]")
        print(f"  Recall:   {best_f1['recall']:.4f} ({best_f1['recall']*100:.2f}%) "
              f"[{'+' if best_f1['recall'] > baseline_recall else ''}{(best_f1['recall']-baseline_recall)*100:.2f}%]")
        print(f"  F1:       {best_f1['f1']:.4f} ({best_f1['f1']*100:.2f}%) "
              f"[{'+' if best_f1['f1'] > baseline_f1 else ''}{(best_f1['f1']-baseline_f1)*100:.2f}%]")

    # 保存最终报告
    report_path = os.path.join(script_dir, 'models/experiments/comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write(" BERT优化实验对比报告\n")
        f.write("="*100 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("="*100 + "\n")
        f.write(f" 最佳模型: {best_f1['experiment_name']}\n")
        f.write("="*100 + "\n")
        f.write(f"F1: {best_f1['f1']:.4f}\n")
        f.write(f"Accuracy: {best_f1['accuracy']:.4f}\n")
        f.write(f"Recall: {best_f1['recall']:.4f}\n")
        f.write(f"Precision: {best_f1['precision']:.4f}\n")

    print(f"\n✓ 详细报告已保存至: {report_path}")
    print(f"✓ 完整结果已保存至: {os.path.join(script_dir, 'models/experiments/results.json')}")


if __name__ == "__main__":
    main()

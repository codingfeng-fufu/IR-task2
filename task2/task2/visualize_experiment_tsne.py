"""
visualize_experiment_tsne.py
=============================
为 run_bert_experiments.py 生成的所有实验模型生成 t-SNE 可视化图

用法:
    python visualize_experiment_tsne.py
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict
from data_loader import DataLoader as TitleDataLoader
from train_bert_optimized_v2 import OptimizedBERTClassifier
from visualizer import ResultVisualizer
from tqdm import tqdm


def visualize_all_experiments(
    test_titles: List[str],
    test_labels: List[int],
    script_dir: str,
    output_dir: str = 'output/bert_experiments'
):
    """为所有实验模型生成 t-SNE 可视化"""

    # 实验配置 (需要与 run_bert_experiments.py 保持一致)
    experiments = [
        {
            'name': 'exp1_bert_base_baseline',
            'display_name': 'BERT-base (Baseline)',
            'model_name': 'bert-base',
            'max_length': 64
        },
        {
            'name': 'exp2_scibert_focal',
            'display_name': 'SciBERT + Focal Loss',
            'model_name': 'scibert',
            'max_length': 96
        },
        {
            'name': 'exp3_roberta_weighted',
            'display_name': 'RoBERTa + Weighted CE',
            'model_name': 'roberta',
            'max_length': 96
        },
        {
            'name': 'exp5_scibert_maxlen128',
            'display_name': 'SciBERT (MaxLen=128)',
            'model_name': 'scibert',
            'max_length': 128
        }
    ]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 从测试集中采样 (t-SNE 计算量大，采样可以加速)
    sample_size = min(500, len(test_titles))
    indices = np.random.RandomState(42).choice(len(test_titles), sample_size, replace=False)
    sampled_titles = [test_titles[i] for i in indices]
    sampled_labels = [test_labels[i] for i in indices]

    print("\n" + "="*100)
    print(" 🎨 为 BERT 实验生成 t-SNE 可视化")
    print("="*100)
    print(f"测试集大小: {len(test_titles)}")
    print(f"采样大小: {sample_size}")
    print(f"输出目录: {output_dir}")

    # 对每个实验生成 t-SNE
    successful_visualizations = []
    failed_visualizations = []

    for exp in experiments:
        print("\n" + "-"*100)
        print(f" 📊 处理实验: {exp['display_name']}")
        print("-"*100)

        # 模型路径 (优先使用 _best.pt, 否则使用 .pt)
        model_path_best = os.path.join(script_dir, f"models/experiments/{exp['name']}_best.pt")
        model_path_final = os.path.join(script_dir, f"models/experiments/{exp['name']}.pt")

        if os.path.exists(model_path_best):
            model_path = model_path_best
            print(f"✓ 找到最佳模型: {exp['name']}_best.pt")
        elif os.path.exists(model_path_final):
            model_path = model_path_final
            print(f"✓ 找到最终模型: {exp['name']}.pt")
        else:
            print(f"❌ 模型文件不存在!")
            failed_visualizations.append({
                'name': exp['name'],
                'error': 'Model file not found'
            })
            continue

        try:
            # 创建分类器并加载模型
            classifier = OptimizedBERTClassifier(
                model_name=exp['model_name'],
                max_length=exp['max_length'],
                model_path=model_path
            )

            # 加载模型
            if not classifier.load_model():
                print(f"❌ 加载模型失败!")
                failed_visualizations.append({
                    'name': exp['name'],
                    'error': 'Failed to load model'
                })
                continue

            # 手动设置 is_trained 标志 (修复加载后的状态)
            classifier.is_trained = True
            print(f"✓ 模型状态已设置为已训练")

            # 获取特征向量
            print(f"提取特征向量 (样本数: {sample_size})...")
            feature_vectors = classifier.get_feature_vectors(sampled_titles)
            print(f"✓ 特征向量维度: {feature_vectors.shape}")

            # 生成 t-SNE 可视化
            save_path = os.path.join(output_dir, f"tsne_{exp['name']}.png")
            ResultVisualizer.visualize_embeddings_tsne(
                vectors=feature_vectors,
                labels=sampled_labels,
                title=exp['display_name'],
                save_path=save_path,
                perplexity=30,
                n_iter=1000
            )

            print(f"✓ t-SNE 可视化已保存: {save_path}")
            successful_visualizations.append({
                'name': exp['name'],
                'display_name': exp['display_name'],
                'path': save_path
            })

        except Exception as e:
            print(f"❌ 生成可视化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_visualizations.append({
                'name': exp['name'],
                'error': str(e)
            })

    # 打印总结
    print("\n\n" + "="*100)
    print(" 📊 t-SNE 可视化生成总结")
    print("="*100)

    print(f"\n✓ 成功生成: {len(successful_visualizations)} 个可视化")
    for item in successful_visualizations:
        print(f"  - {item['display_name']}: {item['path']}")

    if failed_visualizations:
        print(f"\n❌ 失败: {len(failed_visualizations)} 个")
        for item in failed_visualizations:
            print(f"  - {item['name']}: {item['error']}")

    # 保存总结到 JSON
    summary = {
        'successful': successful_visualizations,
        'failed': failed_visualizations,
        'total': len(experiments),
        'sample_size': sample_size
    }

    summary_path = os.path.join(output_dir, 'tsne_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 总结已保存: {summary_path}")

    return successful_visualizations, failed_visualizations


def main():
    """主函数"""

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载测试数据
    print("\n" + "="*100)
    print(" 📦 加载测试数据")
    print("="*100)
    print(f"脚本目录: {script_dir}")

    _, _, test_titles, test_labels = TitleDataLoader.prepare_dataset(
        os.path.join(script_dir, 'data/positive.txt'),
        os.path.join(script_dir, 'data/negative.txt'),
        os.path.join(script_dir, 'data/testSet-1000.xlsx')
    )

    if len(test_titles) == 0:
        print("❌ 未找到测试数据!")
        return

    print(f"✓ 测试数据加载完成: {len(test_titles)} 样本")

    # 生成所有可视化
    visualize_all_experiments(
        test_titles,
        test_labels,
        script_dir,
        output_dir=os.path.join(script_dir, 'output/bert_experiments')
    )

    print("\n" + "="*100)
    print(" ✓ 所有任务完成!")
    print("="*100)


if __name__ == "__main__":
    main()

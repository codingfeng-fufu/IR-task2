#!/usr/bin/env python3
"""Stage5 LLM Framework - 统一训练脚本

本阶段使用四个中国大模型进行Few-shot分类:
- GLM-4.6 (智谱AI)
- Qwen3-Turbo (阿里云通义千问)
- Kimi-K2-Turbo (Moonshot AI)
- DeepSeek-Chat (DeepSeek)

用法:
    python train.py                      # 训练所有模型（交互选择）
    python train.py --all                # 训练所有4个模型
    python train.py --model glm-4.6      # 仅训练指定模型
    python train.py --sample 100         # 使用100个样本测试
    python train.py --config custom.json # 使用自定义配置
"""

import os
import sys
import argparse
import warnings
import time
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 添加Stage1到路径以访问基础设施
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Stage1_Foundation'))

from config import get_data_path, get_output_path
from data_loader import DataLoader
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer

# 导入run_llm_experiment的功能
from run_llm_experiment import (
    load_config,
    validate_api_keys,
    run_single_experiment,
    save_experiment_results
)


def print_banner():
    print("\n" + "="*80)
    print(" " * 25 + "Stage5 - LLM Framework 训练")
    print("="*80)
    print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Stage5 LLM Framework 统一训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py                          # 交互式选择模型
  python train.py --all                    # 训练所有4个模型
  python train.py --model glm-4.6          # 仅训练GLM-4.6
  python train.py --model deepseek --sample 100  # DeepSeek + 100样本测试
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['glm-4.6', 'qwen3', 'kimi', 'deepseek', 'all'],
        help='选择要训练的模型'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='训练所有启用的模型'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='llm_config.json',
        help='配置文件路径（默认: llm_config.json）'
    )

    parser.add_argument(
        '--sample',
        type=int,
        help='测试样本数（默认使用配置文件中的值）'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/llm_experiments',
        help='输出目录'
    )

    args = parser.parse_args()

    # 打印横幅
    print_banner()

    # 1. 加载配置
    print("[步骤 1/5] 加载配置")
    print("-" * 80)

    # 确保配置文件路径是相对于脚本所在目录的
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, args.config)

    try:
        config = load_config(config_file)
        print(f"✓ 配置文件已加载: {args.config}")
    except FileNotFoundError as e:
        print(f"❌ 配置文件不存在: {args.config}")
        print(f"\n提示: 请确保 {args.config} 文件存在于 Stage5_LLM_Framework 目录")
        print(f"      可以从 llm_config_template.json 复制并修改")
        sys.exit(1)

    # 2. 验证API密钥
    print("\n[步骤 2/5] 验证API密钥")
    print("-" * 80)

    available_models = validate_api_keys(config)

    if not available_models:
        print("\n❌ 没有可用的模型！请在配置文件中设置API密钥并启用模型。")
        sys.exit(1)

    print(f"✓ 可用模型: {', '.join(available_models)}")
    print(f"✓ 总计: {len(available_models)} 个模型可用")

    # 3. 选择模型
    print("\n[步骤 3/5] 选择模型")
    print("-" * 80)

    if args.all or args.model == 'all':
        selected_models = available_models
        print(f"✓ 将运行所有 {len(selected_models)} 个模型")
    elif args.model:
        if args.model not in available_models:
            print(f"❌ 模型 '{args.model}' 不可用")
            print(f"可用模型: {', '.join(available_models)}")
            sys.exit(1)
        selected_models = [args.model]
        print(f"✓ 将运行模型: {args.model}")
    else:
        # 交互式选择
        print("\n请选择要运行的模型:")
        for i, model in enumerate(available_models, 1):
            model_info = config["llms"][model]
            print(f"  {i}. {model} ({model_info.get('comment', model_info['model'])})")
        print(f"  {len(available_models)+1}. 运行所有模型")

        try:
            choice = int(input(f"\n请输入选项 (1-{len(available_models)+1}): "))
            if choice == len(available_models) + 1:
                selected_models = available_models
            elif 1 <= choice <= len(available_models):
                selected_models = [available_models[choice-1]]
            else:
                print("❌ 无效选项")
                sys.exit(1)
        except (ValueError, KeyboardInterrupt):
            print("\n已取消")
            sys.exit(0)

    # 4. 加载数据
    print("\n[步骤 4/5] 加载测试数据")
    print("-" * 80)

    try:
        _, _, test_titles, test_labels = DataLoader.prepare_dataset(
            get_data_path('positive.txt'),
            get_data_path('negative.txt'),
            get_data_path('testSet-1000.xlsx')
        )
        print(f"✓ 测试集: {len(test_titles)} 样本")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        sys.exit(1)

    # 确定样本数
    sample_size = args.sample or config["experiment"].get("sample_size", len(test_titles))
    if sample_size > len(test_titles):
        sample_size = len(test_titles)

    print(f"✓ 将使用 {sample_size} 个样本")

    # 5. 运行实验
    print("\n[步骤 5/5] 运行实验")
    print("-" * 80)

    all_results = {}
    total_start_time = time.time()

    for idx, model_name in enumerate(selected_models, 1):
        try:
            print(f"\n{'='*80}")
            print(f"实验 [{idx}/{len(selected_models)}]: {model_name}")
            print(f"{'='*80}")

            results = run_single_experiment(
                model_name,
                config,
                test_titles,
                test_labels,
                sample_size
            )

            all_results[model_name] = results

            # 保存结果
            save_experiment_results(results, model_name, args.output)

            print(f"\n✓ {model_name} 实验完成")

        except Exception as e:
            print(f"\n❌ {model_name} 实验失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start_time

    # 实验总结
    print("\n" + "="*80)
    print(" "*30 + "实验总结")
    print("="*80)

    if len(all_results) == 0:
        print("\n❌ 所有实验都失败了")
        sys.exit(1)

    print(f"\n{'模型':<20} {'准确率':>10} {'召回率':>10} {'F1分数':>10} {'Token消耗':>12} {'平均耗时':>12}")
    print("-" * 90)

    for model_name, results in all_results.items():
        metrics = results["eval_result"]
        stats = results["stats"]
        avg_time = stats["total_time"] / stats["total_calls"] if stats["total_calls"] > 0 else 0

        print(f"{model_name:<20} {metrics['accuracy']*100:>9.2f}% {metrics['recall']*100:>9.2f}% "
              f"{metrics['f1']*100:>9.2f}% {stats['total_tokens']:>12} {avg_time:>11.2f}s")

    print("\n" + "="*80)
    print(" "*30 + "训练完成！")
    print("="*80)
    print(f"\n总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"成功实验: {len(all_results)}/{len(selected_models)}")
    print(f"\n结果保存位置: {args.output}/")
    print(f"\n生成的文件:")
    if len(all_results) > 1:
        print(f"  - llm_comparison.png (模型性能对比)")
        print(f"  - llm_confusion_matrices.png (混淆矩阵)")
    for model_name in all_results.keys():
        print(f"  - {model_name}_*.json (实验结果)")
        print(f"  - {model_name}_*_report.txt (详细报告)")
    print(f"\n注: LLM 模型通过 API 调用，无中间特征向量，无法生成 t-SNE 可视化")
    print()

    # 生成模型对比可视化
    if len(all_results) > 1:
        print("\n" + "="*80)
        print(" "*28 + "生成模型对比图表")
        print("="*80 + "\n")

        try:
            # 准备数据用于可视化
            comparison_results = []
            for model_name, results in all_results.items():
                metrics = results["eval_result"]
                comparison_results.append({
                    "model": f"LLM_{model_name}",
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "confusion_matrix": metrics["confusion_matrix"]
                })

            visualizer = ResultVisualizer()

            # 模型性能对比图
            comparison_path = os.path.join(args.output, "llm_comparison.png")
            visualizer.plot_comparison(comparison_results, save_path=comparison_path)
            print(f"✓ 模型对比图已保存: {comparison_path}")

            # 混淆矩阵对比图
            confusion_path = os.path.join(args.output, "llm_confusion_matrices.png")
            visualizer.plot_confusion_matrices(comparison_results, save_path=confusion_path)
            print(f"✓ 混淆矩阵已保存: {confusion_path}")

        except Exception as e:
            print(f"⚠️  模型对比可视化失败: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

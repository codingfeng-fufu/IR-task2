"""
test_llm_classifier.py
======================
测试和评估LLM In-Context Learning分类器
与其他模型进行对比
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from llm_in_context_classifier import LLMInContextClassifier
from evaluator import ModelEvaluator
import json


def test_llm_classifier():
    """测试LLM分类器并与其他模型对比"""

    print("=" * 80)
    print(" " * 20 + "LLM In-Context Learning 分类器测试")
    print("=" * 80)

    # ========== 1. 加载数据 ==========
    print("\n[步骤 1/4] 加载测试数据")
    print("-" * 80)

    try:
        # 只加载测试集（不需要训练集）
        _, _, test_titles, test_labels = DataLoader.prepare_dataset(
            'data/positive.txt',
            'data/negative.txt',
            'data/testSet-1000.xlsx'
        )

        print(f"✓ 测试集: {len(test_titles)} 样本")
        print(f"  - 正样本: {sum(test_labels)} ({sum(test_labels)/len(test_labels)*100:.1f}%)")
        print(f"  - 负样本: {len(test_labels)-sum(test_labels)} ({(len(test_labels)-sum(test_labels))/len(test_labels)*100:.1f}%)")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("\n提示: 请确保data/目录下有正确的数据文件")
        sys.exit(1)

    # ========== 2. 初始化LLM分类器 ==========
    print("\n[步骤 2/4] 初始化LLM分类器")
    print("-" * 80)

    # 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 未设置OPENAI_API_KEY环境变量")
        print("\n请运行: export OPENAI_API_KEY='your-api-key-here'")
        print("或在代码中直接传入api_key参数\n")
        sys.exit(1)

    try:
        classifier = LLMInContextClassifier(
            provider="openai",
            model="gpt-3.5-turbo",  # 使用GPT-3.5节省成本
            api_key=api_key,
            temperature=0.0  # 确定性输出
        )

        print("✓ LLM分类器初始化成功")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # ========== 3. 进行预测 ==========
    print("\n[步骤 3/4] 进行预测")
    print("-" * 80)

    # 可选：先在小样本上测试
    use_sample = input("\n是否只测试前100个样本？(y/n，完整测试约花费$0.40): ").lower()

    if use_sample == 'y':
        test_titles_sample = test_titles[:100]
        test_labels_sample = test_labels[:100]
        print(f"✓ 将测试前100个样本")
    else:
        test_titles_sample = test_titles
        test_labels_sample = test_labels
        print(f"✓ 将测试全部{len(test_titles)}个样本")

    print(f"\n预计API调用次数: {len(test_titles_sample)}")
    print(f"预计Token消耗: ~{len(test_titles_sample) * 250} tokens")
    print(f"预计成本 (GPT-3.5-turbo): ~${len(test_titles_sample) * 250 / 1000 * 0.002:.2f}\n")

    # 开始预测
    try:
        predictions = classifier.predict(
            test_titles_sample,
            delay=0.3,  # API调用间隔（秒）
            verbose=False
        )

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        sys.exit(1)

    # ========== 4. 评估性能 ==========
    print("\n[步骤 4/4] 评估性能")
    print("-" * 80)

    evaluator = ModelEvaluator()

    # 评估LLM分类器
    result = evaluator.evaluate_model(
        test_labels_sample,
        predictions,
        model_name="LLM (GPT-3.5 Few-shot)",
        verbose=True
    )

    # 保存详细结果
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 保存预测详情（包含LLM的reasoning）
    classifier.save_results(f"{output_dir}/llm_predictions_detailed.json")

    # 保存评估结果
    with open(f"{output_dir}/llm_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 评估结果已保存至: {output_dir}/llm_evaluation.json")

    # ========== 5. 与其他模型对比 ==========
    print("\n" + "=" * 80)
    print(" " * 25 + "与其他模型对比")
    print("=" * 80)

    # 加载其他模型的结果（如果存在）
    comparison_data = [
        {"model": "朴素贝叶斯 V1", "accuracy": 73.46, "f1": 78.82},
        {"model": "朴素贝叶斯 V2", "accuracy": 79.20, "f1": 83.69},
        {"model": "Word2Vec+SVM", "accuracy": 82.99, "f1": 85.74},
        {"model": "BERT V1", "accuracy": 87.91, "f1": 89.59},
        {"model": "SciBERT优化", "accuracy": 89.04, "f1": 90.57},
        {
            "model": "LLM (GPT-3.5)",
            "accuracy": result["accuracy"] * 100,
            "f1": result["f1"] * 100
        }
    ]

    print("\n模型性能对比:")
    print(f"{'模型':<20} {'准确率':>10} {'F1分数':>10}")
    print("-" * 45)
    for item in comparison_data:
        print(f"{item['model']:<20} {item['accuracy']:>9.2f}% {item['f1']:>9.2f}%")

    print("\n" + "=" * 80)

    # ========== 6. 技术路线总结 ==========
    print("\n" + "=" * 80)
    print(" " * 25 + "技术路线总结")
    print("=" * 80)

    print("""
现在您的项目包含了**4种技术路线**：

1. 传统机器学习：朴素贝叶斯（TF-IDF特征）
2. 词嵌入 + 机器学习：Word2Vec + SVM
3. 判别式深度学习：BERT/SciBERT（Fine-tuning）
4. 生成式AI：LLM In-Context Learning（Few-shot，零训练）✨

技术多样性：
✓ 覆盖了从传统ML到最新生成式AI的完整技术栈
✓ 展示了不同范式：判别式 vs 生成式
✓ 展示了不同学习方式：监督学习 vs In-Context Learning
✓ 增强了项目的创新性和学术价值
    """)

    print("=" * 80)
    print(" 测试完成!")
    print("=" * 80)

    # 返回结果供进一步分析
    return {
        "classifier": classifier,
        "predictions": predictions,
        "result": result,
        "comparison": comparison_data
    }


if __name__ == "__main__":
    test_llm_classifier()

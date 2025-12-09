"""
test_optimized_nb.py
====================
测试优化的朴素贝叶斯分类器并与原版本对比
"""

import numpy as np
from data_loader import DataLoader
from naive_bayes_classifier import NaiveBayesClassifier
from naive_bayes_classifier_optimized import NaiveBayesClassifierOptimized
from evaluator import ModelEvaluator


def main():
    print("="*80)
    print(" 朴素贝叶斯分类器优化效果对比")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'data/positive.txt',
        'data/negative.txt',
        'data/testSet-1000.xlsx'
    )

    if len(train_titles) == 0:
        print("❌ 数据加载失败!")
        return

    # ========== 训练原版本 ==========
    print("\n" + "="*80)
    print(" 1. 原版朴素贝叶斯分类器")
    print("="*80)

    nb_original = NaiveBayesClassifier(
        max_features=5000,
        ngram_range=(1, 2),
        model_path='models/naive_bayes_original_model.pkl'
    )

    nb_original.train(train_titles, train_labels, save_model=True)
    predictions_original = nb_original.predict(test_titles)

    # ========== 训练优化版本 ==========
    print("\n" + "="*80)
    print(" 2. 优化版朴素贝叶斯分类器")
    print("="*80)

    nb_optimized = NaiveBayesClassifierOptimized(
        max_features_word=10000,
        max_features_char=5000,
        word_ngram_range=(1, 3),
        char_ngram_range=(3, 5),
        alpha=0.5,
        use_complement_nb=True,
        add_statistical_features=True,
        model_path='models/naive_bayes_optimized_model.pkl'
    )

    nb_optimized.train(train_titles, train_labels, save_model=True)
    predictions_optimized = nb_optimized.predict(test_titles)

    # ========== 评估对比 ==========
    print("\n" + "="*80)
    print(" 3. 性能对比")
    print("="*80)

    evaluator = ModelEvaluator()

    # 评估原版本
    result_original = evaluator.evaluate_model(
        test_labels,
        predictions_original,
        "原版 Naive Bayes",
        verbose=True
    )

    # 评估优化版本
    result_optimized = evaluator.evaluate_model(
        test_labels,
        predictions_optimized,
        "优化版 Naive Bayes",
        verbose=True
    )

    # 对比结果
    evaluator.compare_models([result_original, result_optimized])

    # ========== 改进分析 ==========
    print("\n" + "="*80)
    print(" 4. 改进分析")
    print("="*80)

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_micro']
    metric_names = {
        'accuracy': '准确率',
        'precision': '精确率',
        'recall': '召回率',
        'f1': 'F1分数',
        'f1_macro': 'F1宏平均',
        'f1_micro': 'F1微平均'
    }

    print("\n指标提升:")
    print(f"{'指标':<15} {'原版':<12} {'优化版':<12} {'提升':<12} {'提升率':<10}")
    print("-" * 70)

    for metric in metrics:
        original_val = result_original[metric]
        optimized_val = result_optimized[metric]
        improvement = optimized_val - original_val
        improvement_pct = (improvement / original_val * 100) if original_val > 0 else 0

        print(f"{metric_names[metric]:<15} {original_val:<12.4f} {optimized_val:<12.4f} "
              f"{improvement:+<12.4f} {improvement_pct:+.2f}%")

    # 总结
    print("\n" + "="*80)
    print(" 总结")
    print("="*80)

    avg_improvement = np.mean([
        result_optimized[m] - result_original[m] for m in metrics
    ])

    print(f"\n平均性能提升: {avg_improvement:+.4f} ({avg_improvement*100:+.2f}%)")

    if result_optimized['accuracy'] > result_original['accuracy']:
        print(f"✓ 优化成功! 准确率从 {result_original['accuracy']:.4f} 提升到 {result_optimized['accuracy']:.4f}")
    else:
        print("⚠️  优化未达到预期效果")

    print("\n优化方法:")
    print("  1. 增加词级特征数量: 5000 → 10000")
    print("  2. 扩展n-gram范围: (1,2) → (1,3)")
    print("  3. 添加字符级n-gram特征: (3,5)")
    print("  4. 添加23个统计特征（长度、标点、特殊模式等）")
    print("  5. 使用ComplementNB替代MultinomialNB")
    print("  6. 调整平滑参数: alpha=1.0 → 0.5")
    print("  7. 添加max_df过滤高频词")


if __name__ == "__main__":
    main()

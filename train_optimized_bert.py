"""
train_optimized_bert.py
=======================
使用优化版BERT分类器训练学术标题分类模型
"""

import sys
from data_loader import DataLoader
from optimized_BERT import BERTClassifierOptimized
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


def main():
    print("=" * 80)
    print("  优化版 BERT 分类器 - 学术标题分类")
    print("=" * 80)
    
    # ========== 加载数据 ==========
    print("\n[步骤 1/3] 加载数据集")
    print("-" * 80)

    try:
        train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
            'data/positive.txt',
            'data/negative.txt',
            'data/testSet-1000.xlsx'
        )

        print(f"✓ 训练集: {len(train_titles)} 样本")
        print(f"  - 正样本: {sum(train_labels)} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
        print(f"  - 负样本: {len(train_labels)-sum(train_labels)} ({(len(train_labels)-sum(train_labels))/len(train_labels)*100:.1f}%)")
        print(f"✓ 测试集: {len(test_titles)} 样本")

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 创建和训练模型 ==========
    print("\n[步骤 2/3] 训练优化版 BERT 分类器")
    print("-" * 80)
    
    try:
        classifier = BERTClassifierOptimized(
            model_name='bert-base-uncased',
            max_length=64,
            use_fgm=True,
            use_ema=True
        )
        
        best_val_f1 = classifier.train(
            train_titles,
            train_labels,
            val_titles=None,
            val_labels=None,
            epochs=10,
            batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            patience=3,
            use_focal_loss=False,
            augment_data=True
        )
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 评估模型 ==========
    print("\n[步骤 3/3] 评估模型")
    print("-" * 80)
    
    try:
        # 在测试集上进行预测
        print("\n在测试集上进行预测...")
        predictions = classifier.predict(test_titles, batch_size=16)
        probabilities = classifier.predict_proba(test_titles, batch_size=16)
        
        # 计算各项指标
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, pos_label=1, zero_division=0)
        recall = recall_score(test_labels, predictions, pos_label=1, zero_division=0)
        f1 = f1_score(test_labels, predictions, pos_label=1, zero_division=0)
        
        # 显示预测结果示例
        print("\n预测结果示例:")
        print(f"{'标题':<50} {'真实':<8} {'预测':<8} {'置信度':<10}")
        print("-" * 80)
        
        for i in range(min(10, len(test_titles))):
            title = test_titles[i][:47] + "..." if len(test_titles[i]) > 50 else test_titles[i]
            true_label = "正确" if test_labels[i] == 1 else "错误"
            pred_label = "正确" if predictions[i] == 1 else "错误"
            confidence = probabilities[i][predictions[i]]
            
            print(f"{title:<50} {true_label:<8} {pred_label:<8} {confidence:.3f}")
        
        # 保存结果
        output_results = {
            'model': 'BERTClassifierOptimized',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'best_val_f1': float(best_val_f1),
            'test_samples': len(test_titles),
            'train_samples': len(train_titles)
        }
        
        import os
        os.makedirs('output', exist_ok=True)
        with open('output/optimized_bert_results.json', 'w') as f:
            json.dump(output_results, f, indent=2)
        
        print(f"\n✓ 结果已保存到 output/optimized_bert_results.json")
        
        # 打印总结
        print("\n" + "=" * 80)
        print("  最终结果总结")
        print("=" * 80)
        print(f"验证集最佳 F1 分数: {best_val_f1:.4f}")
        print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"测试集精确率: {precision:.4f}")
        print(f"测试集召回率: {recall:.4f}")
        print(f"测试集 F1 分数: {f1:.4f}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

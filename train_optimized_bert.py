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
        # 使用 prepare_dataset 方法加载数据
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
        # 创建分类器
        classifier = BERTClassifierOptimized(
            model_name='bert-base-uncased',
            max_length=64,
            use_fgm=True,      # 对抗训练
            use_ema=True       # 指数移动平均
        )
        
        # 训练模型
        best_val_f1 = classifier.train(
            train_titles,
            train_labels,
            val_titles=None,           # 自动从训练集划分20%作为验证集
            val_labels=None,
            epochs=10,                 # 10 轮训练
            batch_size=16,             # 批次大小 16
            learning_rate=2e-5,        # BERT 推荐学习率
            warmup_ratio=0.1,          # 10% 步数用于预热
            weight_decay=0.01,         # L2 正则化
            patience=3,                # 3 轮不提升就早停
            use_focal_loss=False,      # 不使用 Focal Loss
            augment_data=True          # 数据增强
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
        # 在测试集上进行详细评估
        results = classifier.detailed_evaluation(test_titles, test_labels, batch_size=16)
        
        # 保存结果
        output_results = {
            'model': 'BERTClassifierOptimized',
            'accuracy': float(results['accuracy']),
            'f1_score': float(results['f1']),
            'best_val_f1': float(best_val_f1),
            'test_samples': len(test_titles),
            'train_samples': len(train_titles),
            'optimizations': {
                'fgm_adversarial_training': True,
                'ema_exponential_moving_average': True,
                'differential_learning_rate': True,
                'cosine_schedule_with_warmup': True,
                'gradient_clipping': True,
                'data_augmentation': True,
                'early_stopping': True,
                'validation_based_early_stopping': True
            }
        }
        
        # 保存到文件
        with open('output/optimized_bert_results.json', 'w') as f:
            json.dump(output_results, f, indent=2)
        
        print(f"\n✓ 结果已保存到 output/optimized_bert_results.json")
        
        # 打印总结
        print("\n" + "=" * 80)
        print("  最终结果总结")
        print("=" * 80)
        print(f"验证集最佳 F1 分数: {best_val_f1:.4f}")
        print(f"测试集准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"测试集 F1 分数: {results['f1']:.4f}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


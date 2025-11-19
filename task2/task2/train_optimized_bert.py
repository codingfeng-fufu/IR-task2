"""
train_optimized_bert.py
=======================
使用优化版BERT分类器训练学术标题分类模型
包含完整的评估报告和可视化输出
"""

import sys
import os

# 添加 core 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from data_loader import DataLoader
from optimized_BERT import BERTClassifierOptimized
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
import json


def main():
    print("=" * 80)
    print("  优化版 BERT 分类器 - 学术标题分类")
    print("=" * 80)
    
    # ========== 加载数据 ==========
    print("\n[步骤 1/5] 加载数据集")
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
    print("\n[步骤 2/5] 训练优化版 BERT 分类器")
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
    print("\n[步骤 3/5] 评估模型")
    print("-" * 80)
    
    try:
        # 在测试集上进行预测
        print("\n在测试集上进行预测...")
        predictions = classifier.predict(test_titles, batch_size=16)
        probabilities = classifier.predict_proba(test_titles, batch_size=16)
        
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
        
        # 使用 ModelEvaluator 进行详细评估
        print("\n" + "=" * 80)
        print("  详细评估报告")
        print("=" * 80)
        
        evaluator = ModelEvaluator()
        result = evaluator.evaluate_model(
            test_labels, 
            predictions, 
            'BERT (Optimized)',
            verbose=True
        )
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 生成可视化 ==========
    print("\n[步骤 4/5] 生成可视化图表")
    print("-" * 80)
    
    try:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        visualizer = ResultVisualizer()
        
        # 1. 生成性能对比图
        print("\n生成性能对比图...")
        visualizer.plot_comparison(
            [result],
            save_path=os.path.join(output_dir, 'bert_performance.png')
        )
        
        # 2. 生成混淆矩阵
        print("\n生成混淆矩阵...")
        visualizer.plot_confusion_matrices(
            [result],
            save_path=os.path.join(output_dir, 'bert_confusion_matrix.png')
        )
        
        # 3. 生成 t-SNE 可视化
        print("\n生成 t-SNE 可视化图（提取特征向量）...")
        feature_vectors = classifier.get_feature_vectors(test_titles, batch_size=16)
        visualizer.visualize_embeddings_tsne(
            feature_vectors,
            test_labels,
            'BERT (Optimized)',
            save_path=os.path.join(output_dir, 'bert_tsne_visualization.png'),
            perplexity=30,
            n_iter=1000
        )
        
        print(f"\n✓ 所有可视化图表已保存到 {output_dir}/ 目录")
        
    except Exception as e:
        print(f"⚠️  可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 保存结果 ==========
    print("\n[步骤 5/5] 保存结果")
    print("-" * 80)
    
    try:
        # 保存详细评估结果
        results_file = os.path.join(output_dir, 'bert_evaluation_results.txt')
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(" 优化版 BERT 分类器 - 评估结果\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("【训练信息】\n")
            f.write(f"  训练样本数: {len(train_titles)}\n")
            f.write(f"  测试样本数: {len(test_titles)}\n")
            f.write(f"  验证集最佳 F1: {best_val_f1:.4f}\n\n")
            
            f.write("【测试集性能】\n")
            f.write(f"  准确率 (Accuracy):     {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
            f.write(f"  精确率 (Precision):    {result['precision']:.4f}\n")
            f.write(f"  召回率 (Recall):       {result['recall']:.4f}\n")
            f.write(f"  F1分数 (F1-Score):     {result['f1']:.4f}\n")
            f.write(f"  F1宏平均 (F1-Macro):   {result['f1_macro']:.4f}\n")
            f.write(f"  F1微平均 (F1-Micro):   {result['f1_micro']:.4f}\n\n")
            
            f.write("【混淆矩阵】\n")
            cm = result['confusion_matrix']
            f.write(f"  真负例 (TN): {result['tn']}\n")
            f.write(f"  假正例 (FP): {result['fp']}\n")
            f.write(f"  假负例 (FN): {result['fn']}\n")
            f.write(f"  真正例 (TP): {result['tp']}\n")
            f.write(f"  特异度 (Specificity):  {result['specificity']:.4f}\n")
            f.write(f"  敏感度 (Sensitivity):  {result['sensitivity']:.4f}\n")
        
        print(f"✓ 评估结果已保存到: {results_file}")
        
        # 保存 JSON 格式结果
        json_results = {
            'model': 'BERT (Optimized)',
            'training_info': {
                'train_samples': len(train_titles),
                'test_samples': len(test_titles),
                'best_val_f1': float(best_val_f1)
            },
            'test_performance': {
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1']),
                'f1_macro': float(result['f1_macro']),
                'f1_micro': float(result['f1_micro'])
            },
            'confusion_matrix': {
                'tn': int(result['tn']),
                'fp': int(result['fp']),
                'fn': int(result['fn']),
                'tp': int(result['tp']),
                'specificity': float(result['specificity']),
                'sensitivity': float(result['sensitivity'])
            }
        }
        
        json_file = os.path.join(output_dir, 'bert_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON 结果已保存到: {json_file}")
        
    except Exception as e:
        print(f"⚠️  结果保存失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 最终总结 ==========
    print("\n" + "=" * 80)
    print("  训练完成！")
    print("=" * 80)
    print(f"\n📊 模型性能:")
    print(f"  验证集最佳 F1:   {best_val_f1:.4f}")
    print(f"  测试集准确率:     {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print(f"  测试集 F1 分数:   {result['f1']:.4f}")
    
    print(f"\n📁 输出文件:")
    print(f"  1. {output_dir}/bert_performance.png - 性能指标对比图")
    print(f"  2. {output_dir}/bert_confusion_matrix.png - 混淆矩阵图")
    print(f"  3. {output_dir}/bert_tsne_visualization.png - t-SNE 可视化图")
    print(f"  4. {output_dir}/bert_evaluation_results.txt - 详细评估报告")
    print(f"  5. {output_dir}/bert_results.json - JSON 格式结果")
    
    print("\n" + "=" * 80)
    print("  感谢使用优化版 BERT 分类器！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

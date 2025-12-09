"""
main.py
=======
主程序 - 运行完整的学术标题分类流水线
集成所有模块,完成数据加载、模型训练、评估和可视化
"""

import sys
import os
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_loader import DataLoader, create_sample_data
from naive_bayes_classifier_optimized import NaiveBayesClassifierOptimized  # 使用优化版本
from word2vec_svm_classifier import Word2VecSVMClassifier
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer


def print_header():
    """打印程序头部信息"""
    print("\n" + "="*80)
    print(" " * 15 + "学术标题分类系统")
    print(" " * 10 + "Scholar Title Classification System")
    print("="*80)
    print("\n项目描述:")
    print("  识别CiteSeer数据库中错误提取的学术论文标题")
    print("  使用三种机器学习方法:")
    print("    1. 朴素贝叶斯 (Naive Bayes - 优化版)")
    print("    2. Word2Vec + SVM")
    print("    3. BERT (Transformer)")
    print("\n" + "="*80 + "\n")


def load_data(use_sample_data: bool = False, max_train_samples: int = None):
    """
    加载数据集

    参数:
        use_sample_data: 是否使用示例数据
        max_train_samples: 最大训练样本数（用于加快训练速度）

    返回:
        (train_titles, train_labels, test_titles, test_labels)
    """
    print("[步骤 1/5] 加载数据集")
    print("-" * 80)

    if use_sample_data:
        print("使用示例数据...")
        train_titles, train_labels, test_titles, test_labels = create_sample_data()
    else:
        # 尝试加载实际文件
        train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
            'data/positive.txt',
            'data/negative.txt',
            'data/testSet-1000.xlsx'
        )

        # 如果文件不存在,使用示例数据
        if len(train_titles) == 0:
            print("\n⚠️  未找到数据文件,切换到示例数据...")
            train_titles, train_labels, test_titles, test_labels = create_sample_data()

        # 限制训练集大小以加快训练速度
        if max_train_samples and len(train_titles) > max_train_samples:
            print(f"\n⚠️  为加快训练速度，从 {len(train_titles)} 条中随机采样 {max_train_samples} 条训练数据")
            import random
            random.seed(42)
            indices = list(range(len(train_titles)))
            random.shuffle(indices)
            indices = indices[:max_train_samples]
            train_titles = [train_titles[i] for i in indices]
            train_labels = [train_labels[i] for i in indices]

    print(f"\n数据加载完成!")
    print(f"  训练集: {len(train_titles)} 条")
    print(f"  测试集: {len(test_titles)} 条")

    return train_titles, train_labels, test_titles, test_labels


def train_naive_bayes(train_titles: List[str], train_labels: List[int]):
    """训练朴素贝叶斯分类器（优化版）"""
    print("\n[步骤 2/5] 训练朴素贝叶斯分类器（优化版）")
    print("-" * 80)

    classifier = NaiveBayesClassifierOptimized(
        max_features_word=10000,
        max_features_char=5000,
        word_ngram_range=(1, 3),
        char_ngram_range=(3, 5),
        alpha=0.5,
        use_complement_nb=True,
        add_statistical_features=True,
        model_path='models/naive_bayes_optimized_model.pkl'
    )
    classifier.train(train_titles, train_labels)

    return classifier


def train_word2vec_svm(train_titles: List[str], train_labels: List[int]):
    """训练Word2Vec+SVM分类器（优化版）"""
    print("\n[步骤 3/5] 训练 Word2Vec + SVM 分类器（优化版）")
    print("-" * 80)

    classifier = Word2VecSVMClassifier(
        vector_size=100,
        window=5,
        min_count=2,
        epochs=10,
        use_linear_svm=False,  # 使用 RBF 核 SVM（更强的非线性能力）
        add_features=True      # 添加统计特征（特征工程）
    )
    classifier.train(train_titles, train_labels)

    return classifier


def train_bert(train_titles: List[str], train_labels: List[int], epochs: int = 5, use_full_data: bool = True):
    """训练BERT分类器（优化版v2）"""
    print("\n[步骤 4/5] 训练 BERT 分类器（优化版v2）")
    print("-" * 80)

    try:
        from bert_classifier import BERTClassifier
        classifier = BERTClassifier(model_name='bert-base-uncased', max_length=64)

        # 根据配置决定是否使用全部数据
        if not use_full_data and len(train_titles) > 10000:
            print(f"\n注意: 为加快训练速度,只使用前10000条训练数据")
            train_titles = train_titles[:10000]
            train_labels = train_labels[:10000]
        else:
            print(f"\n使用全部 {len(train_titles)} 条训练数据")

        classifier.train(
            train_titles,
            train_labels,
            epochs=epochs,        # 增加到 5 轮
            batch_size=32,        # 保持 32（更好的泛化能力）
            learning_rate=2e-5,
            warmup_steps=500      # 添加学习率预热
        )

        return classifier
    except Exception as e:
        print(f"\n⚠️  BERT训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        print("跳过BERT模型...")
        return None


def evaluate_models(classifiers: Dict, test_titles: List[str], test_labels: List[int]):
    """评估所有模型"""
    print("\n[步骤 5/5] 评估模型性能")
    print("-" * 80)
    
    evaluator = ModelEvaluator()
    results = []
    predictions_dict = {}
    
    for model_name, classifier in classifiers.items():
        if classifier is None:
            continue
        
        print(f"\n正在评估 {model_name}...")
        
        # 进行预测
        predictions = classifier.predict(test_titles)
        predictions_dict[model_name] = predictions
        
        # 评估
        result = evaluator.evaluate_model(
            test_labels,
            predictions,
            model_name,
            verbose=True
        )
        results.append(result)
    
    # 比较模型
    if len(results) > 1:
        evaluator.compare_models(results)
    
    return results, predictions_dict


def generate_visualizations(
    classifiers: Dict,
    results: List[Dict],
    test_titles: List[str],
    test_labels: List[int],
    output_dir: str = 'output'
):
    """生成可视化图表"""
    print("\n" + "="*80)
    print(" 生成可视化图表")
    print("="*80)
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}/")
    
    visualizer = ResultVisualizer()
    
    # 1. 性能对比图
    visualizer.plot_comparison(
        results,
        save_path=os.path.join(output_dir, 'model_comparison.png')
    )
    
    # 2. 混淆矩阵
    visualizer.plot_confusion_matrices(
        results,
        save_path=os.path.join(output_dir, 'confusion_matrices.png')
    )
    
    # 3. t-SNE可视化(为每个模型生成)
    for model_name, classifier in classifiers.items():
        if classifier is None:
            continue
        
        try:
            # 获取特征向量
            feature_vectors = classifier.get_feature_vectors(test_titles)
            
            # 生成t-SNE可视化
            safe_name = model_name.replace(' ', '_').replace('+', '_')
            visualizer.visualize_embeddings_tsne(
                feature_vectors,
                test_labels,
                model_name,
                save_path=os.path.join(output_dir, f'tsne_{safe_name}.png'),
                perplexity=30,
                n_iter=1000
            )
        except Exception as e:
            print(f"⚠️  生成 {model_name} 的t-SNE图失败: {str(e)}")


def save_results(results: List[Dict], predictions_dict: Dict, output_dir: str = 'output'):
    """保存结果到文件"""
    import json
    
    print("\n" + "="*80)
    print(" 保存结果")
    print("="*80)
    
    # 保存评估结果
    results_file = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" 模型评估结果\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"模型: {result['model']}\n")
            f.write(f"  准确率: {result['accuracy']:.4f}\n")
            f.write(f"  精确率: {result['precision']:.4f}\n")
            f.write(f"  召回率: {result['recall']:.4f}\n")
            f.write(f"  F1分数: {result['f1']:.4f}\n")
            f.write(f"  F1宏平均: {result['f1_macro']:.4f}\n")
            f.write(f"  F1微平均: {result['f1_micro']:.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"✓ 评估结果已保存至: {results_file}")
    
    # 保存预测结果
    predictions_file = os.path.join(output_dir, 'predictions.json')
    predictions_json = {
        model: pred.tolist() if hasattr(pred, 'tolist') else pred
        for model, pred in predictions_dict.items()
    }
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_json, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 预测结果已保存至: {predictions_file}")


def main():
    """主函数:运行完整的分类流水线"""
    
    # 打印头部信息
    print_header()
    
    # 配置选项（只训练 BERT）
    USE_SAMPLE_DATA = False  # 是否使用示例数据
    MAX_TRAIN_SAMPLES = None  # 最大训练样本数（None=使用全部数据）
    TRAIN_ONLY_BERT = False  # 只训练 BERT
    BERT_EPOCHS = 5  # BERT训练轮数（从 3 增加到 5）
    OUTPUT_DIR = 'output'  # 输出目录

    try:
        # 1. 加载数据
        train_titles, train_labels, test_titles, test_labels = load_data(USE_SAMPLE_DATA, MAX_TRAIN_SAMPLES)

        if TRAIN_ONLY_BERT:
            # 只训练 BERT
            print("\n" + "="*80)
            print("  模式：只训练 BERT（跳过 Naive Bayes 和 Word2Vec+SVM）")
            print("="*80)

            bert_classifier = train_bert(train_titles, train_labels, epochs=BERT_EPOCHS, use_full_data=True)

            # 只收集 BERT 分类器
            classifiers = {
                'BERT': bert_classifier
            }
        else:
            # 完整训练流程
            # 2. 训练朴素贝叶斯
            nb_classifier = train_naive_bayes(train_titles, train_labels)

            # 3. 训练Word2Vec+SVM
            w2v_classifier = train_word2vec_svm(train_titles, train_labels)

            # 4. 训练BERT
            bert_classifier = train_bert(train_titles, train_labels, epochs=BERT_EPOCHS, use_full_data=True)

            # 5. 收集所有分类器
            classifiers = {
                'Naive Bayes': nb_classifier,
                'Word2Vec+SVM': w2v_classifier,
                'BERT': bert_classifier
            }
        
        # 6. 评估模型
        results, predictions_dict = evaluate_models(classifiers, test_titles, test_labels)
        
        # 7. 生成可视化
        generate_visualizations(
            classifiers,
            results,
            test_titles,
            test_labels,
            output_dir=OUTPUT_DIR
        )
        
        # 8. 保存结果
        save_results(results, predictions_dict, output_dir=OUTPUT_DIR)
        
        # 完成
        print("\n" + "="*80)
        print(" 流水线执行完成!")
        print("="*80)
        print(f"\n所有结果已保存至 '{OUTPUT_DIR}/' 目录")
        print("\n生成的文件:")
        print("  1. model_comparison.png - 模型性能对比图")
        print("  2. confusion_matrices.png - 混淆矩阵")
        print("  3. tsne_*.png - t-SNE可视化图")
        print("  4. evaluation_results.txt - 详细评估结果")
        print("  5. predictions.json - 预测结果")
        print("\n感谢使用学术标题分类系统!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

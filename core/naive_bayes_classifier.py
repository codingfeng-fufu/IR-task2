"""
naive_bayes_classifier.py
==========================
朴素贝叶斯分类器实现
使用TF-IDF特征和MultinomialNB
"""

import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from data_loader import DataLoader


class NaiveBayesClassifier:
    """
    基于TF-IDF特征的朴素贝叶斯分类器
    用于识别错误提取的学术论文标题
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        初始化朴素贝叶斯分类器
        
        参数:
            max_features: TF-IDF的最大特征数
            ngram_range: n-gram范围,默认使用unigrams和bigrams
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # TF-IDF向量化器 - 将文本转换为数值特征向量
        # TF-IDF = Term Frequency(词频) × Inverse Document Frequency(逆文档频率)
        # 作用: 衡量一个词对文档的重要性
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,     # 最多保留5000个最重要的特征(词/n-gram)
            ngram_range=ngram_range,       # (1,2)表示使用1-gram和2-gram，如"machine"和"machine learning"
            min_df=2,                      # 最小文档频率：忽略出现次数少于2次的词(去除罕见词)
            preprocessor=DataLoader.preprocess_title,  # 预处理函数(小写、去特殊字符)
            sublinear_tf=True              # 使用对数TF缩放：tf_scaled = 1 + log(tf)，减少高频词的影响
        )
        
        # 朴素贝叶斯分类器,使用Laplace平滑
        self.classifier = MultinomialNB(alpha=1.0)
        
        # 训练状态
        self.is_trained = False
        
    def train(self, titles: List[str], labels: List[int]):
        """
        训练朴素贝叶斯分类器
        
        参数:
            titles: 训练标题列表
            labels: 训练标签列表 (0=错误标题, 1=正确标题)
        """
        print("\n" + "="*60)
        print("训练朴素贝叶斯分类器")
        print("="*60)
        
        # 将标题转换为TF-IDF特征
        print("步骤 1/2: 提取TF-IDF特征...")
        X_train = self.vectorizer.fit_transform(titles)
        
        print(f"  - 词汇表大小: {len(self.vectorizer.vocabulary_)}")
        print(f"  - 特征矩阵形状: {X_train.shape}")
        print(f"  - n-gram范围: {self.ngram_range}")
        
        # 训练分类器
        print("\n步骤 2/2: 训练分类器...")
        self.classifier.fit(X_train, labels)
        
        self.is_trained = True
        print("✓ 训练完成!")
        
        # 显示一些统计信息
        self._print_training_stats(X_train, labels)
        
    def _print_training_stats(self, X_train, labels):
        """打印训练统计信息"""
        print("\n训练集统计:")
        print(f"  - 样本总数: {len(labels)}")
        print(f"  - 正样本数: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        print(f"  - 负样本数: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
        print(f"  - 特征稀疏度: {(1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]))*100:.2f}%")
        
    def predict(self, titles: List[str]) -> np.ndarray:
        """
        预测给定标题的标签
        
        参数:
            titles: 待分类的标题列表
            
        返回:
            预测标签数组 (0=错误标题, 1=正确标题)
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法")
        
        X_test = self.vectorizer.transform(titles)
        predictions = self.classifier.predict(X_test)
        
        return predictions
    
    def predict_proba(self, titles: List[str]) -> np.ndarray:
        """
        预测给定标题的概率
        
        参数:
            titles: 待分类的标题列表
            
        返回:
            概率数组,形状为 (n_samples, 2)
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法")
        
        X_test = self.vectorizer.transform(titles)
        probabilities = self.classifier.predict_proba(X_test)
        
        return probabilities
    
    def get_feature_vectors(self, titles: List[str]) -> np.ndarray:
        """
        获取标题的TF-IDF特征向量(用于可视化)
        
        参数:
            titles: 标题列表
            
        返回:
            TF-IDF特征矩阵
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法")
        
        return self.vectorizer.transform(titles).toarray()
    
    def get_top_features(self, n=20):
        """
        获取每个类别最重要的特征(词)
        
        参数:
            n: 返回的top特征数量
            
        返回:
            字典,包含每个类别的top特征
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法")
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # 获取每个类别的对数概率
        log_probs = self.classifier.feature_log_prob_
        
        results = {}
        class_names = ['错误标题', '正确标题']
        
        for idx, class_name in enumerate(class_names):
            # 获取该类别的top特征索引
            top_indices = np.argsort(log_probs[idx])[-n:][::-1]
            top_features = feature_names[top_indices]
            top_probs = log_probs[idx][top_indices]
            
            results[class_name] = list(zip(top_features, top_probs))
        
        return results


def main():
    """
    主函数:演示朴素贝叶斯分类器的使用
    """
    from data_loader import DataLoader, create_sample_data
    
    print("="*70)
    print(" 朴素贝叶斯分类器演示")
    print("="*70)
    
    # 加载数据
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'positive_titles.txt',
        'negative_titles.txt',
        'test_titles.txt'
    )
    
    # 如果没有实际文件,使用示例数据
    if len(train_titles) == 0:
        train_titles, train_labels, test_titles, test_labels = create_sample_data()
    
    # 创建并训练分类器
    classifier = NaiveBayesClassifier(max_features=5000, ngram_range=(1, 2))
    classifier.train(train_titles, train_labels)
    
    # 进行预测
    print("\n" + "="*60)
    print("在测试集上进行预测")
    print("="*60)
    
    predictions = classifier.predict(test_titles)
    probabilities = classifier.predict_proba(test_titles)
    
    # 显示一些预测结果
    print("\n预测结果示例:")
    print(f"{'标题':<50} {'真实':<8} {'预测':<8} {'置信度':<10}")
    print("-" * 80)
    
    for i in range(min(10, len(test_titles))):
        title = test_titles[i][:47] + "..." if len(test_titles[i]) > 50 else test_titles[i]
        true_label = "正确" if test_labels[i] == 1 else "错误"
        pred_label = "正确" if predictions[i] == 1 else "错误"
        confidence = probabilities[i][predictions[i]]
        
        print(f"{title:<50} {true_label:<8} {pred_label:<8} {confidence:.3f}")
    
    # 计算准确率
    accuracy = np.mean(predictions == test_labels)
    print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 显示最重要的特征
    print("\n" + "="*60)
    print("最重要的特征词")
    print("="*60)
    
    top_features = classifier.get_top_features(n=10)
    
    for class_name, features in top_features.items():
        print(f"\n{class_name}:")
        for i, (feature, prob) in enumerate(features, 1):
            print(f"  {i}. {feature} (log_prob: {prob:.4f})")


if __name__ == "__main__":
    main()

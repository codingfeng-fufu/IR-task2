"""
naive_bayes_classifier.py (增强版)
==========================
朴素贝叶斯分类器实现 - 添加了模型保存/加载功能
使用TF-IDF特征和MultinomialNB
"""

import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from data_loader import DataLoader
import joblib
import os


class NaiveBayesClassifier:
    """
    基于TF-IDF特征的朴素贝叶斯分类器
    用于识别错误提取的学术论文标题
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), model_path='models/naive_bayes_model.pkl'):
        """
        初始化朴素贝叶斯分类器
        
        参数:
            max_features: TF-IDF的最大特征数
            ngram_range: n-gram范围,默认使用unigrams和bigrams
            model_path: 模型保存路径
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.model_path = model_path
        
        # TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # 忽略出现次数少于2次的词
            preprocessor=DataLoader.preprocess_title,
            sublinear_tf=True  # 使用对数TF缩放
        )
        
        # 朴素贝叶斯分类器,使用Laplace平滑
        self.classifier = MultinomialNB(alpha=1.0)
        
        # 训练状态
        self.is_trained = False
        
    def train(self, titles: List[str], labels: List[int], save_model=True):
        """
        训练朴素贝叶斯分类器
        
        参数:
            titles: 训练标题列表
            labels: 训练标签列表 (0=错误标题, 1=正确标题)
            save_model: 是否保存模型
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
        
        # 保存模型
        if save_model:
            self.save_model()
        
    def _print_training_stats(self, X_train, labels):
        """打印训练统计信息"""
        print("\n训练集统计:")
        print(f"  - 样本总数: {len(labels)}")
        print(f"  - 正样本数: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        print(f"  - 负样本数: {len(labels)-sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
        print(f"  - 特征稀疏度: {(1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]))*100:.2f}%")
    
    def save_model(self):
        """保存模型到文件"""
        # 创建目录
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # 保存整个分类器对象
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, self.model_path)
        print(f"\n✓ 模型已保存至: {self.model_path}")
        
    def load_model(self):
        """从文件加载模型"""
        if not os.path.exists(self.model_path):
            print(f"⚠️  模型文件不存在: {self.model_path}")
            return False
        
        print(f"加载模型: {self.model_path}")
        model_data = joblib.load(self.model_path)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data['ngram_range']
        self.is_trained = model_data['is_trained']
        
        print("✓ 模型加载成功!")
        return True
        
    def predict(self, titles: List[str]) -> np.ndarray:
        """
        预测给定标题的标签
        
        参数:
            titles: 待分类的标题列表
            
        返回:
            预测标签数组 (0=错误标题, 1=正确标题)
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法或load_model()方法")
        
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
            raise ValueError("模型尚未训练!请先调用train()方法或load_model()方法")
        
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
            raise ValueError("模型尚未训练!请先调用train()方法或load_model()方法")
        
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
            raise ValueError("模型尚未训练!请先调用train()方法或load_model()方法")
        
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
    print(" 朴素贝叶斯分类器演示（增强版 - 支持保存/加载）")
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
    
    # 创建分类器
    classifier = NaiveBayesClassifier(
        max_features=5000,
        ngram_range=(1, 2),
        model_path='models/naive_bayes_model.pkl'
    )
    
    # 尝试加载已有模型
    if not classifier.load_model():
        # 如果没有已有模型，则训练新模型
        classifier.train(train_titles, train_labels, save_model=True)
    else:
        print("使用已加载的模型进行预测")
    
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


if __name__ == "__main__":
    main()
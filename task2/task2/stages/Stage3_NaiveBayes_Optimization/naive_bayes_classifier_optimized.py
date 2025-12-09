"""
naive_bayes_classifier_optimized.py
====================================
朴素贝叶斯分类器优化版
使用多种特征组合和集成学习提升性能
"""

import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from data_loader import DataLoader
import joblib
import os
import re


class NaiveBayesClassifierOptimized:
    """
    优化的朴素贝叶斯分类器
    改进点:
    1. 增加特征数量和多样性（词级+字符级n-gram）
    2. 添加统计特征（特征工程）
    3. 使用ComplementNB（对不平衡数据更好）
    4. 更好的参数调优
    """

    def __init__(self,
                 max_features_word=10000,
                 max_features_char=5000,
                 word_ngram_range=(1, 3),
                 char_ngram_range=(3, 5),
                 alpha=0.5,
                 use_complement_nb=True,
                 add_statistical_features=True,
                 model_path='models/naive_bayes_optimized_model.pkl'):
        """
        初始化优化的朴素贝叶斯分类器

        参数:
            max_features_word: 词级TF-IDF最大特征数（增加到10000）
            max_features_char: 字符级TF-IDF最大特征数
            word_ngram_range: 词级n-gram范围（扩展到trigrams）
            char_ngram_range: 字符级n-gram范围
            alpha: Laplace平滑参数（调整为0.5）
            use_complement_nb: 使用ComplementNB而非MultinomialNB
            add_statistical_features: 是否添加统计特征
            model_path: 模型保存路径
        """
        self.max_features_word = max_features_word
        self.max_features_char = max_features_char
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.alpha = alpha
        self.use_complement_nb = use_complement_nb
        self.add_statistical_features = add_statistical_features
        self.model_path = model_path

        # 词级TF-IDF向量化器
        self.word_vectorizer = TfidfVectorizer(
            max_features=max_features_word,
            ngram_range=word_ngram_range,
            min_df=2,
            max_df=0.95,  # 过滤出现在95%以上文档中的词
            preprocessor=DataLoader.preprocess_title,
            sublinear_tf=True,
            strip_accents='unicode',
            token_pattern=r'\b\w+\b'  # 更宽松的token模式
        )

        # 字符级TF-IDF向量化器（捕捉拼写模式）
        self.char_vectorizer = TfidfVectorizer(
            max_features=max_features_char,
            ngram_range=char_ngram_range,
            analyzer='char',
            min_df=2,
            sublinear_tf=True
        )

        # 选择分类器类型
        if use_complement_nb:
            # ComplementNB: 更适合不平衡数据，性能通常更好
            self.classifier = ComplementNB(alpha=alpha, norm=True)
        else:
            self.classifier = MultinomialNB(alpha=alpha)

        # 特征缩放器（用于统计特征）
        self.scaler = StandardScaler(with_mean=False)  # 稀疏矩阵不能减均值

        # 训练状态
        self.is_trained = False

    def _extract_statistical_features(self, titles: List[str]) -> np.ndarray:
        """
        提取标题的统计特征

        参数:
            titles: 标题列表

        返回:
            统计特征矩阵 [n_samples, n_features]
        """
        features_list = []

        for title in titles:
            features = []

            # 预处理后的标题
            processed = DataLoader.preprocess_title(title)
            words = processed.split()

            # 1. 长度特征
            features.append(len(words))  # 词数
            features.append(len(title))  # 字符数
            features.append(len(title) / max(len(words), 1))  # 平均词长

            # 2. 标点符号和特殊字符
            features.append(title.count('.'))  # 点号数量
            features.append(title.count(','))  # 逗号数量
            features.append(title.count(':'))  # 冒号数量
            features.append(title.count(';'))  # 分号数量
            features.append(len(re.findall(r'[0-9]', title)))  # 数字数量

            # 3. 大写字母特征
            features.append(sum(1 for c in title if c.isupper()))  # 大写字母数
            features.append(sum(1 for c in title if c.isupper()) / max(len(title), 1))  # 大写比例

            # 4. 词汇多样性
            features.append(len(set(words)) / max(len(words), 1))  # 唯一词比例

            # 5. 特殊模式检测（错误标题常见模式）
            features.append(1 if 'abstract' in processed else 0)
            features.append(1 if 'introduction' in processed else 0)
            features.append(1 if 'reference' in processed else 0)
            features.append(1 if 'page' in processed else 0)
            features.append(1 if 'vol' in processed or 'volume' in processed else 0)
            features.append(1 if 'copyright' in processed else 0)
            features.append(1 if 'appendix' in processed else 0)
            features.append(1 if re.search(r'\d{4}', title) else 0)  # 包含年份
            features.append(1 if re.search(r'pp?\s*\d+', title.lower()) else 0)  # 包含页码

            # 6. 连续标点符号（错误标题常见）
            features.append(1 if '..' in title else 0)
            features.append(len(re.findall(r'\.{2,}', title)))  # 连续点号数量

            features_list.append(features)

        return np.array(features_list, dtype=np.float32)

    def train(self, titles: List[str], labels: List[int], save_model=True):
        """
        训练优化的朴素贝叶斯分类器

        参数:
            titles: 训练标题列表
            labels: 训练标签列表 (0=错误标题, 1=正确标题)
            save_model: 是否保存模型
        """
        print("\n" + "="*70)
        print("训练优化的朴素贝叶斯分类器")
        print("="*70)

        # 1. 提取词级TF-IDF特征
        print("\n步骤 1/4: 提取词级TF-IDF特征...")
        X_word = self.word_vectorizer.fit_transform(titles)
        print(f"  - 词级词汇表大小: {len(self.word_vectorizer.vocabulary_)}")
        print(f"  - 词级特征矩阵形状: {X_word.shape}")
        print(f"  - 词级n-gram范围: {self.word_ngram_range}")

        # 2. 提取字符级TF-IDF特征
        print("\n步骤 2/4: 提取字符级TF-IDF特征...")
        X_char = self.char_vectorizer.fit_transform(titles)
        print(f"  - 字符级词汇表大小: {len(self.char_vectorizer.vocabulary_)}")
        print(f"  - 字符级特征矩阵形状: {X_char.shape}")
        print(f"  - 字符级n-gram范围: {self.char_ngram_range}")

        # 3. 提取统计特征
        if self.add_statistical_features:
            print("\n步骤 3/4: 提取统计特征...")
            X_stat = self._extract_statistical_features(titles)
            print(f"  - 统计特征数量: {X_stat.shape[1]}")

            # 缩放统计特征
            X_stat_scaled = self.scaler.fit_transform(X_stat)

            # 转换为稀疏矩阵
            X_stat_sparse = csr_matrix(X_stat_scaled)

            # 合并所有特征
            X_train = hstack([X_word, X_char, X_stat_sparse])
        else:
            print("\n步骤 3/4: 跳过统计特征...")
            X_train = hstack([X_word, X_char])

        print(f"  - 合并后特征矩阵形状: {X_train.shape}")

        # 4. 训练分类器
        print("\n步骤 4/4: 训练分类器...")
        classifier_type = "ComplementNB" if self.use_complement_nb else "MultinomialNB"
        print(f"  - 分类器类型: {classifier_type}")
        print(f"  - 平滑参数alpha: {self.alpha}")

        self.classifier.fit(X_train, labels)

        self.is_trained = True
        print("✓ 训练完成!")

        # 显示统计信息
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
        print(f"  - 总特征数: {X_train.shape[1]}")
        print(f"  - 特征稀疏度: {(1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]))*100:.2f}%")

    def save_model(self):
        """保存模型到文件"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        model_data = {
            'word_vectorizer': self.word_vectorizer,
            'char_vectorizer': self.char_vectorizer,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'max_features_word': self.max_features_word,
            'max_features_char': self.max_features_char,
            'word_ngram_range': self.word_ngram_range,
            'char_ngram_range': self.char_ngram_range,
            'alpha': self.alpha,
            'use_complement_nb': self.use_complement_nb,
            'add_statistical_features': self.add_statistical_features,
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

        self.word_vectorizer = model_data['word_vectorizer']
        self.char_vectorizer = model_data['char_vectorizer']
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.max_features_word = model_data['max_features_word']
        self.max_features_char = model_data['max_features_char']
        self.word_ngram_range = model_data['word_ngram_range']
        self.char_ngram_range = model_data['char_ngram_range']
        self.alpha = model_data['alpha']
        self.use_complement_nb = model_data['use_complement_nb']
        self.add_statistical_features = model_data['add_statistical_features']
        self.is_trained = model_data['is_trained']

        print("✓ 模型加载成功!")
        return True

    def _prepare_features(self, titles: List[str]):
        """准备特征矩阵"""
        # 词级特征
        X_word = self.word_vectorizer.transform(titles)

        # 字符级特征
        X_char = self.char_vectorizer.transform(titles)

        # 统计特征
        if self.add_statistical_features:
            X_stat = self._extract_statistical_features(titles)
            X_stat_scaled = self.scaler.transform(X_stat)
            X_stat_sparse = csr_matrix(X_stat_scaled)
            X = hstack([X_word, X_char, X_stat_sparse])
        else:
            X = hstack([X_word, X_char])

        return X

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

        X_test = self._prepare_features(titles)
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

        X_test = self._prepare_features(titles)
        probabilities = self.classifier.predict_proba(X_test)

        return probabilities

    def get_feature_vectors(self, titles: List[str]) -> np.ndarray:
        """
        获取标题的特征向量(用于可视化)

        参数:
            titles: 标题列表

        返回:
            特征矩阵
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法或load_model()方法")

        return self._prepare_features(titles).toarray()


def main():
    """
    主函数:演示优化的朴素贝叶斯分类器
    """
    from data_loader import DataLoader, create_sample_data

    print("="*70)
    print(" 优化的朴素贝叶斯分类器演示")
    print("="*70)

    # 加载数据
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'data/positive.txt',
        'data/negative.txt',
        'data/testSet-1000.xlsx'
    )

    # 如果没有实际文件,使用示例数据
    if len(train_titles) == 0:
        train_titles, train_labels, test_titles, test_labels = create_sample_data()

    # 创建优化的分类器
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

    # 训练模型
    classifier.train(train_titles, train_labels, save_model=True)

    # 进行预测
    print("\n" + "="*70)
    print("在测试集上进行预测")
    print("="*70)

    predictions = classifier.predict(test_titles)

    # 计算准确率
    accuracy = np.mean(predictions == test_labels)
    print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 详细评估
    from evaluator import ModelEvaluator
    evaluator = ModelEvaluator()
    evaluator.evaluate_model(test_labels, predictions, "Optimized Naive Bayes", verbose=True)


if __name__ == "__main__":
    main()

"""
word2vec_svm_classifier.py
===========================
Word2Vec + SVM分类器实现
使用Word2Vec词嵌入和支持向量机
"""

import numpy as np
from typing import List
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from gensim.models import Word2Vec
from data_loader import DataLoader
from tqdm import tqdm
import time


class Word2VecSVMClassifier:
    """
    基于Word2Vec词嵌入和SVM的分类器
    将标题转换为词向量的平均值,然后使用SVM分类
    """
    
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=10, use_linear_svm=True, add_features=True):
        """
        初始化Word2Vec+SVM分类器

        参数:
            vector_size: 词向量维度
            window: 上下文窗口大小
            min_count: 最小词频(低于此频率的词会被忽略)
            epochs: Word2Vec训练轮数
            use_linear_svm: 是否使用线性SVM（更快，推荐用于大数据集）
            add_features: 是否添加统计特征（特征工程）
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.use_linear_svm = use_linear_svm
        self.add_features = add_features

        # Word2Vec模型
        self.w2v_model = None

        # SVM分类器
        if use_linear_svm:
            # LinearSVC 更快，适合大数据集
            base_svm = LinearSVC(
                C=1.0,
                max_iter=1000,
                verbose=1,  # 显示训练进度
                random_state=42
            )
            # 使用 CalibratedClassifierCV 来获得概率输出
            self.svm_classifier = CalibratedClassifierCV(base_svm, cv=3)
        else:
            # RBF 核 SVM，更准确但更慢
            self.svm_classifier = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                verbose=True  # 显示训练进度
            )

        # 训练状态
        self.is_trained = False
        
    def _tokenize(self, title: str) -> List[str]:
        """
        将标题分词
        
        参数:
            title: 输入标题
            
        返回:
            词列表
        """
        return DataLoader.preprocess_title(title).split()
    
    def _extract_statistical_features(self, title: str) -> np.ndarray:
        """
        提取标题的统计特征（特征工程）

        这些手工设计的特征能够捕捉错误标题的模式特征：
        - 错误标题可能过长（包含多个片段）或过短（只有部分词）
        - 错误标题可能包含大量数字（如页码、卷号）
        - 错误标题可能有异常的大小写模式（全大写的元数据）
        - 错误标题可能包含异常的特殊符号

        参数:
            title: 输入标题

        返回:
            统计特征向量 [8维] - 与词向量拼接后作为SVM的输入
        """
        features = []

        # ====================
        # 特征1: 标题长度（词数）
        # ====================
        # 正常标题一般5-15个词，错误标题可能只有1-2个词或超过20个词
        # 例如: "Abstract" (1词，错误) vs "Deep Learning for Image Recognition" (6词，正确)
        words = title.split()
        features.append(len(words))

        # ====================
        # 特征2: 平均词长
        # ====================
        # 正常单词平均5-7个字母，异常长的词可能是错误提取的合并词
        # 例如: "ImageNetClassificationWithDeepConvolutionalNeuralNetworks"
        if words:
            features.append(np.mean([len(w) for w in words]))
        else:
            features.append(0)

        # ====================
        # 特征3: 大写字母比例
        # ====================
        # 正常标题首字母大写，错误标题可能全大写(元数据)或全小写(片段)
        # 例如: "IEEE TRANSACTIONS ON..." (错误，全大写)
        if len(title) > 0:
            features.append(sum(1 for c in title if c.isupper()) / len(title))
        else:
            features.append(0)

        # ====================
        # 特征4: 数字比例
        # ====================
        # 正常标题很少包含数字，错误标题可能包含页码、卷号、年份等
        # 例如: "pp. 1-25, Vol. 3, 2024" (错误，大量数字)
        if len(title) > 0:
            features.append(sum(1 for c in title if c.isdigit()) / len(title))
        else:
            features.append(0)

        # ====================
        # 特征5: 特殊字符数量
        # ====================
        # 正常标题特殊字符少，错误标题可能有连续的点、破折号等
        # 例如: "........." 或 "---Appendix---"
        special_chars = sum(1 for c in title if not c.isalnum() and not c.isspace())
        features.append(special_chars)

        # ====================
        # 特征6-8: 布尔特征
        # ====================
        # 这些二值特征帮助识别明显的模式
        features.append(1 if any(c.isdigit() for c in title) else 0)  # 是否包含数字
        features.append(1 if title.isupper() else 0)                  # 是否全大写
        features.append(1 if title.islower() else 0)                  # 是否全小写

        return np.array(features)

    def _title_to_vector(self, title: str) -> np.ndarray:
        """
        将标题转换为向量(词向量平均 + 可选的统计特征)

        参数:
            title: 输入标题

        返回:
            标题的向量表示
        """
        words = self._tokenize(title)
        word_vectors = []

        # 收集所有存在于词汇表中的词的向量
        for word in words:
            if word in self.w2v_model.wv:
                word_vectors.append(self.w2v_model.wv[word])

        # 如果找到了词向量,返回平均值;否则返回零向量
        if word_vectors:
            word_vec = np.mean(word_vectors, axis=0)
        else:
            word_vec = np.zeros(self.vector_size)

        # 如果启用特征工程，添加统计特征
        if self.add_features:
            stat_features = self._extract_statistical_features(title)
            combined_vec = np.concatenate([word_vec, stat_features])
            return combined_vec
        else:
            return word_vec
    
    def train(self, titles: List[str], labels: List[int]):
        """
        训练Word2Vec模型和SVM分类器
        
        参数:
            titles: 训练标题列表
            labels: 训练标签列表 (0=错误标题, 1=正确标题)
        """
        print("\n" + "="*60)
        print("训练 Word2Vec + SVM 分类器")
        print("="*60)
        
        # ===== 步骤1: 训练Word2Vec模型 =====
        print("\n步骤 1/3: 训练Word2Vec模型...")

        # 对所有标题进行分词
        print(f"  - 正在分词 {len(titles)} 个标题...")
        tokenized_titles = [self._tokenize(title) for title in tqdm(titles, desc="  分词进度", ncols=80)]

        # 训练Word2Vec
        self.w2v_model = Word2Vec(
            sentences=tokenized_titles,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=4,  # 使用4个线程
            sg=0,  # 使用CBOW模型(sg=1为Skip-gram)
            seed=42
        )

        vocab_size = len(self.w2v_model.wv)
        print(f"  - Word2Vec词汇表大小: {vocab_size}")
        print(f"  - 词向量维度: {self.vector_size}")
        print(f"  - 训练轮数: {self.epochs}")
        
        # ===== 步骤2: 将标题转换为向量 =====
        print("\n步骤 2/3: 将标题转换为向量...")
        if self.add_features:
            print("  - 启用特征工程：词向量 + 8个统计特征")

        X_train = np.array([self._title_to_vector(title) for title in tqdm(titles, desc="  向量化进度", ncols=80)])

        print(f"  - 特征矩阵形状: {X_train.shape}")
        if self.add_features:
            print(f"  - 词向量维度: {self.vector_size}")
            print(f"  - 统计特征维度: 8")
            print(f"  - 总特征维度: {X_train.shape[1]}")

        # 统计零向量的数量（只检查词向量部分）
        if self.add_features:
            word_vec_part = X_train[:, :self.vector_size]
            zero_vectors = np.sum(np.all(word_vec_part == 0, axis=1))
        else:
            zero_vectors = np.sum(np.all(X_train == 0, axis=1))

        if zero_vectors > 0:
            print(f"  ⚠️  警告: {zero_vectors} 个标题的词向量为零(没有找到词汇)")
        
        # ===== 步骤3: 训练SVM分类器 =====
        print("\n步骤 3/3: 训练SVM分类器...")
        print(f"  - 训练样本数: {len(labels)}")
        print(f"  - 特征维度: {X_train.shape[1]}")

        if self.use_linear_svm:
            print(f"  - 使用 LinearSVM（快速模式，适合大数据集）")
            print(f"  - 正在训练（预计 2-5 分钟）...")
        else:
            print(f"  - 使用 RBF 核 SVM（高精度模式）")
            print(f"  - 正在训练（预计 15-30 分钟，数据量大时可能更久）...")

        start_time = time.time()
        self.svm_classifier.fit(X_train, labels)
        elapsed_time = time.time() - start_time

        print(f"  - SVM 训练完成，耗时: {elapsed_time/60:.1f} 分钟")

        # 获取支持向量数量（仅对 SVC 有效）
        if hasattr(self.svm_classifier, 'n_support_'):
            n_support = self.svm_classifier.n_support_
            print(f"  - 支持向量数量: {n_support}")

        if hasattr(self.svm_classifier, 'classes_'):
            print(f"  - 类别: {self.svm_classifier.classes_}")
        
        self.is_trained = True
        print("\n✓ 训练完成!")
        
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

        # 将标题转换为向量
        if len(titles) > 100:
            # 只有数据量大时才显示进度条
            X_test = np.array([self._title_to_vector(title) for title in tqdm(titles, desc="预测向量化", ncols=80)])
        else:
            X_test = np.array([self._title_to_vector(title) for title in titles])

        # 使用SVM进行预测
        predictions = self.svm_classifier.predict(X_test)

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
        
        X_test = np.array([self._title_to_vector(title) for title in titles])
        probabilities = self.svm_classifier.predict_proba(X_test)
        
        return probabilities
    
    def get_feature_vectors(self, titles: List[str]) -> np.ndarray:
        """
        获取标题的Word2Vec特征向量(用于可视化)
        
        参数:
            titles: 标题列表
            
        返回:
            特征矩阵
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法")
        
        return np.array([self._title_to_vector(title) for title in titles])
    
    def find_similar_words(self, word: str, topn=10):
        """
        找到与给定词最相似的词
        
        参数:
            word: 查询词
            topn: 返回的相似词数量
            
        返回:
            相似词列表及其相似度
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法")
        
        word = DataLoader.preprocess_title(word)
        
        if word in self.w2v_model.wv:
            similar_words = self.w2v_model.wv.most_similar(word, topn=topn)
            return similar_words
        else:
            return None


def main():
    """
    主函数:演示Word2Vec+SVM分类器的使用
    """
    from data_loader import DataLoader, create_sample_data
    
    print("="*70)
    print(" Word2Vec + SVM 分类器演示")
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
    classifier = Word2VecSVMClassifier(
        vector_size=100,
        window=5,
        min_count=2,
        epochs=10
    )
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
    
    # 演示词相似度功能
    print("\n" + "="*60)
    print("词相似度示例")
    print("="*60)
    
    test_words = ['learning', 'network', 'algorithm', 'model']
    
    for word in test_words:
        similar = classifier.find_similar_words(word, topn=5)
        if similar:
            print(f"\n与 '{word}' 最相似的词:")
            for similar_word, similarity in similar:
                print(f"  - {similar_word}: {similarity:.4f}")
        else:
            print(f"\n'{word}' 不在词汇表中")


if __name__ == "__main__":
    main()

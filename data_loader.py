"""
data_loader.py
==============
数据加载和预处理模块
负责从文件中读取数据并进行预处理
"""

import re
from typing import List, Tuple


class DataLoader:
    """处理标题数据的加载和预处理"""

    @staticmethod
    def load_titles(filepath: str, encoding='utf-8') -> List[str]:
        """
        从文本文件加载标题

        参数:
            filepath: 文件路径,每行一个标题
            encoding: 文件编码

        返回:
            标题字符串列表
        """
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                titles = [line.strip() for line in f if line.strip()]
            print(f"成功加载 {len(titles)} 条标题从 {filepath}")
            return titles
        except FileNotFoundError:
            print(f"错误: 找不到文件 {filepath}")
            return []
        except Exception as e:
            print(f"加载 {filepath} 时出错: {str(e)}")
            return []

    @staticmethod
    def preprocess_title(title: str) -> str:
        """
        预处理单个标题:转小写、移除特殊字符

        参数:
            title: 原始标题字符串

        返回:
            预处理后的标题
        """
        # 转换为小写
        title = title.lower()

        # 移除特殊字符但保留空格
        title = re.sub(r'[^a-z0-9\s]', ' ', title)

        # 移除多余空格
        title = ' '.join(title.split())

        return title

    @staticmethod
    def prepare_dataset(
            positive_file: str,
            negative_file: str,
            test_file: str
    ) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        加载并准备训练和测试数据集

        参数:
            positive_file: 正样本文件路径
            negative_file: 负样本文件路径
            test_file: 测试数据文件路径 (Excel格式)

        返回:
            (训练标题, 训练标签, 测试标题, 测试标签)
        """
        print("\n" + "=" * 60)
        print("开始加载数据集...")
        print("=" * 60)

        loader = DataLoader()

        # 加载训练数据
        positive_titles = loader.load_titles(positive_file)
        negative_titles = loader.load_titles(negative_file)

        # 创建训练集
        train_titles = positive_titles + negative_titles
        train_labels = [1] * len(positive_titles) + [0] * len(negative_titles)

        # 加载测试数据 (Excel格式)
        test_titles = []
        test_labels = []

        try:
            import pandas as pd
            df = pd.read_excel(test_file)

            # 列名: 'title given by manchine', 'Y/N'
            # Y/N: Y表示正确标题(1), N表示错误标题(0)
            for _, row in df.iterrows():
                title = row['title given by manchine']
                label = 1 if row['Y/N'] == 'Y' else 0

                if pd.notna(title) and str(title).strip():
                    test_titles.append(str(title).strip())
                    test_labels.append(label)

            print(f"成功加载 {len(test_titles)} 条测试数据")
        except Exception as e:
            print(f"加载测试文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()

        # 打印数据集统计信息
        print(f"\n数据集统计:")
        print(f"  训练集总数: {len(train_titles)} 条标题")
        print(f"    - 正样本 (正确标题): {len(positive_titles)}")
        print(f"    - 负样本 (错误标题): {len(negative_titles)}")
        print(f"  测试集总数: {len(test_titles)} 条标题")
        if test_labels:
            print(f"    - 正样本 (Y): {sum(test_labels)}")
            print(f"    - 负样本 (N): {len(test_labels) - sum(test_labels)}")

        return train_titles, train_labels, test_titles, test_labels


def create_sample_data() -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    创建示例数据用于演示(当实际文件不可用时)

    返回:
        (训练标题, 训练标签, 测试标题, 测试标签)
    """
    print("\n⚠️  警告: 未找到数据文件,使用示例数据...")

    # 正确的学术标题示例
    positive_samples = [
                           "Deep Learning for Computer Vision Applications",
                           "Natural Language Processing with Transformer Models",
                           "Introduction to Machine Learning Algorithms",
                           "Reinforcement Learning in Autonomous Robotics",
                           "Graph Neural Networks for Social Network Analysis",
                           "Convolutional Neural Networks for Image Recognition",
                           "Recurrent Neural Networks for Time Series Prediction",
                           "Transfer Learning in Deep Learning Systems",
                           "Attention Mechanisms in Neural Machine Translation",
                           "Generative Adversarial Networks for Image Synthesis",
                           "BERT: Pre-training of Deep Bidirectional Transformers",
                           "ResNet: Deep Residual Learning for Image Recognition",
                           "AlexNet: ImageNet Classification with Deep CNNs",
                           "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
                           "Batch Normalization: Accelerating Deep Network Training"
                       ] * 15  # 复制以获得更多样本

    # 错误提取的标题示例
    negative_samples = [
                           "Call for Papers......41 Fragments......42 Special Issue",
                           "Special Issue on......Page 1-25......Vol 3 Number 2",
                           "Conference Proceedings......Abstract......Introduction......References",
                           "Table of Contents......Chapter 1......Appendix A......Index",
                           "Copyright Notice......All Rights Reserved......2024 IEEE",
                           "Page Header......Footer......Section 2.1......Figure 1",
                           "Bibliography......Citation [1]......Citation [2]......End",
                           "Authors: John Doe......Jane Smith......Affiliation Info",
                           "Abstract: This paper......Keywords: machine learning......ACM",
                           "Acknowledgments......Funding Information......Ethics Statement",
                           "Appendix B......Supplementary Materials......Online Resources",
                           "Review Comments......Editor's Note......Submission Guidelines",
                           "Volume 12......Issue 3......ISSN 1234-5678......DOI Link",
                           "Conference Program......Session Chair......Coffee Break 10:30",
                           "Poster Presentation......Demo Session......Workshop Schedule"
                       ] * 15

    # 创建测试集 (20%的数据)
    test_size_pos = len(positive_samples) // 5
    test_size_neg = len(negative_samples) // 5

    test_positive = positive_samples[:test_size_pos]
    test_negative = negative_samples[:test_size_neg]

    train_positive = positive_samples[test_size_pos:]
    train_negative = negative_samples[test_size_neg:]

    # 组合训练集
    train_titles = train_positive + train_negative
    train_labels = [1] * len(train_positive) + [0] * len(train_negative)

    # 组合测试集
    test_titles = test_positive + test_negative
    test_labels = [1] * len(test_positive) + [0] * len(test_negative)

    print(f"创建了 {len(train_titles)} 条训练数据 和 {len(test_titles)} 条测试数据")

    return train_titles, train_labels, test_titles, test_labels


if __name__ == "__main__":
    # 测试数据加载模块
    print("测试 DataLoader 模块\n")

    # 尝试加载实际文件
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'positive_titles.txt',
        'negative_titles.txt',
        'test_titles.txt'
    )

    # 如果没有实际文件,使用示例数据
    if len(train_titles) == 0:
        train_titles, train_labels, test_titles, test_labels = create_sample_data()

    # 显示一些示例
    print("\n正样本示例:")
    for i, (title, label) in enumerate(zip(train_titles[:3], train_labels[:3])):
        if label == 1:
            print(f"  {i + 1}. {title}")

    print("\n负样本示例:")
    for i, (title, label) in enumerate(zip(train_titles, train_labels)):
        if label == 0:
            print(f"  {i + 1}. {title}")
            if i >= 2:
                break
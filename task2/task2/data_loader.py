"""
data_loader.py
==============
数据加载和预处理模块
负责从文件中读取数据并进行预处理
"""

import re
import zipfile
import xml.etree.ElementTree as ET
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
            # 尝试使用指定的编码加载
            with open(filepath, 'r', encoding=encoding) as f:
                titles = [line.strip() for line in f if line.strip()]
            print(f"成功加载 {len(titles)} 条标题从 {filepath}")
            return titles
        except UnicodeDecodeError:
            # 如果失败，尝试使用 GBK 编码
            print(f"⚠️  {filepath} 不是 {encoding} 编码，尝试使用 GBK...")
            try:
                with open(filepath, 'r', encoding='gbk') as f:
                    titles = [line.strip() for line in f if line.strip()]
                print(f"成功加载 {len(titles)} 条标题从 {filepath} (GBK)")
                return titles
            except Exception as e:
                print(f"加载 {filepath} 时出错 (GBK): {str(e)}")
                return []
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
    def _read_excel_fallback(filepath: str):
        """
        当 openpyxl 失败时，手动解析 XLSX 文件
        """
        import pandas as pd
        
        print(f"⚠️  启动手动 XML 解析模式读取 {filepath}...")
        
        try:
            with zipfile.ZipFile(filepath, 'r') as z:
                # 1. 加载共享字符串 (Shared Strings)
                shared_strings = []
                if 'xl/sharedStrings.xml' in z.namelist():
                    with z.open('xl/sharedStrings.xml') as f:
                        tree = ET.parse(f)
                        root = tree.getroot()
                        ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                        # 查找所有 <si> 标签
                        for si in root.findall('main:si', ns):
                            # 尝试直接获取 <t>
                            t = si.find('main:t', ns)
                            if t is not None and t.text:
                                shared_strings.append(t.text)
                            else:
                                # 尝试获取 <r><t> (富文本)
                                text_parts = []
                                for t_part in si.findall('.//main:t', ns):
                                    if t_part.text:
                                        text_parts.append(t_part.text)
                                shared_strings.append("".join(text_parts))

                # 2. 加载第一个工作表
                sheet_path = 'xl/worksheets/sheet1.xml'
                if sheet_path not in z.namelist():
                    # 尝试查找 workbook.xml 来确定 sheet 路径 (简化处理，假设是 sheet1)
                    # 如果没有 sheet1，尝试列出所有 worksheets
                    sheets = [n for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')]
                    if sheets:
                        sheet_path = sheets[0]
                    else:
                        raise ValueError("无法找到工作表 XML 文件")

                data = []
                with z.open(sheet_path) as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                    
                    # 遍历所有行
                    for row in root.findall('.//main:row', ns):
                        row_values = []
                        # 遍历行中的所有单元格
                        for cell in row.findall('main:c', ns):
                            t = cell.get('t') # 类型
                            v_tag = cell.find('main:v', ns) # 值
                            value = v_tag.text if v_tag is not None else None
                            
                            if t == 's' and value is not None:
                                # 共享字符串索引
                                try:
                                    idx = int(value)
                                    if 0 <= idx < len(shared_strings):
                                        value = shared_strings[idx]
                                except:
                                    pass
                            elif t == 'str':
                                # 内联字符串
                                pass 
                            elif value is not None:
                                # 数值
                                try:
                                    f_val = float(value)
                                    if f_val.is_integer():
                                        value = int(f_val)
                                    else:
                                        value = f_val
                                except:
                                    pass
                            
                            row_values.append(value)
                        
                        if row_values:
                            data.append(row_values)

            if not data:
                raise ValueError("解析后数据为空")

            # 假设第一行是表头
            # 注意：XML解析可能丢失空单元格，导致列对齐问题。
            # 对于这个特定的数据集，我们假设它是规整的。
            # 如果第一行比其他行短，可能需要填充
            max_cols = max(len(r) for r in data)
            
            # 规范化数据长度
            normalized_data = []
            for r in data:
                if len(r) < max_cols:
                    r.extend([None] * (max_cols - len(r)))
                normalized_data.append(r)
                
            headers = normalized_data[0]
            rows = normalized_data[1:]
            return pd.DataFrame(rows, columns=headers)

        except Exception as e:
            print(f"❌ 手动 XML 解析失败: {e}")
            raise e

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
            # 尝试使用 openpyxl 引擎，并忽略筛选器错误
            try:
                df = pd.read_excel(test_file, engine='openpyxl')
            except ValueError as ve:
                if "Value must be either numerical or a string containing a wildcard" in str(ve):
                    print(f"⚠️  检测到 Excel 筛选器错误，尝试使用手动 XML 解析...")
                    try:
                        df = DataLoader._read_excel_fallback(test_file)
                    except Exception as e_fallback:
                        print(f"❌ 所有读取尝试均失败。")
                        raise ve
                else:
                    raise ve

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
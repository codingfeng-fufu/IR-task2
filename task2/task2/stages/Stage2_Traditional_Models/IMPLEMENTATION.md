# Stage2_Traditional_Models 实现文档

## 📋 阶段概述

**阶段名称**: Stage2 - 传统模型实现
**实现时间**: 2024年11月15日
**主要目标**: 实现三种基础分类方法(朴素贝叶斯、Word2Vec+SVM、BERT)
**代码行数**: ~1,400行(3个文件)

## 🎯 实现目标

- ✅ 朴素贝叶斯V1: 73.46%准确率
- ✅ Word2Vec+SVM: 82.99%准确率
- ✅ BERT基础版: 87.91%准确率
- ✅ 建立三种不同技术路线的baseline

## 📁 文件结构

```
Stage2_Traditional_Models/
├── naive_bayes_classifier.py      # 朴素贝叶斯 (~273行)
├── word2vec_svm_classifier.py     # Word2Vec+SVM (~450行)
├── bert_classifier.py             # BERT基础版 (~348行)
├── config.py                      # 配置文件(输出路径)
├── output/                        # 本阶段输出目录 ⭐
│   ├── naive_bayes_evaluation.txt
│   ├── word2vec_svm_evaluation.txt
│   ├── bert_evaluation.txt
│   └── [可视化文件]
├── models/                        # 本阶段模型目录 ⭐
│   ├── naive_bayes_model.pkl
│   ├── word2vec_svm_model_w2v.model
│   ├── word2vec_svm_model_svm.pkl
│   └── best_bert_model.pt
└── README.md
```

## 🔧 核心实现

### 1. 朴素贝叶斯分类器

**技术栈**: TF-IDF + MultinomialNB

**关键代码**:
```python
from config import get_model_path
from naive_bayes_classifier import NaiveBayesClassifier

# 初始化(使用config获取路径)
classifier = NaiveBayesClassifier(
    max_features=5000,
    ngram_range=(1, 2),
    model_path=get_model_path('naive_bayes_model.pkl')  # ⭐正确路径
)

# 训练
classifier.train(train_titles, train_labels)

# 预测
predictions = classifier.predict(test_titles)
```

**性能**: 73.46%准确率,F1 78.82%

### 2. Word2Vec + SVM分类器

**技术栈**: Gensim Word2Vec + LinearSVC

**关键代码**:
```python
from config import get_model_path
from word2vec_svm_classifier import Word2VecSVMClassifier

# 初始化
classifier = Word2VecSVMClassifier(
    vector_size=100,
    window=5,
    model_path=get_model_path('word2vec_svm_model')  # 不含扩展名
)

# 训练
classifier.train(train_titles, train_labels)
```

**性能**: 82.99%准确率,F1 85.74%

### 3. BERT分类器

**技术栈**: bert-base-uncased + PyTorch

**关键代码**:
```python
from config import get_model_path
from bert_classifier import BERTClassifier

# 初始化
classifier = BERTClassifier(
    model_name='bert-base-uncased',
    max_length=64,
    model_path=get_model_path('best_bert_model.pt')  # ⭐正确路径
)

# 训练
classifier.train(
    train_titles,
    train_labels,
    epochs=3,
    batch_size=16
)
```

**性能**: 87.91%准确率,F1 89.59%

## 📂 输出位置说明

### 输出目录结构

```
Stage2_Traditional_Models/
├── output/                    # ⭐所有输出保存在此
│   ├── [评估结果.txt]
│   └── [可视化图表.png]
│
├── models/                    # ⭐所有模型保存在此
│   ├── naive_bayes_model.pkl       # ~11 MB
│   ├── word2vec_svm_model_w2v.model  # ~25 MB
│   ├── word2vec_svm_model_svm.pkl    # ~114 MB
│   └── best_bert_model.pt             # ~438 MB
```

### 如何使用config.py

**在代码中导入config**:
```python
from config import get_output_path, get_model_path, get_data_path

# 获取模型保存路径
model_path = get_model_path('my_model.pkl')
# → .../Stage2_Traditional_Models/models/my_model.pkl

# 获取输出文件路径
output_file = get_output_path('evaluation.txt')
# → .../Stage2_Traditional_Models/output/evaluation.txt
```

### 检查输出

```bash
# 查看模型文件
ls -lh /home/u2023312337/task2/task2/stages/Stage2_Traditional_Models/models/

# 查看输出文件
ls -lh /home/u2023312337/task2/task2/stages/Stage2_Traditional_Models/output/
```

## 🚀 运行示例

### 训练单个模型

```bash
cd /home/u2023312337/task2/task2/stages/Stage2_Traditional_Models

# 训练朴素贝叶斯
python naive_bayes_classifier.py

# 训练Word2Vec+SVM
python word2vec_svm_classifier.py

# 训练BERT
python bert_classifier.py
```

### 从Stage1导入工具

```python
# 需要访问Stage1的工具模块
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Stage1_Foundation'))

from data_loader import DataLoader
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
```

## 📊 性能对比

| 模型 | 准确率 | 精确率 | 召回率 | F1 | 训练时间 | 模型大小 |
|------|--------|--------|--------|-----|----------|----------|
| 朴素贝叶斯V1 | 73.46% | 73.59% | 84.86% | 78.82% | ~2分钟 | 11 MB |
| Word2Vec+SVM | 82.99% | 85.84% | 85.58% | 85.74% | ~10分钟 | 139 MB |
| BERT基础版 | 87.91% | 90.29% | 88.35% | 89.59% | ~1小时 | 438 MB |

## 🔗 与其他阶段的关系

- **依赖**: Stage1_Foundation(数据加载、评估、可视化)
- **被依赖**: Stage3(优化朴素贝叶斯)、Stage4(优化BERT)
- **对比**: 为后续优化提供baseline

## ⚠️ 注意事项

1. **路径配置**: 所有模型保存路径都应使用`config.get_model_path()`
2. **依赖Stage1**: 需要Stage1的data_loader, evaluator, visualizer
3. **GPU支持**: BERT训练需要GPU,否则极慢
4. **内存需求**: BERT训练至少需要8GB内存

## 📝 修改记录

- **2024-11-15**: 实现三个基础分类器
- **2024-12-05**: 添加config.py,实现独立输出目录

## 📚 相关文档

- **README.md** - 阶段概述
- **IMPLEMENTATION.md** (本文档) - 详细实现说明
- **../Stage1_Foundation/IMPLEMENTATION.md** - 基础设施说明

---

**实现完成度**: ✅ 100%
**输出位置**: ✅ 已配置到本阶段目录

# Stage2: 传统模型实现

**时间**：2024年11月15日  
**目标**：实现三种基础分类方法（朴素贝叶斯、Word2Vec+SVM、BERT）

## 📁 文件列表

| 文件 | 行数 | 功能 | 性能 |
|------|------|------|------|
| `naive_bayes_classifier.py` | 273 | 朴素贝叶斯分类器V1 | 73.46% |
| `word2vec_svm_classifier.py` | 450 | Word2Vec+SVM分类器 | 82.99% |
| `bert_classifier.py` | 348 | BERT基础分类器 | 87.91% |

## 🎯 阶段成果

### 1. 朴素贝叶斯V1 (naive_bayes_classifier.py)
**性能**：准确率 73.46%，F1 78.82%

**技术特点**：
- TF-IDF特征提取（5,000维）
- N-gram范围：(1,2) - unigram + bigram
- 分类器：MultinomialNB（alpha=1.0）
- 简单直接的实现

**使用示例**：
```python
from naive_bayes_classifier import NaiveBayesClassifier

classifier = NaiveBayesClassifier(max_features=5000, ngram_range=(1, 2))
classifier.train(train_titles, train_labels)
predictions = classifier.predict(test_titles)
```

### 2. Word2Vec + SVM (word2vec_svm_classifier.py)
**性能**：准确率 82.99%，F1 85.74%

**技术特点**：
- Gensim Word2Vec训练词向量（100维）
- 句子表示：词向量平均
- SVM分类器：LinearSVC
- 支持增量训练

**使用示例**：
```python
from word2vec_svm_classifier import Word2VecSVMClassifier

classifier = Word2VecSVMClassifier(vector_size=100, window=5)
classifier.train(train_titles, train_labels)
predictions = classifier.predict(test_titles)
```

### 3. BERT基础版 (bert_classifier.py)
**性能**：准确率 87.91%，F1 89.59%

**技术特点**：
- 使用 `bert-base-uncased` 预训练模型
- 序列最大长度：64 tokens
- 训练：3 epochs，batch_size 16
- 使用AdamW优化器

**使用示例**：
```python
from bert_classifier import BERTClassifier

classifier = BERTClassifier(model_name='bert-base-uncased', max_length=64)
classifier.train(train_titles, train_labels, epochs=3, batch_size=16)
predictions = classifier.predict(test_titles)
```

## 📊 性能对比

| 模型 | 准确率 | 精确率 | 召回率 | F1 | 训练时间 |
|------|--------|--------|--------|-----|----------|
| 朴素贝叶斯V1 | 73.46% | 73.59% | 84.86% | 78.82% | ~2分钟 |
| Word2Vec+SVM | 82.99% | 85.84% | 85.58% | 85.74% | ~10分钟 |
| BERT基础版 | 87.91% | 90.29% | 88.35% | 89.59% | ~1小时 |

## 🔍 技术分析

### 为什么BERT表现最好？
1. **上下文理解**：BERT捕捉长距离依赖
2. **预训练知识**：在大规模语料上预训练
3. **深层表示**：12层Transformer编码器

### 为什么朴素贝叶斯较弱？
1. **特征独立假设**：忽略词之间的依赖关系
2. **简单特征**：仅使用TF-IDF，无语义信息
3. **线性模型**：表达能力有限

### Word2Vec+SVM的中庸表现
- ✅ 优点：引入词向量语义信息
- ❌ 缺点：简单平均丢失词序信息

## 🔗 后续优化

基于此阶段的结果，后续进行了：
- **Stage3**：朴素贝叶斯深度优化（73.46% → 79.20%）
- **Stage4**：BERT高级优化（87.91% → 89.04%）

## 💻 代码统计

- **总行数**：~1,400行
- **文件数**：3个
- **平均每个模型**：~467行

---

**说明**：这三个文件是整个项目的基石，所有后续优化都基于这些基础实现。

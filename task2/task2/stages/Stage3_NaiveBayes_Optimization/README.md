# Stage3: 朴素贝叶斯优化

**时间**：2024年11月25日  
**目标**：深度优化朴素贝叶斯分类器，探索特征工程和算法改进

## 📁 文件列表

| 文件 | 行数 | 功能 | 性能提升 |
|------|------|------|----------|
| `naive_bayes_classifier_optimized.py` | 399 | 优化版朴素贝叶斯 | +5.74% |
| `test_optimized_nb.py` | 258 | V1 vs V2对比测试 | - |

## 🎯 优化成果

### 性能提升
- **准确率**：73.46% → **79.20%** (+5.74%)
- **F1分数**：78.82% → **83.69%** (+4.87%)
- **召回率**：84.86% → **91.73%** (+6.87%)

## 🔬 优化技术

### 1. 多层级TF-IDF特征
```python
# 原始（V1）：单层词级TF-IDF
TF-IDF(words, max_features=5000, ngram=(1,2))
→ 5,000维特征

# 优化（V2）：双层TF-IDF
TF-IDF(words, max_features=10000, ngram=(1,3)) +  # 词级
TF-IDF(chars, max_features=5000, ngram=(3,5))      # 字符级
→ 15,000维特征
```

### 2. 统计特征工程（22个特征）

**长度特征（3个）**：
- 词数量、字符数量、平均词长度

**标点特征（5个）**：
- 点号、逗号、冒号、分号、数字出现次数

**大写特征（2个）**：
- 大写字母数量、大写字母比例

**词汇多样性（1个）**：
- 唯一词比例（unique words / total words）

**特殊模式检测（9个）**：
- "abstract", "reference", "page", "vol", "copyright"关键词
- 年份模式（1990-2030）
- 页码模式（pp., p.)
- 连续点号（......）

**格式异常检测（2个）**：
- 点号总数、连续点号出现次数

### 3. 算法改进

**原始（V1）**：
```python
MultinomialNB(alpha=1.0)
```

**优化（V2）**：
```python
ComplementNB(alpha=0.5)
# ComplementNB更适合文本分类，特别是不平衡数据
```

### 4. 特征组合

**最终特征空间**：15,022维
- 词级TF-IDF：10,000维
- 字符级TF-IDF：5,000维
- 统计特征：22维

## 📊 详细对比

| 指标 | V1（原始） | V2（优化） | 提升 |
|------|-----------|-----------|------|
| 准确率 | 73.46% | **79.20%** | +5.74% |
| 精确率 | 73.59% | **76.96%** | +3.37% |
| 召回率 | 84.86% | **91.73%** | +6.87% |
| F1分数 | 78.82% | **83.69%** | +4.87% |
| F1宏平均 | 73.03% | **78.99%** | +5.96% |
| 训练时间 | ~2分钟 | ~3分钟 | +50% |
| 模型大小 | 11 MB | 44 MB | +300% |

## 🔧 使用示例

### 训练优化版模型
```python
from naive_bayes_classifier_optimized import NaiveBayesClassifierOptimized

classifier = NaiveBayesClassifierOptimized(
    max_features_word=10000,
    max_features_char=5000,
    word_ngram_range=(1, 3),
    char_ngram_range=(3, 5),
    alpha=0.5,
    use_complement_nb=True,
    add_statistical_features=True
)

classifier.train(train_titles, train_labels)
predictions = classifier.predict(test_titles)
```

### 对比V1与V2
```bash
python test_optimized_nb.py
```

输出示例：
```
模型对比：
  朴素贝叶斯V1: 准确率 73.46%, F1 78.82%
  朴素贝叶斯V2: 准确率 79.20%, F1 83.69%
  
性能提升：+5.74%准确率，+4.87% F1
```

## 💡 优化经验

### 有效的优化
✅ **字符级TF-IDF**：捕捉拼写错误和格式问题  
✅ **统计特征**：简单但有效的模式识别  
✅ **ComplementNB**：比MultinomialNB更稳定  
✅ **三元组（trigram）**：捕捉更长的短语

### 无效的尝试
❌ 增加max_features到50000（过拟合）  
❌ 使用GaussianNB（性能下降）  
❌ 过于复杂的正则表达式特征（噪声）

## 🔍 错误分析

### 改进的案例
```
标题: "pp. 123-145 Neural Networks"
V1预测: 正确（1） → 错误！
V2预测: 错误（0） → 正确！
原因: V2的页码检测特征起作用
```

### 仍然困难的案例
```
标题: "Deep Learning: A Survey"
真实标签: 正确（1）
两版本都预测: 错误（0）
原因: 冒号可能被误认为格式错误
```

## 📚 参考文档

详细优化过程见：`OPTIMIZATION_SUMMARY.md`

## 🔗 后续工作

此优化方法证明了特征工程的有效性，为后续工作提供了思路：
- **Stage4**：BERT也可以添加统计特征
- 混合模型：结合传统特征和深度学习

---

**总结**：通过系统的特征工程，朴素贝叶斯性能提升5.74%，证明了传统方法仍有优化空间。

# Stage3_NaiveBayes_Optimization 实现文档

## 📋 阶段概述

**阶段名称**: Stage3 - 朴素贝叶斯优化
**实现时间**: 2024年11月25日
**主要目标**: 深度优化朴素贝叶斯分类器,从73.46%提升到79.20%
**代码行数**: ~700行(2个文件)
**性能提升**: +5.74%准确率

## 🎯 优化成果

- ✅ 准确率: 73.46% → **79.20%** (+5.74%)
- ✅ F1分数: 78.82% → **83.69%** (+4.87%)
- ✅ 召回率: 84.86% → **91.73%** (+6.87%)

## 📁 文件结构

```
Stage3_NaiveBayes_Optimization/
├── naive_bayes_classifier_optimized.py   # 优化版 (~399行)
├── test_optimized_nb.py                  # V1 vs V2对比 (~258行)
├── config.py                             # 配置文件
├── output/                               # 本阶段输出 ⭐
│   ├── optimized_evaluation.txt
│   ├── comparison_v1_v2.txt
│   └── [对比图表]
├── models/                               # 本阶段模型 ⭐
│   ├── naive_bayes_optimized_model.pkl  # ~44 MB
│   └── naive_bayes_original_model.pkl   # ~11 MB (用于对比)
└── README.md
```

## 🔬 优化技术

### 1. 多层级TF-IDF特征(15,000维)

```python
# V1: 单层词级TF-IDF (5,000维)
TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# V2: 双层TF-IDF (15,000维)
# 词级: 10,000维, (1,3)-grams
# 字符级: 5,000维, (3,5)-grams
```

### 2. 统计特征工程(22个特征)

- **长度特征**(3个): 词数、字符数、平均词长
- **标点特征**(5个): 点号、逗号、冒号、分号、数字
- **大写特征**(2个): 大写字母数、比例
- **词汇多样性**(1个): 唯一词比例
- **特殊模式**(9个): "abstract", "reference", 年份, 页码等
- **格式异常**(2个): 连续点号检测

### 3. 算法改进

```python
# V1: MultinomialNB(alpha=1.0)
# V2: ComplementNB(alpha=0.5)  # 更适合文本分类
```

## 🔧 核心实现

### 使用优化版分类器

```python
from config import get_model_path
from naive_bayes_classifier_optimized import NaiveBayesClassifierOptimized

classifier = NaiveBayesClassifierOptimized(
    max_features_word=10000,
    max_features_char=5000,
    word_ngram_range=(1, 3),
    char_ngram_range=(3, 5),
    alpha=0.5,
    use_complement_nb=True,
    add_statistical_features=True,
    model_path=get_model_path('naive_bayes_optimized_model.pkl')  # ⭐
)

classifier.train(train_titles, train_labels)
predictions = classifier.predict(test_titles)
```

### V1 vs V2 对比测试

```bash
cd /home/u2023312337/task2/task2/stages/Stage3_NaiveBayes_Optimization
python test_optimized_nb.py
```

**输出示例**:
```
模型对比:
  朴素贝叶斯V1: 准确率 73.46%, F1 78.82%
  朴素贝叶斯V2: 准确率 79.20%, F1 83.69%
  
性能提升: +5.74%准确率, +4.87% F1
```

## 📂 输出位置

```
Stage3_NaiveBayes_Optimization/
├── output/              # ⭐所有输出在此
│   ├── v1_evaluation.txt
│   ├── v2_evaluation.txt
│   ├── comparison.png
│   └── error_analysis.txt
│
├── models/              # ⭐所有模型在此
│   ├── naive_bayes_optimized_model.pkl  # 44 MB
│   └── naive_bayes_original_model.pkl   # 11 MB
```

### 使用config.py

```python
from config import get_output_path, get_model_path

# 保存评估结果
with open(get_output_path('evaluation.txt'), 'w') as f:
    f.write(evaluation_text)

# 保存模型
classifier.model_path = get_model_path('optimized_model.pkl')
classifier.save_model()
```

## 📊 详细对比

| 指标 | V1 | V2 | 提升 |
|------|----|----|------|
| 准确率 | 73.46% | **79.20%** | +5.74% |
| 精确率 | 73.59% | **76.96%** | +3.37% |
| 召回率 | 84.86% | **91.73%** | +6.87% |
| F1分数 | 78.82% | **83.69%** | +4.87% |
| 特征维度 | 5,000 | **15,022** | +10,022 |
| 训练时间 | ~2分钟 | ~3分钟 | +50% |
| 模型大小 | 11 MB | 44 MB | +300% |

## 💡 优化经验

### ✅ 有效的优化
- 字符级TF-IDF: 捕捉拼写错误
- 统计特征: 简单但有效
- ComplementNB: 比MultinomialNB稳定
- 三元组(trigram): 捕捉更长短语

### ❌ 无效的尝试
- 增加max_features到50000: 过拟合
- GaussianNB: 性能下降
- 复杂正则表达式特征: 噪声过多

## 🔗 与其他阶段的关系

- **依赖**: Stage1(评估工具), Stage2(原始NB实现)
- **对比**: Stage2的NB V1作为baseline
- **启发**: 为Stage4的BERT优化提供特征工程思路

## ⚠️ 注意事项

1. **模型大小**: 优化版模型较大(44MB),需要更多磁盘空间
2. **训练时间**: 比V1慢50%(3分钟 vs 2分钟)
3. **特征工程**: 22个统计特征需要仔细调试
4. **路径配置**: 使用config.py确保输出到正确位置

## 📚 相关文档

- **README.md** - 阶段概述和性能对比
- **OPTIMIZATION_SUMMARY.md** (根目录) - 详细优化过程
- **IMPLEMENTATION.md** (本文档) - 实现说明

---

**实现完成度**: ✅ 100%
**输出位置**: ✅ 已配置到本阶段目录
**优化效果**: ✅ +5.74%准确率达成

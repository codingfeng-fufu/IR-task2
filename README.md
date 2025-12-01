# 学术标题分类系统 / Scholar Title Classification System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目简介

本项目是一个基于机器学习的学术论文标题分类系统，旨在识别从 **CiteSeer 数据库**中错误提取的学术论文标题。系统采用多模型对比方法，实现了从传统机器学习到深度学习的完整技术栈。

### 问题背景

CiteSeer 是一个学术论文搜索引擎，在自动提取论文标题时可能会出现错误，例如：
- ❌ **错误提取**: `"Abstract......Introduction......References"`
- ❌ **元数据混入**: `"IEEE Transactions on Pattern Analysis, Vol. 25, pp. 1-15"`
- ❌ **标题片段**: `"Table of Contents......Chapter 1"`
- ✅ **正确标题**: `"Deep Learning for Computer Vision Applications"`

### 核心功能

- 🎯 **二分类任务**: 区分正确标题 (Label=1) 和错误标题 (Label=0)
- 🔬 **三种模型对比**: 朴素贝叶斯、Word2Vec+SVM、BERT
- 📊 **全面评估**: 准确率、精确率、召回率、F1分数等多维度指标
- 📈 **可视化分析**: 性能对比图、混淆矩阵、t-SNE降维可视化
- ⚡ **优化技术**: 包含 8 种 BERT 优化技术的高级版本

---

## 🏗️ 项目结构

```
IR-task2/
│
├── core/                           # 核心模块目录
│   ├── data_loader.py             # 数据加载与预处理
│   ├── naive_bayes_classifier.py  # 朴素贝叶斯分类器 (TF-IDF)
│   ├── word2vec_svm_classifier.py # Word2Vec + SVM 分类器
│   ├── bert_classifier.py         # 标准 BERT 分类器
│   ├── evaluator.py               # 模型评估模块
│   ├── visualizer.py              # 结果可视化模块
│   └── main_pipeline.py           # 主执行流程 (完整流水线)
│
├── optimized_BERT.py              # 优化版 BERT (8种优化技术)
├── train_optimized_bert.py        # 优化 BERT 训练脚本
│
├── data/                          # 数据目录 (需自行准备)
│   ├── positive.txt               # 正样本 (正确标题)
│   ├── negative.txt               # 负样本 (错误标题)
│   └── testSet-1000.xlsx          # 测试集
│
├── output/                        # 输出目录 (自动生成)
│   ├── model_comparison.png       # 模型性能对比图
│   ├── confusion_matrices.png     # 混淆矩阵热图
│   ├── tsne_*.png                 # t-SNE 可视化
│   ├── evaluation_results.txt     # 评估结果
│   └── predictions.json           # 预测结果
│
├── README.md                      # 项目文档 (本文件)
└── .gitignore                     # Git 忽略配置
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.7 或更高版本
- **操作系统**: Windows / Linux / macOS
- **硬件**:
  - CPU: 4 核以上推荐
  - 内存: 8GB 以上
  - GPU: 可选 (用于加速 BERT 训练，CUDA 支持)

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd IR-task2

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖包
pip install numpy pandas scikit-learn
pip install torch transformers  # PyTorch 和 Transformers
pip install gensim              # Word2Vec
pip install matplotlib seaborn  # 可视化
pip install tqdm openpyxl       # 工具库
```

### 3. 准备数据

在 `data/` 目录下准备以下文件：

#### (1) 训练数据
- **positive.txt**: 正确的学术标题，每行一个标题
  ```
  Deep Learning for Computer Vision Applications
  Natural Language Processing with Transformer Models
  Introduction to Machine Learning Algorithms
  ...
  ```

- **negative.txt**: 错误提取的标题，每行一个
  ```
  Abstract......Introduction......References
  IEEE Transactions on Pattern Analysis, Vol. 25
  Table of Contents......Chapter 1
  ...
  ```

#### (2) 测试数据
- **testSet-1000.xlsx**: Excel 文件，包含两列
  - `title given by manchine`: 待分类的标题
  - `Y/N`: 标签 (Y=正确标题, N=错误标题)

> **注意**: 如果没有数据文件，程序会自动使用示例数据进行演示

### 4. 运行程序

#### 方式一：完整流水线 (推荐新手)

运行所有三个模型并对比：

```bash
cd core
python main_pipeline.py
```

**配置选项** (在 `main_pipeline.py` 中修改):
```python
USE_SAMPLE_DATA = False     # 是否使用示例数据
MAX_TRAIN_SAMPLES = None    # 训练样本数限制 (None=全部)
TRAIN_ONLY_BERT = False     # 是否只训练 BERT
BERT_EPOCHS = 5             # BERT 训练轮数
OUTPUT_DIR = 'output'       # 输出目录
```

#### 方式二：优化版 BERT (推荐进阶)

只训练优化版 BERT，获得最佳性能：

```bash
python train_optimized_bert.py
```

#### 方式三：单独测试模型

测试单个分类器：

```bash
cd core
python naive_bayes_classifier.py      # 测试朴素贝叶斯
python word2vec_svm_classifier.py     # 测试 Word2Vec+SVM
python bert_classifier.py             # 测试 BERT
```

---

## 🎯 三种分类方法详解

### 方法 1: 朴素贝叶斯分类器 (Naive Bayes)

**文件**: `core/naive_bayes_classifier.py`

#### 算法原理
- **特征提取**: TF-IDF (Term Frequency - Inverse Document Frequency)
  - 词频 (TF): 词在文档中出现的频率
  - 逆文档频率 (IDF): 词的区分能力 (罕见词权重更高)
  - 公式: `TF-IDF = TF × log(N / df)`
- **分类器**: MultinomialNB (多项式朴素贝叶斯)
  - 基于贝叶斯定理: `P(类别|文档) ∝ P(文档|类别) × P(类别)`
  - Laplace 平滑避免零概率

#### 技术特点
- ✅ **优势**: 训练速度快、可解释性强、适合文本分类
- ❌ **劣势**: 假设特征独立 (实际上词之间有依赖关系)
- ⚙️ **参数**:
  - `max_features=5000`: 最多 5000 个特征
  - `ngram_range=(1,2)`: 使用 1-gram 和 2-gram
  - `alpha=1.0`: Laplace 平滑参数

#### 关键代码
```python
# TF-IDF 向量化
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(titles)

# 训练朴素贝叶斯
classifier = MultinomialNB(alpha=1.0)
classifier.fit(X_train, labels)
```

---

### 方法 2: Word2Vec + SVM 分类器

**文件**: `core/word2vec_svm_classifier.py`

#### 算法原理
1. **Word2Vec 词嵌入**:
   - 将每个词映射到 100 维向量空间
   - 语义相似的词在向量空间中距离更近
   - 使用 CBOW (Continuous Bag of Words) 模型训练

2. **标题向量化**:
   - 方法: 对标题中所有词向量取平均
   - 公式: `title_vec = (1/n) × Σ word_vec_i`

3. **特征工程** (8 个统计特征):
   - 标题长度、平均词长
   - 大写字母比例、数字比例
   - 特殊字符数、是否含数字、全大写/小写标志

4. **SVM 分类**:
   - 使用 RBF 核函数 (非线性分类)
   - 找到最优超平面分隔两类样本

#### 技术特点
- ✅ **优势**:
  - 捕捉语义信息 (Word2Vec)
  - 手工特征增强性能
  - SVM 在高维空间表现优秀
- ❌ **劣势**:
  - 训练时间较长
  - 需要手工设计特征
- ⚙️ **参数**:
  - `vector_size=100`: 词向量维度
  - `window=5`: 上下文窗口
  - `use_linear_svm=False`: 使用 RBF 核
  - `add_features=True`: 启用统计特征

#### 关键代码
```python
# 训练 Word2Vec
w2v_model = Word2Vec(sentences=tokenized_titles, vector_size=100, window=5)

# 标题向量化
def title_to_vector(title):
    word_vecs = [w2v_model.wv[word] for word in title.split() if word in w2v_model.wv]
    avg_vec = np.mean(word_vecs, axis=0) if word_vecs else np.zeros(100)
    stat_features = extract_statistical_features(title)  # 8 维
    return np.concatenate([avg_vec, stat_features])      # 108 维

# 训练 SVM
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, labels)
```

---

### 方法 3: BERT 分类器 (Transformer)

**文件**: `core/bert_classifier.py` (标准版) 和 `optimized_BERT.py` (优化版)

#### 算法原理
- **BERT** (Bidirectional Encoder Representations from Transformers)
  - 预训练模型: `bert-base-uncased` (12层, 768维, 110M参数)
  - 双向 Transformer: 同时考虑上下文信息
  - [CLS] token: 句子级别的表示向量

- **微调** (Fine-tuning):
  - 在预训练 BERT 基础上添加分类层
  - 使用标题数据进行端到端训练
  - 自动学习任务相关的特征表示

#### 标准版 BERT

**文件**: `core/bert_classifier.py`

**特点**:
- 基础 BERT 微调
- AdamW 优化器 + 线性学习率预热
- 5 轮训练，batch_size=32

**关键代码**:
```python
# 加载预训练模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 训练循环
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=500)

for epoch in range(epochs):
    for batch in dataloader:
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
```

#### 优化版 BERT ⭐

**文件**: `optimized_BERT.py`

**8 种优化技术**:

1. **FGM 对抗训练** (Fast Gradient Method)
   - 在嵌入层添加对抗扰动，提高鲁棒性
   - `r_adv = ε × grad / ||grad||`

2. **EMA 指数移动平均** (Exponential Moving Average)
   - 平滑模型参数，提高泛化能力
   - `θ_ema = 0.999 × θ_ema + 0.001 × θ`

3. **差异化学习率**
   - 分类层学习率 = 10 × BERT 基础学习率
   - 快速适应新任务

4. **Warmup + Cosine 学习率调度**
   - 前 10% 步数线性增加学习率
   - 后续按余弦曲线衰减

5. **早停机制** (Early Stopping)
   - 监控验证集 F1 分数
   - Patience=3，防止过拟合

6. **数据增强**
   - 随机删除词 (10% 概率)
   - 随机交换相邻词 (10% 概率)

7. **Focal Loss** (可选)
   - 处理类别不平衡
   - `FL = -α(1-p)^γ × log(p)`

8. **梯度裁剪**
   - 限制梯度范数 ≤ 1.0
   - 防止梯度爆炸

**关键代码**:
```python
# 对抗训练 (FGM)
fgm = FGM(model, epsilon=1.0)
loss.backward()
fgm.attack()           # 添加对抗扰动
loss_adv.backward()    # 对抗样本的梯度
fgm.restore()          # 恢复原始参数

# EMA
ema = EMA(model, decay=0.999)
ema.update()           # 每步更新 EMA 参数
ema.apply_shadow()     # 验证时使用 EMA 参数
```

#### 技术特点
- ✅ **优势**:
  - 最先进的 NLP 模型
  - 自动特征学习
  - 优化版性能最佳
- ❌ **劣势**:
  - 训练时间长 (GPU 推荐)
  - 模型参数多 (110M)
  - 需要大量数据
- ⚙️ **参数**:
  - `max_length=64`: 最大序列长度
  - `epochs=10`: 训练轮数 (优化版)
  - `batch_size=16`: 批次大小
  - `learning_rate=2e-5`: 基础学习率

---

## 📊 模型评估

### 评估指标

系统使用多维度指标全面评估模型性能：

| 指标 | 说明 | 公式 |
|------|------|------|
| **准确率** (Accuracy) | 预测正确的样本比例 | `(TP + TN) / Total` |
| **精确率** (Precision) | 预测为正例中真正例的比例 | `TP / (TP + FP)` |
| **召回率** (Recall) | 真正例中被预测出的比例 | `TP / (TP + FN)` |
| **F1 分数** | 精确率和召回率的调和平均 | `2 × P × R / (P + R)` |
| **F1 宏平均** | 各类别 F1 的算术平均 | `(F1_0 + F1_1) / 2` |
| **F1 微平均** | 全局计算 F1 (等于准确率) | `2 × TP / (2×TP + FP + FN)` |

**混淆矩阵**:
```
                预测为负   预测为正
实际为负 (0)      TN        FP
实际为正 (1)      FN        TP
```

### 性能对比 (参考)

基于 1000 样本测试集的典型结果：

| 模型 | 准确率 | 精确率 | 召回率 | F1 分数 | 训练时间 |
|------|--------|--------|--------|---------|----------|
| Naive Bayes | 0.89 | 0.88 | 0.90 | 0.89 | ~1 分钟 |
| Word2Vec+SVM | 0.92 | 0.91 | 0.93 | 0.92 | ~5 分钟 |
| BERT (标准) | 0.95 | 0.94 | 0.96 | 0.95 | ~30 分钟 |
| BERT (优化) | **0.97** | **0.96** | **0.98** | **0.97** | ~45 分钟 |

> 注: 实际性能取决于数据质量、数据量和硬件配置

---

## 📈 可视化输出

系统自动生成以下可视化图表 (保存在 `output/` 目录):

### 1. 模型性能对比图
**文件**: `model_comparison.png`

6 个子图对比所有模型的指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1 分数
- F1 宏平均
- F1 微平均

### 2. 混淆矩阵热图
**文件**: `confusion_matrices.png`

显示每个模型的分类细节：
- 真负例 (TN)、假正例 (FP)
- 假负例 (FN)、真正例 (TP)

### 3. t-SNE 降维可视化
**文件**: `tsne_*.png`

将高维特征向量投影到 2D 平面：
- 红色点: 负样本 (错误标题)
- 绿色点: 正样本 (正确标题)
- 显示模型的特征空间分布

---

## 🛠️ 核心模块说明

### 1. 数据加载器 (data_loader.py)

**功能**:
- 从文本文件加载训练数据
- 从 Excel 加载测试数据
- 文本预处理 (小写、去特殊字符)
- 创建示例数据

**关键函数**:
```python
# 加载数据集
train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
    'data/positive.txt',
    'data/negative.txt',
    'data/testSet-1000.xlsx'
)

# 预处理单个标题
clean_title = DataLoader.preprocess_title("Deep Learning for CV!")
# 输出: "deep learning for cv"
```

### 2. 模型评估器 (evaluator.py)

**功能**:
- 计算各种评估指标
- 生成混淆矩阵
- 模型性能对比
- 错误分析

**关键函数**:
```python
evaluator = ModelEvaluator()

# 评估单个模型
result = evaluator.evaluate_model(y_true, y_pred, "Model Name")

# 比较多个模型
evaluator.compare_models([result1, result2, result3])

# 错误分析
error_analysis = evaluator.calculate_error_analysis(y_true, y_pred, titles)
```

### 3. 结果可视化器 (visualizer.py)

**功能**:
- 生成性能对比图
- 绘制混淆矩阵热图
- t-SNE 降维可视化

**关键函数**:
```python
visualizer = ResultVisualizer()

# 性能对比图
visualizer.plot_comparison(results, save_path='comparison.png')

# 混淆矩阵
visualizer.plot_confusion_matrices(results, save_path='confusion.png')

# t-SNE 可视化
visualizer.visualize_embeddings_tsne(vectors, labels, "Model", save_path='tsne.png')
```

---

## ⚙️ 配置与参数调优

### 朴素贝叶斯参数

```python
classifier = NaiveBayesClassifier(
    max_features=5000,      # TF-IDF 特征数 (↑ 增加可能提高准确率但增加计算量)
    ngram_range=(1, 2)      # n-gram 范围 (1,1)=仅unigram, (1,2)=uni+bigram, (1,3)=uni+bi+trigram
)
```

**调优建议**:
- 小数据集: `max_features=3000`, `ngram_range=(1,1)`
- 大数据集: `max_features=10000`, `ngram_range=(1,3)`

### Word2Vec + SVM 参数

```python
classifier = Word2VecSVMClassifier(
    vector_size=100,        # 词向量维度 (50/100/200，↑ 增加更丰富但训练慢)
    window=5,               # 上下文窗口 (3-10，短标题用小值)
    min_count=2,            # 最小词频 (1-5，小数据集用1)
    epochs=10,              # Word2Vec 训练轮数 (5-20)
    use_linear_svm=False,   # True=LinearSVC(快), False=RBF核(准)
    add_features=True       # 是否添加统计特征 (推荐True)
)
```

**调优建议**:
- 快速原型: `use_linear_svm=True`, `epochs=5`
- 最佳性能: `use_linear_svm=False`, `add_features=True`, `epochs=20`

### BERT 参数

#### 标准版
```python
classifier = BERTClassifier(
    model_name='bert-base-uncased',  # 预训练模型
    max_length=64                     # 最大序列长度 (32-128)
)

classifier.train(
    train_titles, train_labels,
    epochs=5,                         # 训练轮数 (3-10)
    batch_size=32,                    # 批次大小 (8/16/32/64)
    learning_rate=2e-5,               # 学习率 (1e-5 ~ 5e-5)
    warmup_steps=500                  # 预热步数
)
```

#### 优化版
```python
classifier = BERTClassifierOptimized(
    model_name='bert-base-uncased',
    max_length=64,
    use_fgm=True,                     # 对抗训练
    use_ema=True                      # 指数移动平均
)

classifier.train(
    train_titles, train_labels,
    epochs=10,                        # 训练轮数 (5-15)
    batch_size=16,                    # 批次大小 (↓ 减少显存)
    learning_rate=2e-5,
    warmup_ratio=0.1,                 # 预热比例 (0.05-0.15)
    weight_decay=0.01,                # 权重衰减 (正则化)
    patience=3,                       # 早停耐心值
    use_focal_loss=False,             # Focal Loss (类别不平衡时用True)
    augment_data=True                 # 数据增强
)
```

**调优建议**:
- **小数据集** (<1000 样本):
  - `epochs=15`, `batch_size=8`, `warmup_ratio=0.15`
  - `augment_data=True` (重要!)

- **中数据集** (1000-10000):
  - `epochs=10`, `batch_size=16`, `warmup_ratio=0.1`

- **大数据集** (>10000):
  - `epochs=5`, `batch_size=32`, `warmup_ratio=0.05`

- **类别不平衡**:
  - `use_focal_loss=True`

### 硬件配置建议

| 配置 | CPU | GPU | 内存 | 推荐模型 |
|------|-----|-----|------|----------|
| 最低 | 2核 | 无 | 4GB | Naive Bayes |
| 推荐 | 4核 | 无 | 8GB | Word2Vec+SVM |
| 高性能 | 8核 | GTX 1060+ (6GB) | 16GB | BERT (标准) |
| 顶配 | 16核 | RTX 3080+ (10GB) | 32GB | BERT (优化) |

---

## 🔧 常见问题 (FAQ)

### Q1: 如何处理数据文件不存在的情况？

**A**: 程序会自动使用示例数据。如需使用真实数据：
```python
# 在 main_pipeline.py 中设置
USE_SAMPLE_DATA = False

# 确保数据文件存在
data/
  ├── positive.txt
  ├── negative.txt
  └── testSet-1000.xlsx
```

### Q2: BERT 训练太慢怎么办？

**A**: 几种加速方法：
1. **减少数据**: `MAX_TRAIN_SAMPLES = 5000`
2. **减少轮数**: `epochs=3`
3. **增大批次**: `batch_size=32` (需要更多显存)
4. **只训练BERT**: `TRAIN_ONLY_BERT = True`
5. **使用GPU**: 安装 CUDA 版 PyTorch

### Q3: 显存不足 (CUDA out of memory)

**A**:
```python
# 减小批次大小
batch_size = 8  # 或更小 (4)

# 减小序列长度
max_length = 32  # 原来是 64

# 使用梯度累积 (未实现，需手动添加)
```

### Q4: 如何提高模型准确率？

**A**: 按优先级尝试：
1. **增加训练数据** (最重要!)
2. **使用优化版BERT** (`train_optimized_bert.py`)
3. **调整超参数** (学习率、轮数)
4. **数据清洗** (移除噪声样本)
5. **特征工程** (Word2Vec+SVM 的统计特征)

### Q5: 如何保存训练好的模型？

**A**:
```python
# BERT 模型会自动保存
# 文件: best_bert_model.pt

# 加载模型
model.load_state_dict(torch.load('best_bert_model.pt'))

# 其他模型需手动保存
import pickle
with open('nb_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
```

### Q6: 可以用于其他分类任务吗？

**A**: 可以! 只需：
1. 准备数据 (文本 + 二分类标签)
2. 修改数据加载部分
3. 无需修改模型代码

### Q7: 支持多分类吗？

**A**: 当前是二分类，改为多分类需修改：
```python
# BERT
num_labels = 5  # 改为类别数

# 朴素贝叶斯和SVM自动支持多分类
```

---

## 📚 技术栈

### 核心库

| 库 | 版本 | 用途 |
|---|------|------|
| Python | 3.7+ | 编程语言 |
| NumPy | 1.19+ | 数值计算 |
| Pandas | 1.2+ | 数据处理 |
| Scikit-learn | 0.24+ | 传统机器学习 |
| PyTorch | 1.9+ | 深度学习框架 |
| Transformers | 4.10+ | BERT 模型 |
| Gensim | 4.0+ | Word2Vec |
| Matplotlib | 3.3+ | 可视化 |
| Seaborn | 0.11+ | 高级可视化 |
| tqdm | 4.60+ | 进度条 |

### 算法与模型

- **朴素贝叶斯**: MultinomialNB
- **TF-IDF**: TfidfVectorizer
- **Word2Vec**: CBOW 模型
- **SVM**: RBF 核 / Linear
- **BERT**: bert-base-uncased (Hugging Face)
- **优化器**: AdamW
- **学习率调度**: Linear / Cosine with Warmup

---

## 🎓 算法理论

### 1. TF-IDF 原理

**TF (Term Frequency)**: 词频
```
TF(t, d) = count(t in d) / total_words(d)
```

**IDF (Inverse Document Frequency)**: 逆文档频率
```
IDF(t) = log(N / df(t))
```
其中 N 是文档总数，df(t) 是包含词 t 的文档数

**TF-IDF**:
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**意义**:
- 高 TF: 词在当前文档中重要
- 高 IDF: 词在所有文档中罕见 (区分度高)

### 2. 朴素贝叶斯原理

**贝叶斯定理**:
```
P(C|X) = P(X|C) × P(C) / P(X)
```

**朴素假设**: 特征相互独立
```
P(X|C) = P(x1|C) × P(x2|C) × ... × P(xn|C)
```

**分类决策**:
```
C* = argmax_C P(C|X) = argmax_C P(X|C) × P(C)
```

### 3. Word2Vec 原理

**CBOW (Continuous Bag of Words)**:
- 输入: 上下文词
- 输出: 中心词
- 目标: 最大化 `P(中心词 | 上下文)`

**Skip-gram**:
- 输入: 中心词
- 输出: 上下文词
- 目标: 最大化 `P(上下文 | 中心词)`

**负采样** (Negative Sampling):
- 只更新少量负样本，加速训练

### 4. SVM 原理

**目标**: 找到最大间隔超平面

**线性SVM**:
```
minimize: 1/2 ||w||² + C Σ ξi
subject to: yi(w·xi + b) ≥ 1 - ξi
```

**RBF 核**:
```
K(xi, xj) = exp(-γ ||xi - xj||²)
```
将数据映射到高维空间，实现非线性分类

### 5. BERT 原理

**Transformer 架构**:
- Self-Attention: `Attention(Q, K, V) = softmax(QK^T / √dk) V`
- Multi-Head Attention: 多个 attention 并行
- Feed-Forward: 两层全连接网络

**预训练任务**:
1. **Masked LM**: 随机遮盖 15% 的词，预测被遮盖的词
2. **Next Sentence Prediction**: 判断两个句子是否连续

**微调**:
- 添加分类层 `Linear(768 → 2)`
- 端到端训练

### 6. 优化技术原理

**对抗训练 (FGM)**:
```
r_adv = ε × ∇_emb L / ||∇_emb L||
emb_adv = emb + r_adv
L_total = L(emb) + L(emb_adv)
```

**EMA**:
```
θ_ema^(t) = α × θ_ema^(t-1) + (1-α) × θ^(t)
```

**Warmup**:
```
lr(t) = lr_max × min(t/warmup_steps, 1)
```

**Cosine Annealing**:
```
lr(t) = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))
```

---

## 📝 输出文件说明

### 1. evaluation_results.txt

包含所有模型的详细评估指标：
```
模型: Naive Bayes
  准确率: 0.8900
  精确率: 0.8800
  召回率: 0.9000
  F1分数: 0.8899
  ...
```

### 2. predictions.json

每个模型的预测结果：
```json
{
  "Naive Bayes": [1, 0, 1, 0, ...],
  "Word2Vec+SVM": [1, 0, 1, 1, ...],
  "BERT": [1, 0, 1, 0, ...]
}
```

### 3. 可视化图表

- **model_comparison.png**: 6 个子图对比所有指标
- **confusion_matrices.png**: 混淆矩阵热图
- **tsne_*.png**: 每个模型的 t-SNE 降维图

---

## 🚀 进阶使用

### 自定义数据增强

```python
# 在 optimized_BERT.py 的 TitleDataset 类中修改
def augment_text(self, text: str) -> str:
    words = text.split()

    # 自定义增强策略
    # 1. 同义词替换
    # 2. 回译 (Back Translation)
    # 3. 词序打乱

    return ' '.join(words)
```

### 集成学习

```python
# 组合多个模型的预测
def ensemble_predict(models, titles):
    predictions = []
    for model in models:
        pred = model.predict(titles)
        predictions.append(pred)

    # 投票法
    final_pred = np.round(np.mean(predictions, axis=0))
    return final_pred
```

### 超参数搜索

```python
from sklearn.model_selection import GridSearchCV

# 对SVM进行网格搜索
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 1]
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

---

## 📖 参考文献

1. **TF-IDF**:
   - Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.

2. **朴素贝叶斯**:
   - McCallum, A., & Nigam, K. (1998). A comparison of event models for naive bayes text classification.

3. **Word2Vec**:
   - Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. ICLR.

4. **BERT**:
   - Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL.

5. **对抗训练**:
   - Miyato, T., et al. (2017). Adversarial training methods for semi-supervised text classification. ICLR.

6. **EMA**:
   - Polyak, B. T., & Juditsky, A. B. (1992). Acceleration of stochastic approximation by averaging.

---

## 🤝 贡献指南

欢迎贡献代码! 请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 遵循 PEP 8 风格
- 添加详细的注释和文档字符串
- 保持函数简洁 (<50 行)
- 使用有意义的变量名

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📮 联系方式

- **项目维护者**: [Your Name]
- **Email**: your.email@example.com
- **项目主页**: https://github.com/yourusername/IR-task2

---

## 🙏 致谢

- Hugging Face Transformers 团队
- Scikit-learn 社区
- PyTorch 开发团队
- 所有贡献者

---

## 📊 更新日志

### v2.0.0 (2024-12-02)
- ✨ 添加优化版 BERT (8种优化技术)
- ✨ 增强数据增强功能
- 📝 完善文档和注释
- 🐛 修复已知 bug

### v1.0.0 (2024-11-01)
- 🎉 初始版本发布
- ✅ 实现三种分类方法
- 📊 添加完整评估和可视化

---

**祝你使用愉快! 如有问题请提 Issue 或 PR 🎉**

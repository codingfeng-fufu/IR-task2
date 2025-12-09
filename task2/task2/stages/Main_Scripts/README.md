# Main Scripts - 主要运行脚本

**用途**：整合所有模型的主流水线和评估脚本
**特点**：一键运行完整实验流程

## 📁 文件列表

| 文件 | 行数 | 功能 | 使用场景 |
|------|------|------|----------|
| `main_pipeline.py` | 466 | 完整流水线（三种模型） | 训练+评估+可视化 |
| `evaluate_saved.py` | 423 | 已保存模型评估 | 快速评估 |
| `run_optimized_classifier.py` | 368 | 运行优化分类器 | 单独训练 |

## 🎯 脚本说明

### 1. main_pipeline.py - 主流水线

**功能**：端到端完整流水线

**运行的模型**：
1. ✅ 朴素贝叶斯（优化版）- 79.20% accuracy
2. ✅ Word2Vec + SVM - 82.99% accuracy
3. ✅ BERT - 87.91% accuracy

**执行流程**：
```
[步骤 1/5] 加载数据集
  ├── 训练集: 232,402 条（positive.txt + negative.txt）
  └── 测试集: 1,000 条（testSet-1000.xlsx）

[步骤 2/5] 训练模型
  ├── 朴素贝叶斯（优化版）- ~3分钟
  ├── Word2Vec + SVM - ~10分钟
  └── BERT - ~1小时（GPU）

[步骤 3/5] 模型评估
  ├── 准确率、精确率、召回率、F1
  ├── 混淆矩阵
  └── 错误分析（FP/FN样本）

[步骤 4/5] 生成可视化
  ├── 模型性能对比图
  ├── 混淆矩阵热力图
  └── t-SNE特征空间可视化

[步骤 5/5] 保存结果
  ├── models/ - 模型权重
  ├── output/ - 图表和报告
  └── logs/ - 训练日志
```

**配置选项**（文件开头）：
```python
# 数据配置
USE_SAMPLE_DATA = False      # 使用示例数据（当文件缺失时）
MAX_TRAIN_SAMPLES = None     # 限制训练集大小（None=全部）

# 模型配置
TRAIN_ONLY_BERT = False      # 仅训练BERT（跳过NB和W2V）
BERT_EPOCHS = 5              # BERT训练轮数

# 输出配置
OUTPUT_DIR = 'output'        # 输出目录
```

**使用示例**：
```bash
# 完整训练（所有模型）
python main_pipeline.py

# 仅训练BERT（编辑文件：TRAIN_ONLY_BERT = True）
vim main_pipeline.py
python main_pipeline.py

# 快速测试（限制训练集）
# 编辑文件：MAX_TRAIN_SAMPLES = 10000
python main_pipeline.py
```

**输出文件**：
```
output/
├── model_comparison.png          # 模型性能对比
├── confusion_matrices.png        # 混淆矩阵
├── tsne_Naive_Bayes.png         # t-SNE可视化
├── tsne_Word2Vec_SVM.png
├── tsne_BERT.png
├── evaluation_results.txt        # 详细指标
└── predictions.json              # 预测结果

models/
├── naive_bayes_optimized_model.pkl
├── word2vec_svm_model_w2v.model
├── word2vec_svm_model_svm.pkl
└── best_bert_model.pt
```

### 2. evaluate_saved.py - 评估已保存模型

**功能**：快速评估已训练模型（无需重新训练）

**适用场景**：
- ✅ 模型已训练完成
- ✅ 只需要重新评估
- ✅ 测试集发生变化
- ✅ 需要更详细的错误分析

**使用示例**：
```bash
# 评估所有已保存的模型
python evaluate_saved.py

# 查看详细的错误分析
# 输出会包含FP/FN样本
```

**加载的模型**：
- `models/naive_bayes_optimized_model.pkl`
- `models/word2vec_svm_model_w2v.model` + `models/word2vec_svm_model_svm.pkl`
- `models/best_bert_model.pt`

**优势**：
- ⚡ 快速（无需训练）
- 📊 完整评估
- 🔍 详细错误分析
- 💾 节省时间

**注意事项**：
- 确保模型文件存在于 `models/` 目录
- 测试集文件必须存在：`data/testSet-1000.xlsx`

### 3. run_optimized_classifier.py - 运行优化分类器

**功能**：单独训练和测试优化版分类器

**支持的模型**：
- 朴素贝叶斯（优化版）
- Word2Vec + SVM
- BERT

**使用示例**：
```bash
# 运行单个优化分类器
python run_optimized_classifier.py

# 可在脚本中选择特定模型
```

**特点**：
- 🎯 聚焦单个模型
- 📊 详细训练过程
- 🔧 易于调参
- 📈 实时性能监控

## 🚀 快速使用指南

### 场景1：首次运行（完整训练）

```bash
cd /home/u2023312337/task2/task2
source .venv/bin/activate
python main_pipeline.py
```

**预计时间**：
- 朴素贝叶斯：~3分钟
- Word2Vec+SVM：~10分钟
- BERT：~1小时（GPU）
- **总计**：~1.2小时

### 场景2：已有模型（快速评估）

```bash
cd /home/u2023312337/task2/task2
source .venv/bin/activate
python evaluate_saved.py
```

**预计时间**：~2分钟

### 场景3：仅训练BERT

```bash
cd /home/u2023312337/task2/task2
source .venv/bin/activate

# 编辑 main_pipeline.py
# 修改：TRAIN_ONLY_BERT = True
vim main_pipeline.py

python main_pipeline.py
```

**预计时间**：~1小时

### 场景4：快速测试（小数据集）

```bash
cd /home/u2023312337/task2/task2
source .venv/bin/activate

# 编辑 main_pipeline.py
# 修改：MAX_TRAIN_SAMPLES = 10000
vim main_pipeline.py

python main_pipeline.py
```

**预计时间**：~15分钟

## 📊 预期输出

### 终端输出示例

```
================================================================================
               学术标题分类系统
          Scholar Title Classification System
================================================================================

项目描述:
  识别CiteSeer数据库中错误提取的学术论文标题
  使用三种机器学习方法:
    1. 朴素贝叶斯 (Naive Bayes - 优化版)
    2. Word2Vec + SVM
    3. BERT (Transformer)

================================================================================

[步骤 1/5] 加载数据集
--------------------------------------------------------------------------------
数据加载完成!
  训练集: 232402 条
  测试集: 1000 条
  正样本比例: 51.2%

[步骤 2/5] 训练模型
--------------------------------------------------------------------------------
训练朴素贝叶斯（优化版）...
  优化版朴素贝叶斯训练完成
  时间: 174.23s

训练Word2Vec + SVM...
  Word2Vec训练完成，词汇量: 45234
  SVM训练完成
  时间: 621.45s

训练BERT...
  使用设备: cuda
  Epoch 1/5: 100%|████| Loss: 0.234
  ...
  BERT训练完成
  时间: 3245.67s

[步骤 3/5] 模型评估
--------------------------------------------------------------------------------
朴素贝叶斯（优化版）评估结果:
  准确率: 79.20%
  精确率: 76.96%
  召回率: 91.73%
  F1分数: 83.69%

Word2Vec + SVM评估结果:
  准确率: 82.99%
  精确率: 85.84%
  召回率: 85.58%
  F1分数: 85.74%

BERT评估结果:
  准确率: 87.91%
  精确率: 90.29%
  召回率: 88.35%
  F1分数: 89.59%

[步骤 4/5] 生成可视化
--------------------------------------------------------------------------------
✓ 模型对比图: output/model_comparison.png
✓ 混淆矩阵: output/confusion_matrices.png
✓ t-SNE可视化: output/tsne_*.png

[步骤 5/5] 保存结果
--------------------------------------------------------------------------------
✓ 评估报告: output/evaluation_results.txt
✓ 预测结果: output/predictions.json
✓ 模型权重: models/*.pkl, models/*.pt

================================================================================
实验完成！
总用时: 4041.35s (1.12小时)
================================================================================
```

### 生成的文件

```
task2/task2/
├── output/
│   ├── model_comparison.png          # 三个模型的性能对比条形图
│   ├── confusion_matrices.png        # 3x1混淆矩阵热力图
│   ├── tsne_Naive_Bayes.png         # 朴素贝叶斯特征空间
│   ├── tsne_Word2Vec_SVM.png        # Word2Vec特征空间
│   ├── tsne_BERT.png                # BERT特征空间
│   ├── evaluation_results.txt        # 详细评估指标（文本）
│   └── predictions.json              # 所有模型的预测结果
│
└── models/
    ├── naive_bayes_optimized_model.pkl      # 44 MB
    ├── word2vec_svm_model_w2v.model        # 25 MB
    ├── word2vec_svm_model_svm.pkl          # 114 MB
    └── best_bert_model.pt                   # 438 MB
```

## 💡 使用技巧

### 1. 加速训练

**限制训练集大小**：
```python
MAX_TRAIN_SAMPLES = 50000  # 从232K减少到50K
```
- 训练时间：~20分钟（vs 1.2小时）
- 性能下降：~2-3%

**仅训练BERT**：
```python
TRAIN_ONLY_BERT = True
```
- 跳过朴素贝叶斯和Word2Vec
- 适合BERT调参

### 2. 调试技巧

**打印详细日志**：
```python
# 在评估函数中设置 verbose=True
results = evaluator.evaluate_model(test_labels, predictions, "Model", verbose=True)
```

**查看错误样本**：
```python
# 在 main_pipeline.py 中添加
error_analysis = evaluator.calculate_error_analysis(test_labels, predictions, test_titles)
evaluator.print_error_analysis(error_analysis, max_examples=20)
```

### 3. 定制可视化

**修改图表大小**：
```python
# 在 visualizer.py 中调整
plt.figure(figsize=(12, 6))  # 默认 (10, 6)
```

**跳过某些图表**：
```python
# 在 main_pipeline.py 的 generate_visualizations() 中注释掉
# visualizer.plot_tsne(...)  # 跳过t-SNE
```

## 🔧 故障排除

### 问题1：找不到数据文件

**症状**：`FileNotFoundError: data/positive.txt`

**解决**：
- 确保 `data/` 目录下有 `positive.txt`, `negative.txt`, `testSet-1000.xlsx`
- 或者设置 `USE_SAMPLE_DATA = True` 使用示例数据

### 问题2：CUDA内存不足

**症状**：`RuntimeError: CUDA out of memory`

**解决**：
- 在 BERT 训练函数中减小 `batch_size`：32 → 16
- 或减小 `max_length`：96 → 64

### 问题3：模型文件不存在

**症状**：`evaluate_saved.py` 报错找不到模型

**解决**：
- 先运行 `python main_pipeline.py` 训练模型
- 确保 `models/` 目录下有对应的 `.pkl` 和 `.pt` 文件

### 问题4：导入模块失败

**症状**：`ModuleNotFoundError: No module named 'transformers'`

**解决**：
```bash
source .venv/bin/activate  # 激活虚拟环境
pip install -r requirements.txt  # 安装依赖
```

## 📚 相关文档

- **Stage1_Foundation/README.md** - 基础模块说明
- **Stage2_Traditional_Models/README.md** - 三种模型技术细节
- **Stage3_NaiveBayes_Optimization/README.md** - 朴素贝叶斯优化
- **Stage4_BERT_Optimization/README.md** - BERT优化实验
- **VERSION_EVOLUTION.md** - 完整版本演进
- **OPTIMIZATION_SUMMARY.md** - 优化总结

---

**总结**：这三个脚本是项目的主入口，提供了完整的训练-评估-可视化流程。

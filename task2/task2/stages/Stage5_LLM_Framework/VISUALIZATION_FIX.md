# Stage5 可视化修复总结

**修复日期**：2025-12-07
**问题**：t-SNE 可视化对 LLM 模型没有意义

## 问题分析

### Stage 1-4 的工作流程
```
输入文本 → 特征提取/编码 → 高维向量 → 分类器 → 分类结果
                              ↓
                          t-SNE 降维
                              ↓
                     2D 可视化（看聚类效果）
```

- NB/SVM/BERT 都会生成**中间特征向量**
- t-SNE 可视化这些向量，展示**模型学到的语义空间**
- 可以看到同类样本是否聚集在一起

### Stage 5 的工作流程
```
输入文本 → LLM API → 直接输出分类结果 (0/1)
           (黑盒)
```

- LLM 通过 API 调用，**无法获取内部特征向量**
- 直接输出分类结果，跳过了特征向量步骤
- **无法真正画 t-SNE**

## 原代码的问题

之前的代码使用 TF-IDF 作为替代：
```python
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
feature_vectors = vectorizer.fit_transform(test_titles).toarray()
visualizer.visualize_embeddings_tsne(feature_vectors, ...)  # 误导性！
```

问题：
- ❌ 不是可视化 LLM 学到的语义空间
- ❌ 只是可视化输入文本的统计特征
- ❌ 所有 LLM 用同样的 TF-IDF，图完全一样
- ❌ 误导用户以为是模型的特征空间

## 修复内容

### 1. 移除 train.py 的 t-SNE 代码
- 文件：`train.py`
- 移除：第 301-328 行的 t-SNE 生成代码
- 添加说明："注: LLM 模型通过 API 调用，无中间特征向量，无法生成 t-SNE 可视化"

### 2. 移除 run_llm_experiment.py 的 t-SNE 代码  
- 文件：`run_llm_experiment.py`
- 移除：第 800-827 行的 t-SNE 生成代码
- 添加说明：同上

### 3. 更新 README 文档
- 文件：`README.md`
- 移除：t-SNE 可视化文件的描述
- 添加：混淆矩阵和性能对比图的描述
- 添加说明：解释为什么 LLM 不能生成 t-SNE

### 4. 修复 visualizer.py 兼容性问题
- 文件：`stages/Stage1_Foundation/visualizer.py`
- 修复1：`plot_comparison` 动态检测可用指标（兼容 LLM 只有 f1 而没有 f1_macro/f1_micro）
- 修复2：`visualize_embeddings_tsne` 参数从 `n_iter` 改为 `max_iter`（新版 scikit-learn）

## 保留的可视化

✅ **混淆矩阵** (`llm_confusion_matrices.png`)
- 展示模型的分类混淆情况
- TP/TN/FP/FN 的分布
- 对所有模型都有意义

✅ **性能对比图** (`llm_comparison.png`)
- 对比不同 LLM 的 accuracy/precision/recall/f1
- 条形图展示
- 帮助选择最佳模型

❌ **t-SNE 可视化** (已移除)
- 对 LLM 没有意义
- 无法获取内部特征向量

## 验证

所有修改已完成并测试通过：
- ✓ train.py 不再生成 t-SNE
- ✓ run_llm_experiment.py 不再生成 t-SNE  
- ✓ README 已更新说明
- ✓ visualizer.py 兼容性问题已修复
- ✓ 混淆矩阵和性能对比图正常工作

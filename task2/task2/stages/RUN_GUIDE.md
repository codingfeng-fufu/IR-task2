# Stages 目录运行指南

## 📌 重要说明

`stages/` 目录是为了**展示项目演进和代码组织**而创建的，按时间阶段分类了所有文件。

**推荐的运行方式**：
- ✅ **在项目根目录运行原始文件**（路径：`/home/u2023312337/task2/task2/`）
- ⚠️ stages 目录主要用于查看和学习代码演进

## 🚀 方案1：在项目根目录运行（推荐）

```bash
# 进入项目根目录
cd /home/u2023312337/task2/task2

# 激活虚拟环境
source .venv/bin/activate

# 运行完整流水线
python main_pipeline.py

# 或运行其他脚本
python evaluate_saved.py
python check_environment.py
python test_optimized_nb.py
```

**优势**：
- ✅ 所有导入路径正确
- ✅ 数据文件路径正确（data/）
- ✅ 输出目录路径正确（output/，models/）
- ✅ 无需额外配置

## 🔧 方案2：在 stages 目录运行（实验性）

如果你确实想在 stages 目录运行，可以使用 `run_from_stages.py` 辅助脚本：

### 使用方法

```bash
# 进入 stages 目录
cd /home/u2023312337/task2/task2/stages

# 运行环境检查
python run_from_stages.py Stage1_Foundation/check_environment.py

# 运行完整流水线（需要先准备数据）
python run_from_stages.py Main_Scripts/main_pipeline.py

# 运行评估脚本
python run_from_stages.py Main_Scripts/evaluate_saved.py

# 测试优化版朴素贝叶斯
python run_from_stages.py Stage3_NaiveBayes_Optimization/test_optimized_nb.py
```

### 工作原理

`run_from_stages.py` 会：
1. 自动添加所有 stage 目录到 Python 路径
2. 设置正确的工作目录
3. 执行指定的脚本

### ⚠️ 注意事项

**数据文件问题**：
- stages 目录下没有 `data/` 目录
- 需要手动创建符号链接或复制数据文件

```bash
# 在 stages 目录创建数据符号链接
cd /home/u2023312337/task2/task2/stages
ln -s ../data data
ln -s ../models models
ln -s ../output output
```

**或者复制数据文件**：
```bash
cp -r ../data ./
```

## 📊 各阶段可独立运行的脚本

### Stage1 - 基础框架

```bash
# ✅ 可独立运行
python run_from_stages.py Stage1_Foundation/check_environment.py
```

### Stage2 - 传统模型

```bash
# ❌ 需要依赖 Stage1 的模块（data_loader, evaluator, visualizer）
# 必须通过 run_from_stages.py 运行
```

### Stage3 - 朴素贝叶斯优化

```bash
# ✅ 可以运行（通过 run_from_stages.py）
python run_from_stages.py Stage3_NaiveBayes_Optimization/test_optimized_nb.py
```

### Stage4 - BERT优化

```bash
# ✅ 可以运行（需要数据文件）
python run_from_stages.py Stage4_BERT_Optimization/run_bert_experiments.py
```

### Stage5 - LLM框架

```bash
# ✅ 大部分可独立运行
python run_from_stages.py Stage5_LLM_Framework/test_llm_config.py --model deepseek
python run_from_stages.py Stage5_LLM_Framework/calculate_llm_cost.py --list-prices

# 主实验脚本需要数据文件
python run_from_stages.py Stage5_LLM_Framework/run_llm_experiment.py --model deepseek
```

### Main Scripts - 主流水线

```bash
# ❌ 需要所有依赖模块 + 数据文件
# 推荐在项目根目录运行
python run_from_stages.py Main_Scripts/main_pipeline.py
```

## 🎯 最佳实践

### 查看代码演进 → 使用 stages 目录
```bash
# 查看某个阶段的 README
cat stages/Stage4_BERT_Optimization/README.md

# 对比不同版本的代码
diff stages/Stage2_Traditional_Models/naive_bayes_classifier.py \
     stages/Stage3_NaiveBayes_Optimization/naive_bayes_classifier_optimized.py
```

### 运行实验 → 使用项目根目录
```bash
cd /home/u2023312337/task2/task2
python main_pipeline.py
```

### 学习某个功能 → 查看 stages 对应目录
```bash
# 学习 BERT 优化技术
ls stages/Stage4_BERT_Optimization/
cat stages/Stage4_BERT_Optimization/README.md
```

## 📚 目录对照表

| Stages 目录 | 项目根目录文件 | 说明 |
|------------|---------------|------|
| `Stage1_Foundation/data_loader.py` | `data_loader.py` | 相同内容 |
| `Stage2_Traditional_Models/bert_classifier.py` | `bert_classifier.py` | 相同内容 |
| `Stage3_NaiveBayes_Optimization/` | `naive_bayes_classifier_optimized.py` | 相同内容 |
| `Main_Scripts/main_pipeline.py` | `main_pipeline.py` | 相同内容 |

**原则**：stages 是副本，项目根目录是运行环境。

## ❓ 常见问题

### Q1: 为什么创建 stages 目录？

**A**: 为了展示项目的演进历程，方便：
- 📖 理解代码开发过程
- 📊 查看工作量统计
- 🔍 学习优化技术
- 📝 撰写项目文档

### Q2: 我应该在哪里运行代码？

**A**: **项目根目录**（`/home/u2023312337/task2/task2/`）

### Q3: stages 目录可以删除吗？

**A**: 可以，不影响项目运行。但建议保留用于：
- 展示工作量
- 项目报告
- 代码复盘

### Q4: 如何更新 stages 目录的文件？

**A**:
```bash
# 如果根目录的文件有更新，重新复制
cp main_pipeline.py stages/Main_Scripts/
```

## 📝 总结

| 用途 | 推荐位置 | 原因 |
|------|---------|------|
| **运行实验** | 项目根目录 | 路径配置正确 |
| **查看代码** | stages 目录 | 按阶段组织清晰 |
| **学习优化** | stages 目录 | 详细 README |
| **展示工作** | stages 目录 | 体现演进过程 |

---

**建议**：日常开发和运行使用项目根目录，展示和学习使用 stages 目录。

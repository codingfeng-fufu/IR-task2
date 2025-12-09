# Task2 分阶段训练指南

## 📋 总览

本项目分为 **6个阶段**，每个阶段都有独立的训练脚本和文档。

### 阶段架构

```
baseline_simple (基础实现)
    ↓
Stage1: Foundation (基础设施测试)
    ↓
Stage2: Traditional Models (三模型baseline)
    ↓
Stage3: NaiveBayes Optimization (+5.74%)
    ↓
Stage4: BERT Optimization (+2-3%)
    ↓
Stage5: LLM Framework (Few-shot学习，4个大模型)
```

### 性能演进

| 阶段 | 朴素贝叶斯 | Word2Vec+SVM | BERT | 特点 |
|------|------------|--------------|------|------|
| baseline_simple | 73% | 82% | 87% | 最简实现 |
| Stage2 | 73% | 82% | 87% | 完整baseline |
| Stage3 | **79%** | 82% | 87% | NB优化 |
| Stage4 | 79% | 82% | **90%** | BERT优化 |
| Stage5 | - | - | - | LLM (85-92%) |

## 🚀 快速开始

### 前置要求

```bash
# 1. 激活虚拟环境
cd /home/u2023312337/task2/task2
source .venv/bin/activate

# 2. 确认数据文件存在
ls -lh data/positive.txt data/negative.txt data/testSet-1000.xlsx

# 3. GPU检查（可选，BERT训练强烈推荐）
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 一键运行所有阶段

```bash
# 运行所有阶段（需要很长时间！）
cd stages

# Baseline
cd baseline_simple && python train.py && cd ..

# Stage 1-5
cd Stage1_Foundation && python test_infrastructure.py && cd ..
cd Stage2_Traditional_Models && python train.py && cd ..
cd Stage3_NaiveBayes_Optimization && python train.py && cd ..
cd Stage4_BERT_Optimization && python train.py --model bert --quick && cd ..
cd Stage5_LLM_Framework && python train.py --all && cd ..
```

## 📂 各阶段详细指南

### Baseline Simple

**目的**: 最简单的三模型实现，快速验证想法

```bash
cd /home/u2023312337/task2/task2/stages/baseline_simple

# 训练所有模型
python train.py

# 仅训练某个模型
python train.py --model nb        # 朴素贝叶斯
python train.py --model w2v       # Word2Vec+SVM
python train.py --model bert      # BERT

# 快速测试（5000样本，BERT 1 epoch）
python train.py --quick
```

**预期时间**: 1.5小时（GPU） / 7小时（CPU）  
**输出**: `output/` 和 `models/`

---

### Stage1: Foundation

**目的**: 测试基础设施（数据加载、评估、可视化）

```bash
cd /home/u2023312337/task2/task2/stages/Stage1_Foundation

# 完整测试
python test_infrastructure.py

# 测试特定模块
python test_infrastructure.py --test data   # 数据加载
python test_infrastructure.py --test viz    # 可视化
python test_infrastructure.py --test env    # 环境检查
```

**预期时间**: 2分钟  
**输出**: `output/test_*.png`

---

### Stage2: Traditional Models

**目的**: 建立三种方法的完整baseline

```bash
cd /home/u2023312337/task2/task2/stages/Stage2_Traditional_Models

# 训练所有模型
python train.py

# 仅某个模型
python train.py --model nb
python train.py --model w2v
python train.py --model bert

# 快速测试
python train.py --quick
```

**预期时间**: 1.5小时（GPU）  
**预期性能**: NB 73%, W2V 82%, BERT 87%

---

### Stage3: NaiveBayes Optimization

**目的**: 深度优化朴素贝叶斯，从73%提升至79%

```bash
cd /home/u2023312337/task2/task2/stages/Stage3_NaiveBayes_Optimization

# 训练V2并与V1对比
python train.py

# 仅训练V2（不对比）
python train.py --no-compare

# V1 vs V2详细对比
python test_optimized_nb.py
```

**预期时间**: 5分钟  
**预期性能**: 79.20% (+5.74%)

**核心优化**:
- 多层TF-IDF (词级+字符级)
- 22维统计特征
- ComplementNB算法

---

### Stage4: BERT Optimization

**目的**: BERT高级优化，追求90%+准确率

```bash
cd /home/u2023312337/task2/task2/stages/Stage4_BERT_Optimization

# 使用统一接口
python train.py --model bert         # BERT baseline
python train.py --model scibert      # SciBERT + Focal Loss
python train.py --model deberta      # DeBERTa (最佳)

# 快速测试
python train.py --model bert --quick

# 批量实验（5组，8-12小时）
python run_bert_experiments.py

# 单模型完整训练
python train_bert_optimized_v2.py --model microsoft/deberta-v3-base
```

**预期时间**: 2-4小时（单模型） / 8-12小时（全部实验）  
**预期性能**: BERT 87% → DeBERTa 90%

**核心优化**:
- 预训练模型选择 (SciBERT, DeBERTa)
- Focal Loss
- 对抗训练 (FGM)
- 早停机制

---

### Stage5: LLM Framework

**目的**: LLM In-Context Learning实验（Few-shot学习）

```bash
cd /home/u2023312337/task2/task2/stages/Stage5_LLM_Framework

# 🆕 使用统一train.py脚本
# 交互式选择模型
python train.py

# 运行所有4个LLM
python train.py --all

# 运行单个模型
python train.py --model glm-4.6      # 智谱AI GLM-4.6
python train.py --model deepseek     # DeepSeek (性价比之王)
python train.py --model qwen3        # 阿里云通义千问
python train.py --model kimi         # Moonshot Kimi

# 快速测试（100样本）
python train.py --model deepseek --sample 100

# 使用原始脚本（高级选项）
python run_llm_experiment.py --model deepseek
python run_llm_experiment.py --all

# 成本估算
python calculate_llm_cost.py --model deepseek --samples 1000
```

**支持的模型**: GLM-4.6, Qwen3-Turbo, Kimi-K2-Turbo, DeepSeek-Chat
**预期时间**: 10-30分钟（100样本） / 2-8小时（全部976样本）
**预期性能**: 85-92%（取决于模型）

**核心特点**:
- Few-shot学习（8个示例）
- 零训练（直接推理）
- 成本追踪
- API调用管理

---

## 💡 实用技巧

### 1. 快速测试流程

如果时间有限，推荐这个快速流程：

```bash
# 20分钟完成所有阶段的快速验证
cd /home/u2023312337/task2/task2/stages

# Baseline (5000样本，BERT 1 epoch) - 15分钟
cd baseline_simple && python train.py --quick && cd ..

# Stage1 (基础设施测试) - 2分钟
cd Stage1_Foundation && python test_infrastructure.py && cd ..

# Stage3 (NB优化，10K样本) - 2分钟
cd Stage3_NaiveBayes_Optimization && python train.py --quick && cd ..

# Stage5 (LLM快速测试，100样本) - 5分钟
cd Stage5_LLM_Framework && python train.py --model deepseek --sample 100 && cd ..
```

### 2. 查看输出文件

每个阶段的输出都在各自的 `output/` 目录：

```bash
# 查看某个阶段的输出
ls -lh /home/u2023312337/task2/task2/stages/Stage3_NaiveBayes_Optimization/output/

# 查看所有阶段的模型文件大小
du -sh */models/ | sort -h
```

### 3. 对比不同阶段的性能

```bash
# 查看各阶段的评估结果
grep -h "准确率\|Accuracy" */output/*.txt
```

## ⚠️ 常见问题

### Q1: 虚拟环境激活失败

```bash
# 确保在项目根目录
cd /home/u2023312337/task2/task2
source .venv/bin/activate

# 如果还是失败，检查.venv是否存在
ls -la .venv/
```

### Q2: CUDA out of memory

```bash
# 降低批次大小
python train.py --model bert --batch-size 8  # 默认16
```

### Q3: 找不到数据文件

```bash
# 确认数据文件位置
ls -lh /home/u2023312337/task2/task2/data/

# 应该有这三个文件:
# - positive.txt (118K samples)
# - negative.txt (114K samples)
# - testSet-1000.xlsx (1000 samples)
```

### Q4: 训练时间太长

```bash
# 使用快速测试模式
python train.py --quick

# 或限制样本数
python train.py --max-samples 10000

# 或仅训练单个模型
python train.py --model nb  # 最快，2分钟
```

## 📊 性能预期总结

| 阶段 | 关键模型 | 准确率 | 训练时间 | 难度 |
|------|----------|--------|----------|------|
| baseline_simple | BERT | 87% | 1小时 | ⭐ |
| Stage1 | - | - | 2分钟 | ⭐ |
| Stage2 | 三模型 | 73-87% | 1.5小时 | ⭐⭐ |
| Stage3 | NB V2 | 79% | 5分钟 | ⭐⭐ |
| Stage4 | DeBERTa | 90% | 4小时 | ⭐⭐⭐⭐ |
| Stage5 | LLM | 85-90% | 5分钟 | ⭐⭐⭐ |

## 📚 相关文档

- 各阶段的 `IMPLEMENTATION.md` - 详细实现说明
- 各阶段的 `README.md` - 快速概述
- `../CLAUDE.md` - 项目总体文档
- `../VERSION_EVOLUTION.md` - 完整演进历史

---

**最后更新**: 2024-12-05  
**文档维护**: 与代码同步更新

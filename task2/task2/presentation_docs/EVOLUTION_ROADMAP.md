# 学术标题分类系统 - 演进路线图

## 🎯 项目目标
识别CiteSeer数据库中错误提取的学术论文标题（二分类：正确/错误）

---

## 📈 性能演进路径

```
73.46% ──────▶ 79.20% ──────▶ 82.99% ──────▶ 87.91% ──────▶ 89.04%
   │              │              │              │              │
   │              │              │              │              │
朴素贝叶斯V1   朴素贝叶斯V2   Word2Vec+SVM   BERT V1.0    SciBERT+优化
 (Nov 15)      (Nov 25)       (Nov 15)      (Nov 16)      (Nov 28)
   │              │              │              │              │
 基础实现      特征工程        词嵌入       深度学习      高级优化
                                                               │
                                                               ▼
                                                      LLM In-Context ✨
                                                          (Dec 1)
                                                          │
                                                      生成式AI
                                                      Few-shot
                                                      零训练
```

**判别式模型提升：+15.58%** (73.46% → 89.04%)
**新技术路线：生成式AI** (零训练，可解释)

---

## 🏗️ 五阶段演进

### 阶段一：基础框架（10月） 🔧
```
data_loader.py  ─┐
evaluator.py    ─┼─▶ 统一数据接口 + 评估体系
visualizer.py   ─┘
```

### 阶段二：三路并进（11月15-16日） 🚀
```
┌─────────────────┬─────────────────┬─────────────────┐
│ 朴素贝叶斯 V1.0  │ Word2Vec + SVM  │   BERT V1.0    │
│   273行代码      │    450行代码     │   513行代码     │
│   73.46%        │    82.99%       │   87.91%       │
│   传统ML        │   词嵌入        │   深度学习      │
└─────────────────┴─────────────────┴─────────────────┘
```

### 阶段三：深度优化（11月16-25日） 💡
```
朴素贝叶斯优化              BERT优化
     │                         │
     ├─ 字符级特征            ├─ 对抗训练(FGM)
     ├─ 22个统计特征          ├─ 数据增强
     ├─ ComplementNB          ├─ 多模型支持
     ├─ 参数调优              ├─ 自定义分类头
     │                         │
   79.20%                   (V2.0框架)
  (+5.74%)                  736行代码
```

### 阶段四：高级实验（11月28日） 🎓
```
train_bert_optimized_v2.py (760行)
         │
         ├─ Focal Loss
         ├─ Layer-wise LR
         ├─ 早停机制
         ├─ PGD对抗训练
         │
         ▼
run_bert_experiments.py
         │
         ├─ exp1: BERT baseline
         ├─ exp2: SciBERT + Focal ⭐
         ├─ exp3: RoBERTa
         ├─ exp4: DeBERTa
         ├─ exp5: SciBERT + Max128
         │
         ▼
    89.04% (最佳)
```

### 阶段五：生成式AI探索（12月1日） 🤖
```
LLM In-Context Learning ✨
         │
         ├─ Few-shot learning（8个示例）
         ├─ 零训练成本
         ├─ 可解释性强（给出分类理由）
         ├─ GPT-3.5/GPT-4 API
         ├─ 无需GPU
         │
         ▼
    75-85% (预期，GPT-3.5)
   零训练！可解释！新范式！
```

---

## 📊 版本对比速查表

| 版本 | 准确率 | 召回率 | F1 | 训练时间 | 代码行数 | 主要技术 |
|-----|--------|--------|-----|---------|---------|---------|
| **NB V1** | 73.46% | 84.86% | 78.82% | 2分钟 | 273 | TF-IDF + MultinomialNB |
| **NB V2** | 79.20% | 91.73% | 83.69% | 3分钟 | 399 | 多级特征 + ComplementNB |
| **W2V** | 82.99% | 85.58% | 85.74% | 10分钟 | 450 | Word2Vec + RBF SVM |
| **BERT V1** | 87.91% | 88.35% | 89.59% | 1小时 | 513 | bert-base-uncased |
| **BERT V2** | - | - | - | - | 736 | 对抗训练 + 数据增强 |
| **SciBERT** | 89.04% | 90.49% | 90.57% | 2-3小时 | 760 | Focal Loss + 层级LR |
| **LLM (GPT-3.5)** ✨ | 75-85% | 待测 | 待测 | 0分钟 | 400 | Few-shot, 零训练 |

---

## 🔬 关键技术突破

### 突破一：特征工程（NB优化）
```
5,000维 TF-IDF (1,2)-gram
           ↓
15,022维多级特征
  ├─ 10,000维 词级TF-IDF (1,3)-gram
  ├─  5,000维 字符级TF-IDF (3,5)-gram
  └─     22维 统计特征

结果: +5.74% 准确率
```

### 突破二：算法升级（NB优化）
```
MultinomialNB (alpha=1.0)
           ↓
ComplementNB (alpha=0.5)

原因: 更适合文本分类，减少类别偏差
```

### 突破三：领域适配（BERT优化）
```
bert-base-uncased (通用)
           ↓
allenai/scibert_scivocab_uncased (学术专用)

结果: +1.13% 准确率
```

### 突破四：损失函数优化
```
Cross-Entropy
     ↓
Focal Loss (alpha=0.25, gamma=2.0)

效果: 提升困难样本识别，召回率+2.14%
```

---

## 📁 文件版本地图

```
task2/
├── 基础框架 (Oct 25-27)
│   ├── data_loader.py
│   ├── evaluator.py
│   └── check_environment.py
│
├── 朴素贝叶斯 (Nov 15 → Nov 25)
│   ├── naive_bayes_classifier.py (V1: 73.46%)
│   └── naive_bayes_classifier_optimized.py (V2: 79.20%) ⭐
│
├── Word2Vec (Nov 15)
│   └── word2vec_svm_classifier.py (82.99%)
│
├── BERT演进 (Nov 15 → Nov 28)
│   ├── bert_classifier.py (V1: 87.91%)
│   ├── optimized_BERT.py (V2框架)
│   ├── bert_classifier_optimized.py (V2: 736行)
│   ├── train_optimized_bert.py (V2训练)
│   └── train_bert_optimized_v2.py (V3: 760行) ⭐
│
├── 批量实验 (Nov 28)
│   ├── run_bert_experiments.py (5组实验)
│   └── models/experiments/comparison_report.txt
│
├── LLM实验框架 (Dec 1-2) ⭐新增
│   ├── run_llm_experiment.py (主实验脚本, 714行)
│   ├── test_llm_config.py (配置测试, 245行)
│   ├── llm_config_template.json (配置模板, 279行)
│   ├── llm_in_context_classifier.py (早期版本, 478行)
│   └── llm_multi_experiment.py (早期版本, 692行)
│
└── 文档产出
    ├── 传统模型文档
    │   ├── CLAUDE.md (项目指南)
    │   ├── OPTIMIZATION_SUMMARY.md (NB优化详解)
    │   ├── BERT_OPTIMIZATION_README.md (BERT优化指南)
    │   ├── QUICK_START.md (快速上手)
    │   ├── VERSION_EVOLUTION.md (详细演进历程)
    │   ├── EVOLUTION_ROADMAP.md (本文档)
    │   ├── PERFORMANCE_COMPARISON.md (性能对比)
    │   └── README_DOCS.md (文档导航)
    │
    └── LLM实验文档 ⭐新增
        ├── LLM_CONFIG_README.md (配置说明, 436行)
        ├── LLM_EXPERIMENT_GUIDE.md (使用指南, 436行)
        ├── LLM_QUICK_START.md (快速上手, 252行)
        ├── WHERE_TO_CONFIG.txt (配置位置, ~130行)
        └── 配置位置总结.txt (配置清单, ~100行)
```

---

## 💻 代码规模统计（更新至2024-12-02）

### 传统模型代码（2024年10月-11月）
| 模块 | 文件数 | 代码行数 | 占比 |
|-----|--------|---------|------|
| 朴素贝叶斯 | 2 | 672 | 13.7% |
| Word2Vec+SVM | 1 | 450 | 9.2% |
| BERT系列 | 3 | 2,009 | 41.0% |
| 训练脚本 | 3 | 1,021 | 20.8% |
| 评估工具 | 4 | 748 | 15.3% |
| **小计** | **13** | **4,900** | **100%** |

### LLM实验框架代码（2024年12月2日）⭐新增
| 模块 | 文件数 | 代码行数 | 说明 |
|-----|--------|---------|------|
| 主实验脚本 | 1 | 714 | run_llm_experiment.py |
| 配置测试 | 1 | 245 | test_llm_config.py |
| 早期版本 | 2 | 1,170 | llm_in_context + llm_multi |
| 配置文件 | 1 | 279 | llm_config_template.json |
| **小计** | **5** | **2,408** | **配置驱动** |

### 总代码量
| 类别 | 文件数 | 代码行数 |
|-----|--------|---------|
| 传统模型 | 13 | 4,900 |
| LLM框架 | 5 | 2,408 |
| 配置文件 | 1 | 279 |
| **总计** | **19** | **~8,200** |

### 文档规模（2024年12月2日更新）

#### 传统模型文档
| 文档 | 行数 | 内容 |
|-----|------|------|
| VERSION_EVOLUTION.md | ~700 | 详细演进历程 |
| CLAUDE.md | 468 | 项目开发指南 |
| OPTIMIZATION_SUMMARY.md | 217 | NB优化总结 |
| QUICK_START.md | 223 | BERT快速上手 |
| BERT_OPTIMIZATION_README.md | ~300 | BERT优化指南 |
| EVOLUTION_ROADMAP.md | ~350 | 本文档 |
| PERFORMANCE_COMPARISON.md | 348 | 性能对比 |
| README_DOCS.md | 350 | 文档导航 |
| **小计** | **~2,950** | **9篇文档** |

#### LLM实验文档 ⭐新增
| 文档 | 行数 | 内容 |
|-----|------|------|
| LLM_CONFIG_README.md | 436 | 配置说明 |
| LLM_EXPERIMENT_GUIDE.md | 436 | 使用指南 |
| LLM_QUICK_START.md | 252 | 快速上手 |
| WHERE_TO_CONFIG.txt | ~130 | 配置位置 |
| 配置位置总结.txt | ~100 | 配置清单 |
| **小计** | **~1,350** | **5篇文档** |

#### 文档总计
| 类别 | 文档数 | 行数 |
|-----|--------|------|
| 传统模型 | 9 | ~2,950 |
| LLM实验 | 5 | ~1,350 |
| **总计** | **14** | **~4,300** |

---

## 🎯 实验对比总结

### BERT批量实验结果（Nov 28）
```
Experiment 1: BERT-base baseline
├─ 准确率: 86.68%
├─ F1: 88.22%
└─ 技术: 基础配置

Experiment 2: SciBERT + Focal ⭐ 最佳
├─ 准确率: 89.04% (+2.36%)
├─ F1: 90.57% (+2.35%)
├─ 召回率: 90.49%
└─ 技术: SciBERT + Focal Loss + Layer-wise LR + FGM

Experiment 3: RoBERTa + Weighted CE
├─ 准确率: 41.80%
└─ 状态: 训练失败 ⚠️

Experiment 5: SciBERT + Max128
├─ 准确率: 88.73%
├─ F1: 90.35%
└─ 技术: 更长序列长度
```

---

## 🏆 成果亮点

### 性能提升
✅ 准确率提升：**73.46% → 89.04% (+15.58%)**
✅ 召回率提升：**84.86% → 90.49% (+5.63%)**
✅ F1分数提升：**78.82% → 90.57% (+11.75%)**

### 技术创新
✅ 多级特征工程（15,022维）
✅ 对抗训练（FGM/PGD）
✅ Focal Loss应用
✅ 层级学习率
✅ 领域预训练模型（SciBERT）

### 工程实践
✅ 6个版本，11次实验
✅ 3种技术路线
✅ 完整的实验对比
✅ 详细的技术文档
✅ 可复现的实验流程

---

## 📚 文档导航

### 想了解具体技术细节？
- **朴素贝叶斯优化** → `OPTIMIZATION_SUMMARY.md`
- **BERT优化方法** → `BERT_OPTIMIZATION_README.md`
- **项目开发指南** → `CLAUDE.md`

### 想快速上手实验？
- **BERT快速开始** → `QUICK_START.md`
- **LLM快速开始** → `LLM_QUICK_START.md` ⭐新增
- **实验报告** → `models/experiments/comparison_report.txt`

### 想了解完整历程？
- **详细演进** → `VERSION_EVOLUTION.md`
- **演进路线** → `EVOLUTION_ROADMAP.md` (本文档)

### 想运行LLM实验？
- **配置说明** → `LLM_CONFIG_README.md` ⭐新增
- **使用指南** → `LLM_EXPERIMENT_GUIDE.md` ⭐新增
- **配置位置** → `WHERE_TO_CONFIG.txt` ⭐新增

---

## 🚀 快速运行

### 查看所有版本对比
```bash
# 朴素贝叶斯V1 vs V2
python test_optimized_nb.py

# BERT 5组实验对比
cat models/experiments/comparison_report.txt
```

### 运行最佳配置
```bash
# SciBERT + Focal Loss (89.04%)
python train_bert_optimized_v2.py
```

### 完整流水线
```bash
# 三个模型全部训练
python main_pipeline.py
```

---

## 📊 时间线总览

```
2024年10月
  ├─ 25日: 数据加载模块
  └─ 27日: 评估和可视化框架

2024年11月
  ├─ 15日: 三个基础模型实现
  │         (NB V1, Word2Vec, BERT V1)
  │
  ├─ 16日: BERT初步优化
  │         (对抗训练, 数据增强)
  │
  ├─ 25日: 朴素贝叶斯深度优化
  │         (79.20%, +5.74%)
  │
  └─ 28日: BERT高级优化实验
            (SciBERT: 89.04%, +1.13%)

2024年12月
  ├─ 01日: 生成式AI探索 + 文档整理
  │         (LLM In-Context Learning)
  │         第4种技术路线 ✨
  │
  └─ 02日: 灵活LLM实验框架
            配置驱动、支持8+模型
            完整实验管理系统
```

---

**总结**：从简单到复杂，从73.46%到89.04%，15.58%的性能提升，涵盖4种技术路线（传统ML、词嵌入、判别式深度学习、生成式AI），见证了系统性研发的全过程。

**最新进展**（12月2日）：构建了灵活的LLM实验框架，通过配置文件即可运行多个LLM模型（DeepSeek、GLM、Qwen等），无需修改代码，实现了配置驱动的实验管理系统。

---

**创建日期**：2024-12-01
**版本**：v1.0
**状态**：✅ 完成

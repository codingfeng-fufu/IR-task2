# Task2 - 学术标题分类系统

基于CiteSeer数据库的学术论文标题质量分类项目

## 📋 项目简介

本项目旨在识别学术数据库(CiteSeer)中错误提取的论文标题。通过机器学习和深度学习方法,对论文标题进行二分类:
- **正类(1)**: 正确提取的标题
- **负类(0)**: 错误提取的标题

### 核心成果

- **最高准确率**: 90.47% (Kimi-K2大语言模型,零训练)
- **最优深度学习**: 89.04% (BERT优化版)
- **最佳传统方法**: 82.99% (Word2Vec+SVM)
- **项目周期**: 2024年10月-12月 (约2.5个月)
- **代码规模**: 10,000+行代码 + 5,600+行文档

## 🎯 性能演进

项目经历了6个开发阶段,性能持续提升:

```
阶段演进:
Baseline (起点)     → Stage3 (特征工程)  → Stage4 (深度优化)  → Stage5 (LLM)
73-88%               79-83%               89.04%              90.47%
├─ NB:  73.36%      ├─ NB:  79.20%      BERT优化:            Kimi-K2:
├─ W2V: 75.61%      ├─ W2V: 82.99%      - SciBERT           - 零训练
└─ BERT:88.32%      └─ 特征工程提升      - Focal Loss        - 超越BERT!
                                         - FGM对抗            - ¥0.051/976样本
```

### 最终性能对比

| 模型 | 准确率 | F1分数 | 训练时间 | 推理速度 | 成本 |
|------|--------|--------|----------|----------|------|
| Naive Bayes优化 | 79.20% | 83.69% | 3分钟 | 极快 | 免费 |
| Word2Vec+SVM优化 | 82.99% | 85.74% | 10分钟 | 快 | 免费 |
| BERT优化 | 89.04% | 90.57% | 2.5小时 | 0.01s | 免费 |
| **Kimi-K2** | **90.47%** | **91.95%** | 零训练 | 0.77s | ¥0.051 |
| Qwen3-Max | 88.52% | 90.54% | 零训练 | 0.87s | ¥0.194 |
| DeepSeek | 88.73% | 90.23% | 零训练 | 1.25s | ¥0.133 |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (可选,用于GPU加速)
- 8GB+ RAM
- 磁盘空间: 2GB+ (不含模型和数据)

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/codingfeng-fufu/IR-task2.git
cd IR-task2

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备数据 (需要自行准备)
# 将训练数据放入 data/ 目录:
# - data/positive.txt  (正类样本)
# - data/negative.txt  (负类样本)
# - data/testSet-1000.xlsx (测试集)

# 5. 运行环境检查
python check_environment.py
```

### 快速运行

```bash
# 运行完整pipeline (训练3个模型+评估+可视化)
python main_pipeline.py

# 仅评估已保存的模型 (不重新训练)
python evaluate_saved.py

# 运行所有阶段的评估对比
python evaluate_all_stages.py
```

## 📊 项目结构

```
task2/
├── data/                          # 数据目录 (需自行准备)
│   ├── positive.txt              # 正类训练样本 (118K)
│   ├── negative.txt              # 负类训练样本 (114K)
│   └── testSet-1000.xlsx         # 测试集 (1000样本)
│
├── output/                        # 输出结果
│   ├── model_comparison.png      # 模型对比图
│   ├── confusion_matrices.png    # 混淆矩阵
│   ├── evaluation_results.txt    # 评估报告
│   └── llm_experiments/          # LLM实验结果
│
├── stages/                        # 各阶段实现
│   ├── baseline_simple/          # Baseline: 快速原型
│   ├── Stage1_Foundation/        # Stage1: 基础设施
│   ├── Stage2_Traditional_Models/ # Stage2: 传统模型
│   ├── Stage3_NaiveBayes_Optimization/ # Stage3: NB优化
│   ├── Stage4_BERT_Optimization/ # Stage4: BERT优化
│   └── Stage5_LLM_Framework/     # Stage5: LLM框架
│
├── presentation_docs/             # 演示文档包
│   ├── README.md                 # 文档导航
│   ├── QUICK_REFERENCE.md        # 一页纸总结
│   ├── PROJECT_SUMMARY.md        # 完整项目总结
│   ├── FILE_INDEX.md             # 文件索引
│   └── 01-06_STAGE_REPORT.md     # 各阶段报告
│
├── main_pipeline.py               # 主程序入口
├── data_loader.py                # 数据加载
├── evaluator.py                  # 模型评估
├── visualizer.py                 # 结果可视化
│
├── naive_bayes_classifier_optimized.py  # NB优化版
├── word2vec_svm_classifier.py    # Word2Vec+SVM
├── bert_classifier.py            # BERT分类器
│
├── run_llm_experiment.py         # LLM实验脚本
├── llm_config_template.json      # LLM配置模板
│
└── requirements.txt              # 依赖包列表
```

## 🔬 技术亮点

### 1. 特征工程突破 (Stage3)

**Naive Bayes优化**: 73% → 79% (+5.74%)
- 双层TF-IDF特征: 词级10K + 字符级5K
- 22维统计特征: 长度、标点、模式检测
- ComplementNB算法替代

**Word2Vec优化**: 74% → 83% (+8.60%)
- 100维词向量 + 8维统计特征
- 语义特征与结构特征互补

### 2. 深度学习优化 (Stage4)

**BERT**: 87% → 89% (+2.05%)
- SciBERT领域模型 (+2.36%)
- Focal Loss困难样本 (+0.62%)
- FGM对抗训练 (+0.3%)
- 序列长度优化 (+0.62%)

5组对比实验验证各技术有效性

### 3. LLM零训练框架 (Stage5)

**配置驱动架构**:
- 编辑JSON即可切换模型
- 8-shot In-Context Learning
- 支持zhipuai/openai两种provider
- **Kimi-K2超越BERT**: 90.47% > 89.04%

**两批实验迭代**:
- 第一批(12月7日): 准确率83-85%
- 第二批(12月8日): 准确率84-90%
- 提示词优化显著提升性能

## 📖 文档导航

### 快速了解 (5分钟)
- [`presentation_docs/QUICK_REFERENCE.md`](presentation_docs/QUICK_REFERENCE.md) - 一页纸项目总结

### 深度理解 (30分钟)
- [`presentation_docs/PROJECT_SUMMARY.md`](presentation_docs/PROJECT_SUMMARY.md) - 完整项目总结
- [`presentation_docs/README.md`](presentation_docs/README.md) - 文档包导航

### 技术细节 (3小时+)
- 按序阅读 [`presentation_docs/01-06_STAGE_REPORT.md`](presentation_docs/) - 6个阶段详细报告
- [`CLAUDE.md`](CLAUDE.md) - 项目使用指南
- [`VERSION_EVOLUTION.md`](VERSION_EVOLUTION.md) - 完整技术演进
- [`EVOLUTION_ROADMAP.md`](EVOLUTION_ROADMAP.md) - 可视化路线图

### LLM实验指南
- [`LLM_EXPERIMENT_GUIDE.md`](LLM_EXPERIMENT_GUIDE.md) - LLM实验完整指南
- [`LLM_COST_SUMMARY.md`](LLM_COST_SUMMARY.md) - 成本分析
- [`llm_config_template.json`](llm_config_template.json) - 配置模板

## 🎓 使用场景

### 场景1: 资源受限环境
推荐: **Naive Bayes优化版**
- 准确率: 79.20%
- 训练时间: 3分钟
- 无需GPU,内存占用小

### 场景2: 生产部署
推荐: **BERT优化版**
- 准确率: 89.04%
- 推理速度: 0.01秒/样本
- 稳定可靠,免费部署

### 场景3: 追求极致性能
推荐: **Kimi-K2 (LLM)**
- 准确率: 90.47% (最高)
- 零训练即用
- 成本: ¥0.051/976样本

### 场景4: 成本敏感
推荐: **DeepSeek (LLM)**
- 准确率: 88.73%
- 成本: ¥0.133/976样本
- 性价比之王

## 🔧 三种模型使用方法

### 1. Naive Bayes优化版

```python
from naive_bayes_classifier_optimized import NaiveBayesClassifierOptimized
from data_loader import DataLoader

# 加载数据
train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
    'data/positive.txt',
    'data/negative.txt',
    'data/testSet-1000.xlsx'
)

# 训练模型
classifier = NaiveBayesClassifierOptimized()
classifier.train(train_titles, train_labels)

# 预测
predictions = classifier.predict(test_titles)

# 保存模型
classifier.save_model()
```

### 2. BERT优化版

```python
from bert_classifier import BertClassifier

# 训练
classifier = BertClassifier()
classifier.train(
    train_titles,
    train_labels,
    epochs=5,
    batch_size=32,
    learning_rate=2e-5
)

# 预测
predictions = classifier.predict(test_titles)

# 保存模型
classifier.save_model()
```

### 3. LLM实验 (需要API密钥)

```bash
# 1. 配置API密钥
cp llm_config_template.json llm_config.json
vim llm_config.json  # 填入API密钥

# 2. 运行实验
python run_llm_experiment.py --model kimi

# 3. 查看结果
cat output/llm_experiments/kimi_report.txt
```

## 📈 工作量统计

- **开发时间**: 296小时 (37个工作日)
- **代码规模**: 10,438行Python代码
- **文档规模**: 5,600+行Markdown文档
- **GPU训练**: 约50小时
- **LLM实验**: 约¥20 (两批实验)

## 🏆 关键发现

✅ **传统方法潜力巨大**: 特征工程使NB提升5.74%, W2V提升8.60%

✅ **领域模型效果最佳**: SciBERT > RoBERTa > BERT-base

✅ **组合优化最有效**: 单一技术提升有限,组合应用效果最佳

✅ **LLM达到SOTA**: 零训练即可超越微调后的BERT

✅ **成本极低**: Kimi仅需¥0.051/976样本,DeepSeek仅¥0.133

## 💡 技术路线建议

```
快速验证   → LLM (Kimi, 90%, ¥0.05, 即时可用)
追求性能   → LLM (Kimi, 90%) 或 BERT优化 (89%)
生产部署   → BERT优化 (89%, 稳定, 免费)
资源受限   → NB优化 (79%, 3分钟训练)
成本敏感   → DeepSeek (89%, ¥0.13)
```

## 📊 数据集说明

**训练集**: 232,402个样本
- 正类 (正确标题): 118,239个
- 负类 (错误标题): 114,163个

**测试集**: 976个样本 (1000个样本中有24个重复)
- 来源: CiteSeer数据库
- 格式: Excel文件 (testSet-1000.xlsx)
- 列名: "title given by manchine", "Y/N"

**注意**: 数据文件较大(~15MB),未包含在仓库中,需自行准备

## 🤝 贡献者

本项目由个人独立完成,包括:
- 完整的6阶段开发
- 10,000+行代码实现
- 5,600+行文档编写
- 所有实验设计与执行

## 📄 许可证

MIT License

## 📮 联系方式

如有问题或建议,欢迎通过以下方式联系:
- GitHub Issues: [提交Issue](https://github.com/codingfeng-fufu/IR-task2/issues)
- Email: [您的邮箱]

## 🌟 致谢

感谢以下开源项目和工具:
- PyTorch & Hugging Face Transformers
- scikit-learn & gensim
- CiteSeer数据库
- 各大语言模型API提供商 (智谱AI, 通义千问, Moonshot, DeepSeek)

---

**项目完成时间**: 2024年12月

**最终成果**: 90.47%准确率 (Kimi-K2, 零训练)

**核心突破**: LLM超越BERT, 配置驱动架构, 特征工程显著提升

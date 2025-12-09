# 项目文件分阶段组织

本目录按照开发阶段组织所有项目文件。

## 📁 目录结构

```
stages/
├── Stage1_Foundation/              阶段1：基础框架搭建（2024年10月）
├── Stage2_Traditional_Models/      阶段2：传统模型实现（2024年11月15日）
├── Stage3_NaiveBayes_Optimization/ 阶段3：朴素贝叶斯优化（2024年11月25日）
├── Stage4_BERT_Optimization/       阶段4：BERT优化实验（2024年11月28日）
├── Stage5_LLM_Framework/           阶段5：LLM实验框架（2024年12月1-2日）
├── Main_Scripts/                   主要运行脚本
└── Utils/                          工具脚本
```

## 🔍 各阶段说明

### Stage1_Foundation - 基础框架搭建
**时间**：2024年10月25-27日
**目标**：建立数据处理、评估和可视化基础设施
**文件**：
- data_loader.py - 数据加载模块
- evaluator.py - 模型评估模块
- visualizer.py - 结果可视化模块
- check_environment.py - 环境检查工具

### Stage2_Traditional_Models - 传统模型实现
**时间**：2024年11月15日
**目标**：实现三种基础分类器
**文件**：
- naive_bayes_classifier.py - 朴素贝叶斯V1（73.46%）
- word2vec_svm_classifier.py - Word2Vec+SVM（82.99%）
- bert_classifier.py - BERT基础版（87.91%）

### Stage3_NaiveBayes_Optimization - 朴素贝叶斯优化
**时间**：2024年11月25日
**目标**：深度优化朴素贝叶斯分类器
**文件**：
- naive_bayes_classifier_optimized.py - 优化版（79.20%，+5.74%）
- test_optimized_nb.py - V1与V2对比测试

### Stage4_BERT_Optimization - BERT优化实验
**时间**：2024年11月16-28日
**目标**：BERT高级优化，探索SciBERT、Focal Loss、对抗训练等
**文件**：
- train_optimized_bert.py - BERT训练V1
- bert_classifier_optimized.py - BERT优化类（V2）
- optimized_BERT.py - BERT优化框架
- train_bert_optimized_v2.py - BERT训练V2（最终版）
- run_bert_experiments.py - 批量实验（5组）
- predownload_models.py - 模型预下载工具
- run_quick.sh - 快速实验脚本

### Stage5_LLM_Framework - LLM实验框架
**时间**：2024年12月1-2日
**目标**：构建灵活的LLM In-Context Learning实验系统
**文件**：
- llm_in_context_classifier.py - LLM分类器（早期版本）
- llm_multi_experiment.py - 多模型对比（早期版本）
- run_llm_experiment.py - 主实验脚本（配置驱动）
- test_llm_config.py - 配置测试工具
- llm_config_template.json - 配置模板
- calculate_llm_cost.py - 成本估算工具
- test_llm_classifier.py - LLM分类器测试
- llm_cost_estimation.json - 成本估算数据
- install_llm_dependencies.sh - 依赖安装脚本

### Main_Scripts - 主要运行脚本
**用途**：整合所有模型的主流水线和评估脚本
**文件**：
- main_pipeline.py - 完整流水线（三种模型）
- evaluate_saved.py - 已保存模型评估
- run_optimized_classifier.py - 运行优化分类器

### Utils - 工具脚本
**用途**：辅助工具和修复脚本
**文件**：
- fix_evaluator.py - 评估器修复工具

## 🚀 如何使用

### 方案1：在项目根目录运行（推荐）

```bash
# 进入项目根目录
cd /home/u2023312337/task2/task2

# 运行完整流水线
python main_pipeline.py

# 运行LLM实验
python run_llm_experiment.py --model deepseek
```

### 方案2：在 stages 目录运行（实验性）

**首次使用需要配置环境**：
```bash
cd /home/u2023312337/task2/task2/stages

# 创建数据符号链接（仅需一次）
ln -s ../data data
ln -s ../models models
ln -s ../output output

# 使用辅助脚本运行
python run_from_stages.py Stage1_Foundation/check_environment.py
python run_from_stages.py Stage3_NaiveBayes_Optimization/test_optimized_nb.py
python run_from_stages.py Stage5_LLM_Framework/calculate_llm_cost.py --list-prices
```

**详细说明**：参见 `RUN_GUIDE.md` 和 `TEST_RESULTS.md`

### 查看某个阶段的代码
```bash
# 查看基础框架
ls -lh Stage1_Foundation/

# 查看BERT优化
ls -lh Stage4_BERT_Optimization/

# 阅读某个阶段的说明
cat Stage4_BERT_Optimization/README.md
```

## 📊 各阶段代码量

| 阶段 | 文件数 | 代码行数 | 说明 |
|------|--------|---------|------|
| Stage1 | 4 | ~800 | 基础设施 |
| Stage2 | 3 | ~1,400 | 传统模型 |
| Stage3 | 2 | ~700 | NB优化 |
| Stage4 | 7 | ~2,800 | BERT优化 |
| Stage5 | 9 | ~2,400 | LLM框架 |
| Main | 3 | ~600 | 主脚本 |
| Utils | 1 | ~50 | 工具 |
| **总计** | **29** | **~8,750** | - |

## 🔗 相关文档

- **VERSION_EVOLUTION.md** - 完整的版本演进历程
- **EVOLUTION_ROADMAP.md** - 演进路线图
- **README_DOCS.md** - 文档导航

---

**说明**：原始文件仍保留在项目根目录，此目录为按阶段组织的副本。

# 学术标题分类系统运行指南

## 1. 项目定位与主要脚本
- `main_pipeline.py`：默认入口，串联数据加载、三种模型训练（朴素贝叶斯、Word2Vec+SVM、BERT）、评估与可视化。
- `run_optimized_classifier.py`：对不同的 Transformer 变体（如 SciBERT、RoBERTa）进行对比实验，可在 `--mode full/quick` 间切换。
- `train_optimized_bert.py`：启用 FGM、EMA 等技巧的单模型强化训练脚本。
- `evaluate_saved.py`：跳过训练，仅加载 `models/` 下的已保存模型进行评估与可视化。
- `check_environment.py`：快速校验 Python 版本、依赖包、CUDA/GPU 以及数据与输出目录。

## 2. 环境准备
1. **Python 版本**：>= 3.8（建议 3.10/3.11）。
2. **进入项目目录**：
   ```bash
   cd path/to/IR-task2/task2/task2
   ```
3. **创建虚拟环境并激活**：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
   python -m pip install -U pip
   ```
4. **安装依赖**（可根据自身 GPU 版本调整 PyTorch 源）：
   ```bash
   # 依据显卡选择合适的 PyTorch，以下示例为 CUDA 12.1
   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

   pip install transformers==4.36.2 scikit-learn==1.3.2 gensim==4.3.2 pandas==2.1.4 \
       matplotlib==3.8.2 seaborn==0.13.1 tqdm==4.66.1 openpyxl==3.1.2 numpy==1.24.4
   ```
   若仅在 CPU 上训练，可改用 `pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu`。
5. **验证环境**：
   ```bash
   python check_environment.py
   ```
   所有检查均通过后再启动训练，可及早发现缺失数据或 GPU 不可用等问题。

## 3. 数据准备
```
project_root/
├─ data/
│  ├─ positive.txt   # 正样本：每行一个正确的学术标题
│  ├─ negative.txt   # 负样本：每行一个错误/噪声标题
│  └─ testSet-1000.xlsx  # 测试集（须包含 `title given by manchine`, `Y/N` 两列）
```
- 所有文本文件需为 UTF-8 编码，脚本会自动做小写化与符号清洗。
- Excel 中 `Y/N` 列用 `Y` 表示正确、`N` 表示错误。
- 若数据文件缺失，脚本会回退到内置的示例数据，方便快速体验但不作为正式结果。

## 4. 运行方式
### 4.1 完整多模型流水线
```bash
python main_pipeline.py
```
默认流程：加载真实数据（若存在）→ 训练三种模型 → 使用 `output/` 存放评估报表、预测结果与可视化图。若需要仅训练 BERT 或采样更小的数据集，可直接在脚本顶部修改以下标志：
- `USE_SAMPLE_DATA`：切换到示例数据。
- `MAX_TRAIN_SAMPLES`：限定最大训练样本数。
- `TRAIN_ONLY_BERT`：只运行 BERT 分支。
- `BERT_EPOCHS`：BERT 训练轮数。

### 4.2 评估已有模型（无需重新训练）
```bash
python evaluate_saved.py
```
要求 `models/` 下存在对应权重文件，例如：
- `best_bert_model.pt`
- `word2vec_svm_model_w2v.model` / `_svm.pkl`
- `naive_bayes_model.pkl`

脚本会加载模型 → 在测试集上评估 → 重新生成 `output/` 内的可视化与报告。

### 4.3 优化版 BERT 对比实验
```bash
# 完整实验：对 SciBERT、RoBERTa 等多模型依次训练+比较
python run_optimized_classifier.py --mode full

# 快速冒烟测试：使用更小子集与 3~5 轮训练
python run_optimized_classifier.py --mode quick
```
可在脚本内的 `models_to_test` 列表中按需增删 Transformer 变体，并通过 `freeze_layers`、`use_custom_head` 等参数做消融实验。可视化结果默认写入 `outputs/`。

### 4.4 单模型强化训练（FGM/EMA）
```bash
python train_optimized_bert.py
```
此脚本专注于 `BERTClassifierOptimized`，启用 FGM、EMA、数据增强等策略，适用于追求单模型最佳效果的场景。训练结束会在 `output/` 中保存：
- `bert_performance.png`, `bert_confusion_matrix.png`, `bert_tsne_visualization.png`
- `bert_evaluation_results.txt`, `bert_results.json`

## 5. 输出与模型文件
- `output/`：`main_pipeline.py`、`train_optimized_bert.py`、`evaluate_saved.py` 的报告、预测与可视化默认目录。
- `outputs/`：`run_optimized_classifier.py` 产生的实验图表。
- `models/`：保存/加载各类模型权重与中间文件。
  - 运行脚本会自动写入，如无写权限需提前创建目录。

## 6. 常见问题与提示
- **HuggingFace 模型下载**：首次运行 BERT 相关脚本时需联网以下载 `bert-base-uncased`、`scibert` 等权重，可预先配置 `TRANSFORMERS_CACHE` 以复用缓存。
- **GPU 资源**：没有 GPU 也可运行，但 BERT 训练时间会显著增加；若仅体验流程，可打开 `USE_SAMPLE_DATA` 并减少 `BERT_EPOCHS`。
- **随机性**：脚本内部已设置常见随机种子，但若需要可在相应文件中手动固定 `random.seed`/`numpy.random.seed`/`torch.manual_seed`。
- **日志与排错**：大多数脚本在运行时会输出详细进度；若中途报错，可参考堆栈信息并检查数据路径、依赖版本、显存占用等。

按照以上步骤准备环境与数据后，即可顺利复现项目的训练、评估与可视化流程。祝实验顺利！

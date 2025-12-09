# Stage5: LLM实验框架

**时间**：2024年12月1-2日
**目标**：构建灵活的LLM In-Context Learning实验系统

## 🆕 统一训练脚本 (NEW!)

现在可以使用统一的 `train.py` 脚本运行所有4个LLM，就像其他stages一样！

### 快速开始

```bash
# 方式1：交互式选择模型
python train.py

# 方式2：运行所有4个模型（推荐）
python train.py --all

# 方式3：运行指定模型
python train.py --model glm-4.6    # 智谱AI GLM-4.6
python train.py --model deepseek   # DeepSeek (性价比之王)
python train.py --model qwen3      # 阿里云通义千问
python train.py --model kimi       # Moonshot Kimi

# 方式4：快速测试（100个样本）
python train.py --model deepseek --sample 100

# 方式5：使用自定义配置
python train.py --config custom.json --all
```

### 支持的模型

已配置4个中国大模型（API密钥已设置）：
1. **GLM-4.6** (智谱AI) - `glm-4.6`
2. **Qwen3-Turbo** (阿里云通义千问) - `qwen3`
3. **Kimi-K2-Turbo** (Moonshot AI) - `kimi`
4. **DeepSeek-Chat** (DeepSeek) - `deepseek`

### 脚本输出

- 详细结果：`output/llm_experiments/{model}_*.json`
- 文本报告：`output/llm_experiments/{model}_*_report.txt`
- 性能对比图：`output/llm_experiments/llm_comparison.png`（多模型时）
- 混淆矩阵：`output/llm_experiments/llm_confusion_matrices.png`（多模型时）
- 模型对比表格（终端显示）

**注意**：LLM 模型通过 API 调用，无法获取中间特征向量，因此不生成 t-SNE 可视化（与 Stage1-4 不同）。

## 📁 文件列表

| 文件 | 行数 | 功能 | 特点 |
|------|------|------|------|
| `train.py` | 287 | 🆕 统一训练脚本 | 支持所有4个LLM |
| `run_llm_experiment.py` | 714 | 主实验脚本（配置驱动） | 无需修改代码 |
| `test_llm_config.py` | 245 | 配置测试工具 | API连通性测试 |
| `llm_config_template.json` | 279 | 配置模板 | 8+模型示例 |
| `llm_config.json` | 53 | 实际配置（已设置） | 4个中国大模型 |
| `calculate_llm_cost.py` | 197 | 成本估算工具 | 价格计算 |
| `test_llm_classifier.py` | 187 | LLM分类器测试 | 早期版本 |
| `llm_cost_estimation.json` | 142 | 成本估算数据 | 价格表 |
| `WHERE_TO_CONFIG.txt` | 130 | 配置位置可视化 | ASCII导航 |
| `LLM_CONFIG_README.md` | 436 | 完整配置文档 | 详细说明 |
| `LLM_QUICK_START.md` | 252 | 3分钟快速上手 | 新手指南 |
| `install_llm_dependencies.sh` | 82 | 依赖安装脚本 | 一键安装 |

## 🎯 核心特性

### 1. 配置驱动架构 ⭐

**零代码修改**：所有模型配置通过JSON文件管理

```json
{
  "llms": {
    "my-model": {
      "provider": "openai",           // API类型
      "model": "gpt-3.5-turbo",      // 模型名称
      "api_key": "sk-xxxx",          // 🔑 API密钥
      "base_url": "https://...",     // API端点
      "temperature": 0.0,             // 生成温度
      "max_tokens": 150,              // 最大输出
      "few_shot_examples": 8,         // Few-shot数量
      "enabled": true                 // ✅ 启用/禁用
    }
  }
}
```

**支持的LLM Provider**：
- `openai` - OpenAI兼容API（GPT、DeepSeek、GLM、Qwen、Kimi）
- `anthropic` - Claude系列

### 2. 灵活的实验控制

**三种运行模式**：
```bash
# 1. 单模型实验
python run_llm_experiment.py --model deepseek

# 2. 多模型对比
python run_llm_experiment.py --all

# 3. 指定样本数
python run_llm_experiment.py --model glm-4 --num_samples 100
```

**自动断点续传**：
- 每100个样本自动保存checkpoint
- 异常中断后可恢复
- 支持手动停止（Ctrl+C）

### 3. In-Context Learning（Few-shot）

**默认配置**：8个示例（4正4负）

**Prompt结构**：
```
Task: 判断学术论文标题是否正确提取...

Examples:
1. Title: "Neural Networks for NLP"
   Label: 正确 (1)

2. Title: "pp. 123-145 References"
   Label: 错误 (0)

...

Now classify:
Title: "{test_title}"
Label:
```

**可自定义**：
- Few-shot数量（0-20个）
- Prompt模板
- 示例选择策略

### 4. 成本估算与追踪

**实时统计**：
- 输入/输出token数
- API调用次数
- 预估成本（人民币）
- 平均延迟

**示例输出**：
```
DeepSeek-V3成本估算：
  输入token: 1,234,567 (~¥1.85)
  输出token: 98,765 (~¥1.58)
  总计: ~¥3.43

平均延迟: 1.23s/样本
```

### 5. 详细的结果保存

**输出目录**：`output/llm_experiments/{model_name}_{timestamp}/`

**保存内容**：
- `config.json` - 实验配置
- `predictions.json` - 所有预测结果
- `results.txt` - 详细评估指标
- `checkpoint_*.json` - 断点文件
- `error_analysis.txt` - 错误分析

## 🚀 使用示例

### 快速开始（3分钟）

**步骤1**：复制配置模板
```bash
cp llm_config_template.json llm_config.json
```

**步骤2**：编辑配置文件
```bash
vim llm_config.json
# 修改 api_key 字段（搜索 YOUR_API_KEY_HERE）
# 设置 enabled: true
```

**步骤3**：运行实验
```bash
python run_llm_experiment.py --model deepseek
```

### 添加新模型（30秒）

在 `llm_config.json` 中添加：
```json
{
  "llms": {
    "my-custom-model": {
      "provider": "openai",
      "model": "your-model-name",
      "api_key": "your-key-here",
      "base_url": "https://api.your-provider.com",
      "temperature": 0.0,
      "max_tokens": 150,
      "enabled": true
    }
  }
}
```

运行：
```bash
python run_llm_experiment.py --model my-custom-model
```

### 测试配置

**测试API连通性**：
```bash
python test_llm_config.py --model deepseek
```

**验证配置文件**：
```bash
python test_llm_config.py --validate
```

### 成本估算

**实验前估算**：
```bash
python calculate_llm_cost.py \
  --model deepseek-chat \
  --num_samples 976 \
  --title_length 80 \
  --examples 8
```

**查看价格表**：
```bash
python calculate_llm_cost.py --list-prices
```

## 📊 支持的LLM模型

### 推荐模型（按性价比排序）

| 模型 | 供应商 | 输入价格 | 输出价格 | 976样本成本 | 推荐度 |
|------|--------|---------|---------|-------------|--------|
| **DeepSeek-V3** | DeepSeek | ¥0.25/M | ¥1/M | ~¥0.30 | ⭐⭐⭐⭐⭐ |
| **GLM-4-Flash** | 智谱AI | ¥0.1/M | ¥0.1/M | ~¥0.12 | ⭐⭐⭐⭐⭐ |
| **Qwen-Turbo** | 阿里云 | ¥0.3/M | ¥0.6/M | ~¥0.50 | ⭐⭐⭐⭐ |
| **Kimi** | 月之暗面 | ¥0.1/M | ¥0.1/M | ~¥0.12 | ⭐⭐⭐⭐ |
| **GPT-4o-mini** | OpenAI | $0.15/M | $0.60/M | ~$0.39 | ⭐⭐⭐ |

### 配置模板提供的模型

1. **DeepSeek** (`deepseek-chat`) - 性价比之王
2. **GLM-4** (`glm-4-flash`) - 快速响应
3. **Qwen** (`qwen-turbo`) - 中文优化
4. **Kimi** (`moonshot-v1-8k`) - 性价比高
5. **GPT-3.5-Turbo** (OpenAI) - 经典选择
6. **GPT-4o-mini** (OpenAI) - 小型GPT-4
7. **GPT-4o** (OpenAI) - 高性能
8. **Claude-3.5-Sonnet** (Anthropic) - 顶级模型

## 💡 设计理念

### 为什么使用配置驱动？

**传统方式（需修改代码）**：
```python
# ❌ 每次换模型都要改代码
model = "deepseek-chat"
api_key = "sk-xxx"
base_url = "https://..."
```

**配置驱动方式（无需修改代码）**：
```bash
# ✅ 只需编辑JSON文件
vim llm_config.json
python run_llm_experiment.py --model deepseek
```

**优势**：
- 🔧 零代码修改
- 📦 配置可版本管理
- 🔄 快速切换模型
- 🧪 批量对比实验
- 📝 配置复用

### In-Context Learning vs Fine-tuning

| 特性 | In-Context | Fine-tuning |
|------|-----------|-------------|
| 训练成本 | ✅ 零训练 | ❌ GPU+时间 |
| 数据需求 | ✅ Few-shot | ❌ 大量标注 |
| 部署成本 | ✅ API调用 | ❌ 模型托管 |
| 灵活性 | ✅ 即改即用 | ❌ 需重训练 |
| 性能上限 | ⚠️ 依赖模型 | ✅ 可优化 |

**本框架适用场景**：
- 快速原型验证
- 小样本学习任务
- 成本敏感项目
- 多模型对比实验

## 🔬 技术细节

### API调用流程

```python
# 1. 加载配置
config = load_config("llm_config.json")

# 2. 初始化分类器
classifier = LLMClassifier(config["llms"]["deepseek"])

# 3. 准备Few-shot示例
classifier.prepare_few_shot_examples(train_titles, train_labels)

# 4. 批量预测
predictions = classifier.predict_batch(test_titles)

# 5. 保存结果
save_results(predictions, config)
```

### Prompt工程

**Prompt组成**：
1. **任务描述**（Task Definition）
2. **Few-shot示例**（8个，4正4负）
3. **输出格式**（Output Format）
4. **测试样本**（Test Input）

**输出解析**：
- 支持多种回答格式：`0`, `1`, `错误`, `正确`, `Correct`, `Incorrect`
- 正则表达式匹配
- 异常处理和默认值

### 错误处理

**重试机制**：
- API调用失败自动重试（最多3次）
- 指数退避（exponential backoff）
- 超时处理

**Checkpoint系统**：
- 每100个样本自动保存
- 包含：已预测样本、统计信息、配置
- 恢复时自动跳过已完成样本

## 📚 相关文档

### 快速上手
- **LLM_QUICK_START.md** - 3分钟快速开始
- **WHERE_TO_CONFIG.txt** - 配置位置可视化

### 详细文档
- **LLM_CONFIG_README.md** - 完整配置说明（436行）
- **llm_config_template.json** - 配置模板（带详细注释）

### 工具
- **test_llm_config.py** - 配置测试
- **calculate_llm_cost.py** - 成本估算
- **install_llm_dependencies.sh** - 依赖安装

## 📈 代码统计

- **总行数**：~3,200行
- **核心脚本**：714行（run_llm_experiment.py）
- **文档**：~1,700行（5个MD文档）
- **配置**：421行（JSON + 脚本）

## 🔗 后续扩展

### 已实现
✅ 配置驱动架构
✅ 多模型支持
✅ Few-shot Learning
✅ 成本追踪
✅ 断点续传
✅ 详细日志

### 可扩展方向
🔲 动态Few-shot选择（相似度匹配）
🔲 Ensemble投票（多模型融合）
🔲 Chain-of-Thought（思维链）
🔲 自动Prompt优化
🔲 流式输出支持
🔲 本地模型接入（vLLM、Ollama）

---

**总结**：通过配置驱动设计，实现了灵活的LLM实验框架，支持零训练快速验证学术标题分类任务。

# LLM分类实验使用指南

## 快速开始

### 1. 配置模型

复制配置模板并填写API密钥：

```bash
cp llm_config_template.json llm_config.json
```

编辑 `llm_config.json`，填写你的API密钥：

```json
{
  "llms": {
    "deepseek": {
      "provider": "openai",
      "model": "deepseek-chat",
      "api_key": "sk-your-actual-api-key-here",  // 替换为真实API密钥
      "base_url": "https://api.deepseek.com",
      "temperature": 0.0,
      "max_tokens": 150,
      "enabled": true,  // 设置为true启用该模型
      "comment": "DeepSeek Chat"
    }
  }
}
```

### 2. 运行实验

**交互式选择模型：**
```bash
python run_llm_experiment.py
```

**直接指定模型：**
```bash
# 运行DeepSeek
python run_llm_experiment.py --model deepseek

# 运行智谱GLM
python run_llm_experiment.py --model glm-4.6

# 运行通义千问
python run_llm_experiment.py --model qwen3
```

**运行所有启用的模型：**
```bash
python run_llm_experiment.py --all
```

**指定测试样本数：**
```bash
# 使用100个样本快速测试
python run_llm_experiment.py --model deepseek --sample 100

# 使用全部976个样本
python run_llm_experiment.py --model deepseek --sample 976
```

**自定义输出目录：**
```bash
python run_llm_experiment.py --model deepseek --output my_results/
```

## 配置说明

### 模型配置参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `provider` | API提供商 | `"openai"`, `"anthropic"` |
| `model` | 模型名称 | `"deepseek-chat"`, `"glm-4-plus"` |
| `api_key` | API密钥 | `"sk-xxxx"` |
| `base_url` | API端点 | `"https://api.deepseek.com"` |
| `temperature` | 生成温度 | `0.0`（确定性）~`1.0`（随机性） |
| `max_tokens` | 最大输出token | `150` |
| `few_shot_examples` | Few-shot示例数 | `8` |
| `enabled` | 是否启用 | `true` / `false` |

### 实验配置参数

在 `llm_config.json` 中的 `experiment` 部分：

```json
"experiment": {
  "few_shot_examples": 8,          // Few-shot示例数量
  "sample_size": 976,               // 测试样本数（默认使用全部）
  "delay_between_calls": 0.5,      // API调用间隔（秒），避免限流
  "output_dir": "output/llm_experiments"  // 输出目录
}
```

## 添加新模型

### 方法1：添加OpenAI兼容模型

大部分国内大模型都支持OpenAI格式API，只需修改配置：

```json
"your-model-name": {
  "provider": "openai",
  "model": "actual-model-name",
  "api_key": "YOUR_API_KEY",
  "base_url": "https://api.your-provider.com/v1",
  "temperature": 0.0,
  "max_tokens": 150,
  "few_shot_examples": 8,
  "enabled": true,
  "comment": "你的模型描述"
}
```

### 方法2：添加Anthropic模型

```json
"claude": {
  "provider": "anthropic",
  "model": "claude-3-opus-20240229",
  "api_key": "YOUR_ANTHROPIC_API_KEY",
  "temperature": 0.0,
  "max_tokens": 150,
  "few_shot_examples": 8,
  "enabled": true,
  "comment": "Claude 3 Opus"
}
```

## 支持的模型

### 国内大模型（推荐）

| 模型 | 配置名称 | API文档 | 备注 |
|------|----------|---------|------|
| 智谱GLM-4 | `glm-4.6` | [文档](https://open.bigmodel.cn) | 性能优秀 |
| 通义千问 | `qwen3` | [文档](https://help.aliyun.com/zh/dashscope) | 阿里云提供 |
| Kimi | `kimi` | [文档](https://platform.moonshot.cn) | 长文本能力强 |
| DeepSeek | `deepseek` | [文档](https://platform.deepseek.com) | 性价比高 |

### 国际模型

| 模型 | 配置名称 | API文档 | 备注 |
|------|----------|---------|------|
| GPT-3.5 | `gpt-3.5` | [文档](https://platform.openai.com) | 需要国际网络 |
| GPT-4 | `gpt-4` | [文档](https://platform.openai.com) | 需要国际网络 |
| Claude 3 | `claude-3` | [文档](https://docs.anthropic.com) | 需要国际网络 |

## 输出文件

运行实验后，会在输出目录生成以下文件：

```
output/llm_experiments/
├── deepseek_20241202_153045.json          # JSON格式详细结果
├── deepseek_20241202_153045_report.txt    # 文本格式报告
└── checkpoints/                            # 检查点文件（每100个样本）
    └── deepseek_checkpoint_100.json
```

### JSON结果文件

包含完整的实验数据：

```json
{
  "model": "deepseek",
  "timestamp": "20241202_153045",
  "eval_metrics": {
    "accuracy": 0.85,
    "precision": 0.87,
    "recall": 0.83,
    "f1": 0.85,
    "f1_macro": 0.84,
    "f1_micro": 0.85
  },
  "stats": {
    "total_calls": 976,
    "total_tokens": 245000,
    "total_input_tokens": 195000,
    "total_output_tokens": 50000,
    "errors": 2,
    "total_time": 488.5
  },
  "predictions": [1, 0, 1, ...],
  "details": [
    {
      "index": 0,
      "title": "...",
      "pred_label": 1,
      "response": "✓ 正确标题",
      "tokens": 251,
      "time": 0.5
    }
  ]
}
```

### 文本报告文件

易读的实验报告：

```
================================================================================
deepseek 实验报告
================================================================================

实验时间: 2024-12-02 15:30:45
模型: deepseek

================================================================================
性能指标
================================================================================
准确率 (Accuracy):  85.00%
精确率 (Precision): 87.00%
召回率 (Recall):    83.00%
F1分数 (F1):        85.00%
...
```

## 调优参数

### Temperature（生成温度）

- **0.0**：完全确定性，每次相同输入得到相同输出（推荐用于分类任务）
- **0.3-0.7**：轻微随机性
- **1.0**：高随机性

```json
"temperature": 0.0  // 分类任务推荐使用0.0
```

### Few-shot示例数

控制提供给模型的参考示例数量：

```json
"few_shot_examples": 8  // 可选：0, 4, 8, 16
```

- **0-shot**：不提供示例，纯依靠指令
- **4-shot**：提供4个示例
- **8-shot**：提供8个示例（默认，推荐）
- **16-shot**：提供16个示例（可能提高准确率，但增加成本）

### API调用延迟

避免触发限流：

```json
"delay_between_calls": 0.5  // 单位：秒
```

- 免费账户：建议1.0秒以上
- 付费账户：可以设置0.5秒或更低

## 成本估算

脚本会自动统计token消耗，参考 `cost_estimation` 配置估算成本：

```bash
# 示例：DeepSeek运行976个样本
# 输入tokens: 195,000 × 1元/百万 = 0.195元
# 输出tokens: 50,000 × 2元/百万 = 0.10元
# 总成本: ~0.30元
```

### 性价比对比（976样本）

| 模型 | 输入定价 | 输出定价 | 预估总成本 |
|------|----------|----------|------------|
| DeepSeek | 1元/M | 2元/M | ~0.30元 |
| Kimi | 0.1元/M | 0.1元/M | ~0.03元 |
| Qwen Turbo | 0.3元/M | 0.6元/M | ~0.09元 |
| GLM-4 Plus | 50元/M | 50元/M | ~12元 |
| GPT-3.5 | $0.5/M | $1.5/M | ~$0.15 |
| GPT-4 | $30/M | $60/M | ~$9 |

注：实际成本会因prompt长度和模型输出长度有所不同。

## 故障排除

### 1. API密钥无效

```
❌ 客户端初始化失败: Invalid API key
```

**解决方案**：
- 检查API密钥是否正确
- 确认API密钥有足够余额
- 检查base_url是否正确

### 2. 限流错误

```
❌ API调用失败: Rate limit exceeded
```

**解决方案**：
- 增加 `delay_between_calls` 参数
- 升级API账户
- 使用更少样本测试（`--sample 100`）

### 3. 网络超时

```
❌ API调用失败: Connection timeout
```

**解决方案**：
- 检查网络连接
- 确认API端点可访问
- 使用国内模型替代国际模型

### 4. 模型响应解析失败

```
⚠️  无法明确解析响应，默认为0
```

**解决方案**：
- 检查模型是否遵循指令格式
- 调整prompt（修改 `_create_prompt` 方法）
- 增加temperature让模型输出更规范

## 高级用法

### 断点续传

脚本自动每100个样本保存一个检查点：

```
checkpoints/deepseek_checkpoint_100.json
checkpoints/deepseek_checkpoint_200.json
...
```

如果实验中断，可以手动从检查点恢复。

### 自定义Prompt

编辑 `run_llm_experiment.py` 中的 `_create_prompt` 方法：

```python
def _create_prompt(self, title: str) -> str:
    """创建分类Prompt"""
    prompt = """你的自定义指令...

【分类标准】
...
"""
    return prompt
```

### 批量对比实验

运行多个模型并生成对比报告：

```bash
# 1. 启用所有想对比的模型
vim llm_config.json  # 设置 enabled: true

# 2. 运行所有模型
python run_llm_experiment.py --all

# 3. 查看对比结果
cat output/llm_experiments/*_report.txt
```

## 性能基准

基于CiteSeer学术标题分类任务（976个测试样本）：

| 模型 | 准确率 | F1分数 | 平均耗时 | Token消耗 | 预估成本 |
|------|--------|--------|----------|-----------|----------|
| **传统模型（参考）** |
| Naive Bayes | 79.20% | 83.69% | <0.01s | - | - |
| BERT | 87.91% | 89.59% | ~0.5s | - | - |
| **LLM模型（8-shot）** |
| DeepSeek | ~85% | ~85% | 0.5s | 245K | ¥0.30 |
| Qwen Turbo | ~84% | ~84% | 0.6s | 250K | ¥0.09 |
| GLM-4 Plus | ~86% | ~86% | 0.7s | 240K | ¥12 |
| GPT-3.5 | ~83% | ~83% | 0.8s | 260K | $0.15 |

注：LLM模型性能可能因prompt设计和few-shot示例而变化。

## 相关文件

- `run_llm_experiment.py` - 主实验脚本
- `llm_config.json` - 配置文件（需自行创建）
- `llm_config_template.json` - 配置模板
- `llm_in_context_classifier.py` - 旧版分类器（单模型）
- `llm_multi_experiment.py` - 旧版多模型实验脚本
- `calculate_llm_cost.py` - 成本估算工具

## 常见问题

**Q: 如何选择Few-shot示例数？**

A: 推荐从8个开始，如果效果不理想可以尝试增加到16个。更多示例通常能提高准确率，但会增加成本。

**Q: 哪个模型性价比最高？**

A: DeepSeek和Kimi是国内模型中性价比较高的选择，价格低且性能不错。

**Q: 可以同时运行多个模型吗？**

A: 可以使用 `--all` 参数，但建议先用小样本测试（`--sample 100`）确认配置正确。

**Q: 如何提高准确率？**

A:
1. 增加few-shot示例数
2. 优化prompt设计
3. 使用更强的模型（如GLM-4 Plus, GPT-4）
4. 调整temperature参数

**Q: 实验中断了怎么办？**

A: 检查点文件保存在 `checkpoints/` 目录，可以查看已完成的进度，手动恢复需要修改代码实现。

## 技术支持

如有问题，请查看：
- 各模型的官方API文档
- `llm_classifier_design.md` - 设计文档
- 代码注释

---

**版本**: 1.0
**最后更新**: 2024-12-02

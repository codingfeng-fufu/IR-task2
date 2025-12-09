# Stage5_LLM_Framework 实现文档

## 📋 阶段概述

**实现时间**: 2024年12月1-2日  
**主要目标**: 构建灵活的LLM In-Context Learning实验框架  
**代码行数**: ~2,400行（9个文件）  
**实验方法**: Zero-shot, Few-shot ICL（无需训练）

## 🎯 核心特性

### LLM In-Context Learning

**与传统方法的区别**:
- ❌ 不需要训练/微调
- ✅ 仅通过prompt设计
- ✅ 支持多种LLM API
- ✅ 灵活的成本控制

### 支持的LLM

| 模型 | API | 成本（1K tokens） | 特点 |
|------|-----|-------------------|------|
| DeepSeek | deepseek.com | ¥0.001 | 极低成本 ⭐ |
| GPT-3.5 | openai.com | $0.002 | 平衡性价比 |
| GPT-4 | openai.com | $0.03 | 最高质量 |
| Claude | anthropic.com | $0.015 | 长上下文 |
| 通义千问 | dashscope | ¥0.002 | 国产 |

## 📁 文件结构

```
Stage5_LLM_Framework/
├── llm_in_context_classifier.py    # LLM分类器实现
├── run_llm_experiment.py           # 主实验脚本（配置驱动）
├── llm_multi_experiment.py         # 多模型对比
├── calculate_llm_cost.py           # 成本估算工具
├── llm_config_template.json        # 配置模板
├── llm_cost_estimation.json        # 成本估算数据
├── test_llm_classifier.py          # 测试脚本
├── test_llm_config.py              # 配置测试
├── train.py                        # 新增：统一接口
├── config.py
├── models/                         # （无模型文件，因为无训练）
└── output/llm_experiments/         # 实验结果
    ├── deepseek_results.json
    ├── comparison_report.txt
    └── cost_analysis.txt
```

## 🚀 快速开始

### 1. 成本估算（必做！）

```bash
# 估算成本
python calculate_llm_cost.py --model deepseek --samples 100

# 输出示例:
# DeepSeek (100 samples):
#   预计tokens: 50,000
#   预计成本: ¥0.05
#   预计时间: 2分钟
```

### 2. 运行实验

```bash
# 使用DeepSeek（最便宜）
python run_llm_experiment.py --model deepseek --max-samples 100

# 使用配置文件
python run_llm_experiment.py --config llm_config.json

# 使用train.py统一接口
python train.py --model deepseek --samples 100
```

### 3. Prompt设计

**Zero-shot示例**:
```
你是一个学术论文标题分类专家。

任务：判断以下标题是否是正确提取的学术论文标题。

标题：{title}

请回答：正确 或 错误
```

**Few-shot示例（3-shot）**:
```
你是一个学术论文标题分类专家。以下是一些示例：

示例1：
标题：Deep Learning for Natural Language Processing
判断：正确

示例2：
标题：pp. 123-145
判断：错误

示例3：
标题：Proceedings of the 2024 Conference
判断：错误

现在判断：
标题：{title}
请回答：正确 或 错误
```

## 📊 预期性能与成本

### 性能对比

| 方法 | 准确率 | 成本（1000样本） | 时间 |
|------|--------|------------------|------|
| BERT训练 | 87.9% | ¥0（GPU） | 1小时 |
| DeBERTa训练 | 90.1% | ¥0（GPU） | 4小时 |
| **DeepSeek Few-shot** | 85-88% | ¥0.5 | 5分钟 ⭐ |
| GPT-3.5 Few-shot | 88-90% | $2.0 | 3分钟 |
| GPT-4 Few-shot | 90-92% | $30 | 10分钟 |

### 成本分析

```
1000样本完整测试:
├─ DeepSeek: ¥0.50 （最经济）
├─ GPT-3.5: $2.00
├─ GPT-4: $30.00
└─ 自训BERT: ¥0 （但需GPU和时间）

推荐策略:
- 快速验证: DeepSeek
- 生产环境: 自训练模型
- 最高质量: GPT-4（小规模）
```

## 💡 使用场景

**适合用LLM的情况**:
- ✅ 快速���型和验证
- ✅ 数据量很小（< 1000）
- ✅ 没有GPU资源
- ✅ 需要快速迭代prompt

**不适合用LLM的情况**:
- ❌ 大规模生产（成本高）
- ❌ 实时响应要求（延迟高）
- ❌ 离线环境
- ❌ 对成本敏感

## ⚠️ 注意事项

1. **API密钥**: 需要设置环境变量（DEEPSEEK_API_KEY等）
2. **成本控制**: 始终先用`calculate_llm_cost.py`估算
3. **速率限制**: API有请求频率限制，大规模实验需时间
4. **不稳定性**: LLM输出有随机性，需要设置seed
5. **隐私**: 不要发送敏感数据到商业API

## 📚 配置文件格式

```json
{
  "model": "deepseek",
  "api_key_env": "DEEPSEEK_API_KEY",
  "max_samples": 100,
  "few_shot_examples": 3,
  "temperature": 0.1,
  "prompt_template": "..."
}
```

---

**实现完成度**: ✅ 100%  
**成本效益**: 🎯 DeepSeek最优（¥0.50/1K样本）  
**相关文档**: LLM_EXPERIMENT_GUIDE.md, LLM_COST_SUMMARY.md

#!/bin/bash
# 快速测试脚本 - 使用少量样本测试所有模型

echo "========================================"
echo "快速测试 - 每个模型测试10个样本"
echo "========================================"
echo ""

# 测试GLM-4.6
echo "1. 测试 GLM-4.6 (10样本)..."
python run_llm_experiment.py --model glm-4.6 --sample 10
echo ""

# 测试Qwen3-Max
echo "2. 测试 Qwen3-Max (10样本)..."
python run_llm_experiment.py --model qwen3 --sample 10
echo ""

# 测试Kimi
echo "3. 测试 Kimi (10样本)..."
python run_llm_experiment.py --model kimi --sample 10
echo ""

# 测试DeepSeek
echo "4. 测试 DeepSeek (10样本)..."
python run_llm_experiment.py --model deepseek --sample 10
echo ""

echo "========================================"
echo "快速测试完成!"
echo "========================================"

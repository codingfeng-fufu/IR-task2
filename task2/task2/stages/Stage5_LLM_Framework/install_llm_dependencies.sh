#!/bin/bash

# LLM依赖安装脚本（国内大模型版本）
# 使用方法: bash install_llm_dependencies.sh

echo "========================================"
echo "  国内大模型LLM实验 - 依赖安装脚本"
echo "========================================"
echo ""

# 检查Python版本
echo "[1/3] 检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 安装核心依赖
echo ""
echo "[2/3] 安装核心依赖..."
pip install --upgrade pip

# 基础科学计算库
echo "  - 安装基础库..."
pip install numpy pandas tqdm openpyxl

# OpenAI SDK（兼容所有国内大模型）
echo "  - 安装OpenAI SDK（支持GLM-4, Qwen, Kimi, DeepSeek）..."
pip install "openai>=1.0.0"

# 可选：其他工具
echo "  - 安装其他工具..."
pip install python-dotenv  # 支持.env文件

echo ""
echo "[3/3] 验证安装..."

# 验证安装
python3 << EOF
import sys
try:
    from openai import OpenAI
    print("✓ OpenAI SDK 安装成功（支持GLM-4, Qwen, Kimi, DeepSeek）")
except ImportError:
    print("✗ OpenAI SDK 安装失败")
    sys.exit(1)

try:
    import pandas
    import numpy
    import tqdm
    import openpyxl
    print("✓ 数据处理库安装成功")
except ImportError:
    print("✗ 数据处理库安装失败")
    sys.exit(1)

print("")
print("✓ 所有依赖安装成功！")
EOF

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "1. 获取API密钥（新用户有免费额度）:"
echo "   - 智谱AI: https://open.bigmodel.cn/ (18元)"
echo "   - 阿里云: https://dashscope.console.aliyun.com/ (100万tokens)"
echo "   - Kimi: https://platform.moonshot.cn/ (15元)"
echo "   - DeepSeek: https://platform.deepseek.com/ (10元)"
echo ""
echo "2. 编辑 llm_config.json，填写API密钥"
echo "3. 运行: python llm_multi_experiment.py"
echo ""
echo "详细文档: 查看 LLM_CHINESE_GUIDE.md"
echo ""

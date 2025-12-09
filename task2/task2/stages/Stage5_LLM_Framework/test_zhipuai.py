#!/usr/bin/env python3
"""
测试zhipuai库的调用
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Stage1_Foundation'))

def test_zhipuai_import():
    """测试zhipuai库是否安装"""
    try:
        from zhipuai import ZhipuAI
        print("✓ zhipuai库已安装")
        return True
    except ImportError:
        print("✗ zhipuai库未安装")
        print("请运行: pip install zhipuai")
        return False

def test_glm_connection():
    """测试GLM-4.6连接"""
    try:
        from zhipuai import ZhipuAI

        # 从配置文件读取API密钥
        import json
        with open("llm_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        glm_config = config["llms"]["glm-4.6"]
        api_key = glm_config["api_key"]

        print(f"\n测试GLM-4.6连接...")
        print(f"模型: {glm_config['model']}")
        print(f"Provider: {glm_config['provider']}")
        print(f"Temperature: {glm_config['temperature']}")
        print(f"Max tokens: {glm_config['max_tokens']}")
        print(f"Thinking mode: {glm_config['thinking_mode']}")

        # 初始化客户端
        client = ZhipuAI(api_key=api_key)

        # 简单测试调用
        response = client.chat.completions.create(
            model=glm_config["model"],
            messages=[
                {"role": "user", "content": "你好，请回复'测试成功'"}
            ],
            temperature=glm_config["temperature"],
            max_tokens=50,
            thinking={"type": glm_config["thinking_mode"]}
        )

        result = response.choices[0].message.content
        print(f"\n✓ GLM-4.6连接成功!")
        print(f"响应: {result}")
        print(f"使用tokens: {response.usage.total_tokens}")

        return True

    except Exception as e:
        print(f"\n✗ GLM-4.6连接失败: {str(e)}")
        return False

def test_qwen_connection():
    """测试Qwen3-Max连接"""
    try:
        from openai import OpenAI

        # 从配置文件读取API密钥
        import json
        with open("llm_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        qwen_config = config["llms"]["qwen3"]

        print(f"\n测试Qwen3-Max连接...")
        print(f"模型: {qwen_config['model']}")
        print(f"Provider: {qwen_config['provider']}")

        # 初始化客户端
        client = OpenAI(
            api_key=qwen_config["api_key"],
            base_url=qwen_config["base_url"]
        )

        # 简单测试调用
        response = client.chat.completions.create(
            model=qwen_config["model"],
            messages=[
                {"role": "user", "content": "你好，请回复'测试成功'"}
            ],
            temperature=qwen_config["temperature"],
            max_tokens=50
        )

        result = response.choices[0].message.content
        print(f"\n✓ Qwen3-Max连接成功!")
        print(f"响应: {result}")
        print(f"使用tokens: {response.usage.total_tokens}")

        return True

    except Exception as e:
        print(f"\n✗ Qwen3-Max连接失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("LLM配置测试")
    print("=" * 80)

    # 检查zhipuai库
    if not test_zhipuai_import():
        print("\n请先安装zhipuai库:")
        print("pip install zhipuai")
        sys.exit(1)

    # 测试GLM-4.6
    glm_ok = test_glm_connection()

    # 测试Qwen3-Max
    qwen_ok = test_qwen_connection()

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"GLM-4.6:   {'✓ 成功' if glm_ok else '✗ 失败'}")
    print(f"Qwen3-Max: {'✓ 成功' if qwen_ok else '✗ 失败'}")
    print("=" * 80)

    if glm_ok and qwen_ok:
        print("\n✓ 所有模型连接正常，可以开始实验!")
    else:
        print("\n⚠ 部分模型连接失败，请检查配置")

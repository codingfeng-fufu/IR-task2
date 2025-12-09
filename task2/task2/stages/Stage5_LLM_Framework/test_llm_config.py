"""
test_llm_config.py
==================
快速测试LLM配置是否正确

使用方法:
python test_llm_config.py
python test_llm_config.py --model deepseek
"""

import os
import sys
import json
import argparse
from typing import Dict, List

def load_config(config_file: str = "llm_config.json") -> Dict:
    """加载配置文件"""
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print(f"\n请先创建配置文件:")
        print(f"  cp llm_config_template.json llm_config.json")
        print(f"  vim llm_config.json  # 填写API密钥")
        return None

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def test_single_model(model_name: str, model_config: Dict) -> bool:
    """测试单个模型配置"""
    print(f"\n{'='*80}")
    print(f"测试模型: {model_name}")
    print(f"{'='*80}")

    # 1. 检查必需字段
    required_fields = ["provider", "model", "api_key"]
    missing_fields = [field for field in required_fields if field not in model_config]

    if missing_fields:
        print(f"❌ 缺少必需字段: {', '.join(missing_fields)}")
        return False

    # 2. 检查API密钥
    api_key = model_config.get("api_key", "")
    if not api_key or "YOUR_" in api_key or len(api_key) < 10:
        print(f"❌ API密钥无效或未设置")
        print(f"   当前值: {api_key[:20]}..." if len(api_key) > 20 else f"   当前值: {api_key}")
        return False

    print(f"✓ API密钥格式正确")

    # 3. 显示配置信息
    print(f"\n配置信息:")
    print(f"  Provider: {model_config['provider']}")
    print(f"  Model: {model_config['model']}")
    print(f"  Base URL: {model_config.get('base_url', 'default')}")
    print(f"  Temperature: {model_config.get('temperature', 0.0)}")
    print(f"  Max Tokens: {model_config.get('max_tokens', 150)}")
    print(f"  Few-shot Examples: {model_config.get('few_shot_examples', 8)}")

    # 4. 测试API连接
    print(f"\n测试API连接...")
    try:
        if model_config["provider"] == "openai":
            success = test_openai_api(model_config)
        elif model_config["provider"] == "anthropic":
            success = test_anthropic_api(model_config)
        else:
            print(f"⚠️  未知的provider: {model_config['provider']}")
            return False

        if success:
            print(f"✓ API连接测试成功！")
            return True
        else:
            print(f"❌ API连接测试失败")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_api(config: Dict) -> bool:
    """测试OpenAI兼容API"""
    try:
        from openai import OpenAI

        # 初始化客户端
        client_params = {"api_key": config["api_key"]}
        if config.get("base_url"):
            client_params["base_url"] = config["base_url"]

        client = OpenAI(**client_params)

        # 发送测试请求
        test_prompt = "请回答：1+1等于几？只回答数字。"
        print(f"  发送测试请求: {test_prompt}")

        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )

        answer = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0

        print(f"  收到响应: {answer}")
        print(f"  Token消耗: {tokens}")

        return True

    except ImportError:
        print(f"❌ 未安装openai库: pip install openai>=1.0.0")
        return False
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return False


def test_anthropic_api(config: Dict) -> bool:
    """测试Anthropic API"""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config["api_key"])

        # 发送测试请求
        test_prompt = "请回答：1+1等于几？只回答数字。"
        print(f"  发送测试请求: {test_prompt}")

        message = client.messages.create(
            model=config["model"],
            max_tokens=10,
            messages=[
                {"role": "user", "content": test_prompt}
            ]
        )

        answer = message.content[0].text
        tokens = message.usage.input_tokens + message.usage.output_tokens

        print(f"  收到响应: {answer}")
        print(f"  Token消耗: {tokens}")

        return True

    except ImportError:
        print(f"❌ 未安装anthropic库: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试LLM配置")
    parser.add_argument("--model", type=str, help="指定要测试的模型")
    parser.add_argument("--config", type=str, default="llm_config.json", help="配置文件路径")
    args = parser.parse_args()

    print("=" * 80)
    print(" " * 28 + "LLM配置测试工具")
    print("=" * 80)

    # 加载配置
    config = load_config(args.config)
    if not config:
        sys.exit(1)

    print(f"\n✓ 配置文件加载成功: {args.config}")

    # 获取所有启用的模型
    enabled_models = {
        name: cfg for name, cfg in config["llms"].items()
        if cfg.get("enabled", True)
    }

    if not enabled_models:
        print(f"\n❌ 没有启用的模型")
        print(f"请在配置文件中设置 'enabled': true")
        sys.exit(1)

    print(f"✓ 发现 {len(enabled_models)} 个启用的模型: {', '.join(enabled_models.keys())}")

    # 选择要测试的模型
    if args.model:
        if args.model not in config["llms"]:
            print(f"\n❌ 模型不存在: {args.model}")
            print(f"可用模型: {', '.join(config['llms'].keys())}")
            sys.exit(1)

        models_to_test = {args.model: config["llms"][args.model]}
    else:
        # 测试所有启用的模型
        models_to_test = enabled_models

    # 运行测试
    results = {}
    for model_name, model_config in models_to_test.items():
        success = test_single_model(model_name, model_config)
        results[model_name] = success

    # 总结
    print(f"\n{'='*80}")
    print("测试总结")
    print(f"{'='*80}")

    success_count = sum(results.values())
    total_count = len(results)

    print(f"\n{'模型':<25} {'状态':<10}")
    print("-" * 35)

    for model_name, success in results.items():
        status = "✓ 成功" if success else "❌ 失败"
        print(f"{model_name:<25} {status:<10}")

    print(f"\n总计: {success_count}/{total_count} 测试通过")

    if success_count == total_count:
        print(f"\n✓ 所有测试通过！可以开始运行实验了。")
        print(f"\n运行实验:")
        if len(results) == 1:
            print(f"  python run_llm_experiment.py --model {list(results.keys())[0]}")
        else:
            print(f"  python run_llm_experiment.py --all")
    else:
        print(f"\n⚠️  有 {total_count - success_count} 个模型测试失败")
        print(f"请检查API密钥和网络连接")

    print("=" * 80)


if __name__ == "__main__":
    main()

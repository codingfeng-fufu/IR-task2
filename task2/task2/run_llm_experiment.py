"""
run_llm_experiment.py
=====================
灵活的LLM分类实验脚本
通过修改配置文件即可切换模型，无需修改代码

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 配置说明：所有模型配置都在外部JSON文件中
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ 配置文件位置：
   - 文件名：llm_config.json（需自行创建）
   - 模板：llm_config_template.json
   - 位置：与本脚本同目录

2️⃣ 如何添加/修改模型：
   步骤1：打开 llm_config.json
   步骤2：在 "llms" 部分添加或修改模型配置
   步骤3：填写 api_key、model 等参数
   步骤4：设置 "enabled": true 启用模型

3️⃣ 配置示例：
   {
     "llms": {
       "my-model": {                          // 模型名称（自定义）
         "provider": "openai",                // API类型（openai/anthropic）
         "model": "gpt-3.5-turbo",           // 实际模型名
         "api_key": "sk-xxxx",               // 🔑 API密钥（必填）
         "base_url": "https://api.xxx.com",  // API端点（可选）
         "temperature": 0.0,                  // 生成温度
         "max_tokens": 150,                   // 最大输出token
         "enabled": true                      // ✅ 是否启用
       }
     }
   }

4️⃣ 不需要修改本Python文件中的任何代码！
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

使用方法:
1. 编辑 llm_config.json，配置想要使用的模型
2. 运行: python run_llm_experiment.py
3. 或者指定模型: python run_llm_experiment.py --model glm-4.6
4. 或者运行所有启用的模型: python run_llm_experiment.py --all

特性:
- 支持单模型/多模型对比实验
- 灵活的配置系统
- 自动保存详细结果
- 成本估算
- 错误处理和断点续传
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from evaluator import ModelEvaluator


class LLMClassifier:
    """统一的LLM分类器，支持多种provider"""

    def __init__(self, config: Dict, model_name: str = "unnamed"):
        """
        初始化LLM分类器

        Args:
            config: 模型配置字典
            model_name: 模型显示名称
        """
        self.config = config
        self.model_name = model_name
        self.provider = config["provider"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 150)
        self.api_key = config["api_key"]
        self.base_url = config.get("base_url", None)

        # Few-shot配置
        self.few_shot_count = config.get("few_shot_examples", 8)
        self.examples = []

        # 统计信息
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "errors": 0,
            "total_time": 0.0,
            "failed_indices": []
        }

        # 初始化客户端
        self._init_client()

        print(f"✓ {self.model_name} 初始化成功")
        print(f"  模型: {self.model}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")

    def _init_client(self):
        """初始化对应的API客户端"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                client_params = {"api_key": self.api_key}
                if self.base_url:
                    client_params["base_url"] = self.base_url
                self.client = OpenAI(**client_params)
            except ImportError:
                raise ImportError("请安装openai: pip install openai>=1.0.0")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("请安装anthropic: pip install anthropic")
        else:
            raise ValueError(f"不支持的provider: {self.provider}")

    def set_few_shot_examples(self, examples: List[Dict]):
        """设置Few-shot示例"""
        self.examples = examples[:self.few_shot_count]
        print(f"✓ 已加载 {len(self.examples)} 个Few-shot示例")

    def _create_prompt(self, title: str) -> str:
        """创建分类Prompt"""
        prompt = """你是一个专业的学术论文标题质量评估专家。你的任务是判断给定的标题是否为**正确提取**的学术论文标题。

【分类标准】
✓ 正确标题（1）：
  - 完整、清晰的学术论文标题
  - 语法正确，表达准确
  - 不包含页码、摘要、章节编号等非标题内容

✗ 错误标题（0）：
  - 包含页码（如"pp. 123-145"）
  - 包含摘要片段（如"Abstract: ..."）
  - 包含章节标记（如"1. Introduction"）
  - 包含期刊信息（如"Vol. 25, No. 3"）
  - 包含格式错误（如"......"连续点号）
  - 包含"Reference"、"Appendix"等关键词

【参考示例】

"""
        # 添加Few-shot示例
        for i, ex in enumerate(self.examples, 1):
            label_symbol = "✓" if ex["label"] == 1 else "✗"
            label_text = "正确标题" if ex["label"] == 1 else "错误标题"
            prompt += f"示例 {i}:\n"
            prompt += f"标题：「{ex['title']}」\n"
            prompt += f"判断：{label_symbol} {label_text}\n"
            if "reason" in ex:
                prompt += f"理由：{ex['reason']}\n"
            prompt += "\n"

        # 待分类标题
        prompt += "=" * 60 + "\n"
        prompt += "【现在请判断以下标题】\n\n"
        prompt += f"标题：「{title}」\n"
        prompt += f"判断：\n\n"
        prompt += "请只回答\"✓ 正确标题\"或\"✗ 错误标题\"。"

        return prompt

    def _parse_response(self, response: str) -> int:
        """解析LLM响应为标签"""
        response_lower = response.lower()

        if "✓" in response or "正确标题" in response or "correct" in response_lower:
            return 1
        elif "✗" in response or "错误标题" in response or "incorrect" in response_lower:
            return 0
        else:
            # 尝试从数字解析
            if "1" in response[:10]:
                return 1
            elif "0" in response[:10]:
                return 0
            else:
                return 0  # 默认保守策略

    def _call_api(self, prompt: str) -> Dict:
        """调用API"""
        start_time = time.time()

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个准确、专业的学术标题分类专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                return {
                    "response": response.choices[0].message.content.strip(),
                    "tokens": response.usage.total_tokens if response.usage else 0,
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "time": time.time() - start_time
                }

            elif self.provider == "anthropic":
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                return {
                    "response": message.content[0].text,
                    "tokens": message.usage.input_tokens + message.usage.output_tokens,
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                    "time": time.time() - start_time
                }

        except Exception as e:
            return {
                "response": "",
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "time": time.time() - start_time,
                "error": str(e)
            }

    def predict(
        self,
        titles: List[str],
        delay: float = 0.5,
        verbose: bool = False,
        save_checkpoints: bool = True,
        checkpoint_interval: int = 100
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        批量预测

        Args:
            titles: 标题列表
            delay: API调用间隔（秒）
            verbose: 是否打印详细信息
            save_checkpoints: 是否保存检查点
            checkpoint_interval: 检查点间隔

        Returns:
            (predictions, details)
        """
        print(f"\n{'='*80}")
        print(f"开始预测: {self.model_name}")
        print(f"{'='*80}")
        print(f"样本数: {len(titles)}")
        print(f"Few-shot示例数: {len(self.examples)}")
        print(f"Temperature: {self.temperature}")
        print(f"{'='*80}\n")

        predictions = []
        details = []

        for i, title in enumerate(tqdm(titles, desc=f"{self.model_name} 预测进度")):
            # 创建prompt
            prompt = self._create_prompt(title)

            # 调用API
            result = self._call_api(prompt)

            # 解析响应
            if "error" not in result:
                label = self._parse_response(result["response"])
                self.stats["total_calls"] += 1
                self.stats["total_tokens"] += result["tokens"]
                self.stats["total_input_tokens"] += result.get("input_tokens", 0)
                self.stats["total_output_tokens"] += result.get("output_tokens", 0)
                self.stats["total_time"] += result["time"]
            else:
                label = 0  # 错误时默认为0
                self.stats["errors"] += 1
                self.stats["failed_indices"].append(i)
                if verbose:
                    print(f"\n⚠️  第 {i+1} 个样本调用失败: {result['error']}")

            predictions.append(label)

            # 保存详细信息
            details.append({
                "index": i,
                "title": title,
                "pred_label": label,
                "response": result["response"],
                "tokens": result.get("tokens", 0),
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "time": result.get("time", 0.0),
                "error": result.get("error", None)
            })

            # 保存检查点
            if save_checkpoints and (i + 1) % checkpoint_interval == 0:
                checkpoint_file = f"checkpoints/{self.model_name}_checkpoint_{i+1}.json"
                os.makedirs("checkpoints", exist_ok=True)
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "predictions": predictions,
                        "details": details,
                        "stats": self.stats
                    }, f, ensure_ascii=False, indent=2)
                print(f"\n✓ 检查点已保存: {checkpoint_file}")

            # API限流延迟
            if i < len(titles) - 1:
                time.sleep(delay)

        # 打印统计信息
        print(f"\n{'='*80}")
        print("预测完成统计")
        print(f"{'='*80}")
        print(f"总调用次数: {self.stats['total_calls']}")
        print(f"成功次数: {self.stats['total_calls'] - self.stats['errors']}")
        print(f"失败次数: {self.stats['errors']}")
        print(f"总Token消耗: {self.stats['total_tokens']}")
        print(f"  输入Token: {self.stats['total_input_tokens']}")
        print(f"  输出Token: {self.stats['total_output_tokens']}")
        print(f"总时间: {self.stats['total_time']:.2f}秒")
        print(f"平均时间: {self.stats['total_time']/self.stats['total_calls']:.2f}秒/样本" if self.stats['total_calls'] > 0 else "N/A")
        print(f"预测为正类: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
        print(f"预测为负类: {len(predictions)-sum(predictions)} ({(len(predictions)-sum(predictions))/len(predictions)*100:.1f}%)")
        print(f"{'='*80}\n")

        return np.array(predictions), details


def get_default_few_shot_examples() -> List[Dict]:
    """
    获取默认的Few-shot示例

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📚 Few-shot示例配置位置（如需修改示例内容）
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    如果需要修改Few-shot示例的内容（正负例样本），请修改下方的列表。

    示例数量在配置文件中控制：
        "experiment": {
            "few_shot_examples": 8  // 使用前8个示例
        }

    示例格式：
        {
            "title": "标题文本",
            "label": 1,  // 1=正确, 0=错误
            "reason": "判断理由"
        }

    ⚠️ 注意：一般情况下不需要修改此处，使用默认示例即可
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    return [
        {
            "title": "Deep Learning for Natural Language Processing: A Survey",
            "label": 1,
            "reason": "完整规范的学术论文标题"
        },
        {
            "title": "Machine Translation Using Neural Networks",
            "label": 1,
            "reason": "清晰准确的研究标题"
        },
        {
            "title": "A Comparative Study of Sentiment Analysis Methods",
            "label": 1,
            "reason": "标准的学术论文标题格式"
        },
        {
            "title": "pp. 123-145 Introduction to Machine Learning",
            "label": 0,
            "reason": "包含页码信息，非标准标题"
        },
        {
            "title": "Abstract: This paper presents a new method",
            "label": 0,
            "reason": "包含'Abstract'，是摘要片段"
        },
        {
            "title": "A Novel Approach to Deep Learning......",
            "label": 0,
            "reason": "包含连续点号，疑似提取错误"
        },
        {
            "title": "Vol. 25, No. 3, 2024 - Neural Networks",
            "label": 0,
            "reason": "包含期刊卷号信息"
        },
        {
            "title": "1. Introduction Recent advances in deep learning",
            "label": 0,
            "reason": "包含章节编号"
        },
    ]


def load_config(config_file: str = "llm_config.json") -> Dict:
    """
    加载配置文件

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🔧 配置文件加载位置（重要！）
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    默认配置文件：llm_config.json

    如果要使用不同的配置文件，可以通过命令行参数指定：
        python run_llm_experiment.py --config my_config.json

    配置文件结构：
        {
            "llms": {
                "模型名": {
                    "provider": "openai",
                    "model": "模型ID",
                    "api_key": "你的密钥",
                    "enabled": true
                }
            },
            "experiment": {
                "few_shot_examples": 8,
                "sample_size": 976,
                "delay_between_calls": 0.5
            }
        }
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def validate_api_keys(config: Dict) -> List[str]:
    """验证API密钥，返回可用的模型列表"""
    available_models = []

    for name, llm_config in config["llms"].items():
        if not llm_config.get("enabled", True):
            continue

        api_key = llm_config.get("api_key", "")
        if api_key and "YOUR_" not in api_key and len(api_key) > 10:
            available_models.append(name)

    return available_models


def run_single_experiment(
    model_name: str,
    config: Dict,
    test_titles: List[str],
    test_labels: List[int],
    sample_size: Optional[int] = None
) -> Dict:
    """
    运行单个模型的实验

    Returns:
        {
            "predictions": np.ndarray,
            "eval_result": dict,
            "details": list,
            "stats": dict
        }
    """
    # 获取模型配置
    llm_config = config["llms"][model_name]

    # 采样
    if sample_size and sample_size < len(test_titles):
        test_titles = test_titles[:sample_size]
        test_labels = test_labels[:sample_size]

    # 初始化分类器
    classifier = LLMClassifier(llm_config, model_name=model_name)

    # 设置Few-shot示例
    few_shot_count = config["experiment"].get("few_shot_examples", 8)
    examples = get_default_few_shot_examples()[:few_shot_count]
    classifier.set_few_shot_examples(examples)

    # 预测
    delay = config["experiment"].get("delay_between_calls", 0.5)
    predictions, details = classifier.predict(
        test_titles,
        delay=delay,
        save_checkpoints=True,
        checkpoint_interval=100
    )

    # 评估
    evaluator = ModelEvaluator()
    eval_result = evaluator.evaluate_model(
        test_labels,
        predictions,
        model_name=model_name,
        verbose=True
    )

    return {
        "predictions": predictions,
        "eval_result": eval_result,
        "details": details,
        "stats": classifier.stats
    }


def save_experiment_results(
    results: Dict,
    model_name: str,
    output_dir: str = "output/llm_experiments"
):
    """保存实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # 1. 保存JSON格式详细结果
    json_file = os.path.join(output_dir, f"{model_name}_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": model_name,
            "timestamp": timestamp,
            "eval_metrics": convert_to_serializable(results["eval_result"]),
            "stats": results["stats"],
            "predictions": results["predictions"].tolist(),
            "details": convert_to_serializable(results["details"])
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 详细结果已保存: {json_file}")

    # 2. 保存文本格式报告
    report_file = os.path.join(output_dir, f"{model_name}_{timestamp}_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_name} 实验报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {model_name}\n\n")

        f.write("=" * 80 + "\n")
        f.write("性能指标\n")
        f.write("=" * 80 + "\n")
        metrics = results["eval_result"]
        f.write(f"准确率 (Accuracy):  {metrics['accuracy']*100:.2f}%\n")
        f.write(f"精确率 (Precision): {metrics['precision']*100:.2f}%\n")
        f.write(f"召回率 (Recall):    {metrics['recall']*100:.2f}%\n")
        f.write(f"F1分数 (F1):        {metrics['f1']*100:.2f}%\n")
        f.write(f"F1宏平均:           {metrics['f1_macro']*100:.2f}%\n")
        f.write(f"F1微平均:           {metrics['f1_micro']*100:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("运行统计\n")
        f.write("=" * 80 + "\n")
        stats = results["stats"]
        f.write(f"总调用次数: {stats['total_calls']}\n")
        f.write(f"成功次数: {stats['total_calls'] - stats['errors']}\n")
        f.write(f"失败次数: {stats['errors']}\n")
        f.write(f"成功率: {(stats['total_calls']-stats['errors'])/stats['total_calls']*100:.1f}%\n" if stats['total_calls'] > 0 else "成功率: N/A\n")
        f.write(f"总Token消耗: {stats['total_tokens']}\n")
        f.write(f"  输入Token: {stats['total_input_tokens']}\n")
        f.write(f"  输出Token: {stats['total_output_tokens']}\n")
        f.write(f"总时间: {stats['total_time']:.2f}秒\n")
        f.write(f"平均时间: {stats['total_time']/stats['total_calls']:.2f}秒/样本\n" if stats['total_calls'] > 0 else "平均时间: N/A\n")

    print(f"✓ 实验报告已保存: {report_file}")

    return json_file, report_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="灵活的LLM分类实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置运行（选择一个模型）
  python run_llm_experiment.py

  # 运行指定模型
  python run_llm_experiment.py --model glm-4.6

  # 运行所有启用的模型
  python run_llm_experiment.py --all

  # 指定样本数
  python run_llm_experiment.py --model deepseek --sample 100

  # 使用自定义配置文件
  python run_llm_experiment.py --config my_config.json
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        help="指定模型名称（在配置文件中定义）"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有启用的模型"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="llm_config.json",
        help="配置文件路径（默认: llm_config.json）"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="测试样本数（默认使用配置文件中的值）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/llm_experiments",
        help="输出目录（默认: output/llm_experiments）"
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" " * 25 + "LLM分类实验")
    print("=" * 80)

    # 1. 加载配置
    print("\n[步骤 1/4] 加载配置")
    print("-" * 80)

    try:
        config = load_config(args.config)
        print(f"✓ 配置文件已加载: {args.config}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # 2. 验证API密钥
    available_models = validate_api_keys(config)

    if not available_models:
        print("\n❌ 没有可用的模型！请在配置文件中设置API密钥并启用模型。")
        sys.exit(1)

    print(f"✓ 可用模型: {', '.join(available_models)}")

    # 3. 选择模型
    if args.all:
        selected_models = available_models
        print(f"\n将运行所有 {len(selected_models)} 个模型")
    elif args.model:
        if args.model not in available_models:
            print(f"\n❌ 模型 '{args.model}' 不可用")
            print(f"可用模型: {', '.join(available_models)}")
            sys.exit(1)
        selected_models = [args.model]
        print(f"\n将运行模型: {args.model}")
    else:
        # 交互式选择
        print("\n请选择要运行的模型:")
        for i, model in enumerate(available_models, 1):
            model_info = config["llms"][model]
            print(f"  {i}. {model} ({model_info.get('comment', model_info['model'])})")
        print(f"  {len(available_models)+1}. 运行所有模型")

        try:
            choice = int(input("\n请输入选项 (1-{}): ".format(len(available_models)+1)))
            if choice == len(available_models) + 1:
                selected_models = available_models
            elif 1 <= choice <= len(available_models):
                selected_models = [available_models[choice-1]]
            else:
                print("无效选项")
                sys.exit(1)
        except (ValueError, KeyboardInterrupt):
            print("\n已取消")
            sys.exit(0)

    # 4. 加载数据
    print("\n[步骤 2/4] 加载测试数据")
    print("-" * 80)

    try:
        _, _, test_titles, test_labels = DataLoader.prepare_dataset(
            'data/positive.txt',
            'data/negative.txt',
            'data/testSet-1000.xlsx'
        )
        print(f"✓ 测试集: {len(test_titles)} 样本")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        sys.exit(1)

    # 5. 确定样本数
    sample_size = args.sample or config["experiment"].get("sample_size", len(test_titles))
    if sample_size > len(test_titles):
        sample_size = len(test_titles)

    print(f"✓ 将使用 {sample_size} 个样本")

    # 6. 运行实验
    print("\n[步骤 3/4] 运行实验")
    print("-" * 80)

    all_results = {}

    for model_name in selected_models:
        try:
            print(f"\n{'='*80}")
            print(f"实验: {model_name}")
            print(f"{'='*80}")

            results = run_single_experiment(
                model_name,
                config,
                test_titles,
                test_labels,
                sample_size
            )

            all_results[model_name] = results

            # 保存结果
            save_experiment_results(results, model_name, args.output)

        except Exception as e:
            print(f"\n❌ {model_name} 实验失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 7. 总结
    print("\n[步骤 4/4] 实验总结")
    print("-" * 80)

    if len(all_results) == 0:
        print("\n❌ 所有实验都失败了")
        sys.exit(1)

    print(f"\n{'模型':<20} {'准确率':>10} {'F1分数':>10} {'Token消耗':>12} {'平均耗时':>12}")
    print("-" * 70)

    for model_name, results in all_results.items():
        metrics = results["eval_result"]
        stats = results["stats"]
        avg_time = stats["total_time"] / stats["total_calls"] if stats["total_calls"] > 0 else 0

        print(f"{model_name:<20} {metrics['accuracy']*100:>9.2f}% {metrics['f1']*100:>9.2f}% "
              f"{stats['total_tokens']:>12} {avg_time:>11.2f}s")

    print("\n" + "=" * 80)
    print(" 实验完成!")
    print("=" * 80)
    print(f"\n结果已保存至: {args.output}/")


if __name__ == "__main__":
    main()

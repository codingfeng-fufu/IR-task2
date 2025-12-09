"""
llm_multi_experiment.py
=======================
多个LLM模型对比实验
支持：国内大模型（智谱GLM-4, 阿里Qwen, Kimi, DeepSeek）

使用方法：
1. 编辑 llm_config.json，填写API密钥
2. 运行: python llm_multi_experiment.py
"""

import os
import sys
import json
import time
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from evaluator import ModelEvaluator


class UnifiedLLMClient:
    """统一的LLM客户端，支持多个provider"""

    def __init__(self, config: Dict):
        """
        初始化LLM客户端

        Args:
            config: LLM配置字典，包含provider, model, api_key等
        """
        self.config = config
        self.provider = config["provider"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 150)
        self.api_key = config["api_key"]

        # 统计信息
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "errors": 0,
            "total_time": 0.0
        }

        # 初始化对应的客户端
        self._init_client()

    def _init_client(self):
        """根据provider初始化对应的客户端"""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "google":
            self._init_google()
        else:
            raise ValueError(f"不支持的provider: {self.provider}")

    def _init_openai(self):
        """初始化OpenAI客户端（兼容OpenAI和国内大模型）"""
        try:
            from openai import OpenAI

            # 构建客户端参数
            client_params = {"api_key": self.api_key}

            # 如果提供了自定义base_url（用于国内大模型）
            if self.config.get("base_url"):
                client_params["base_url"] = self.config["base_url"]

            self.client = OpenAI(**client_params)

            # 根据base_url判断是国内还是国际模型
            if self.config.get("base_url"):
                print(f"✓ 客户端初始化成功: {self.model} (自定义端点)")
            else:
                print(f"✓ OpenAI客户端初始化成功: {self.model}")
        except ImportError:
            raise ImportError("请安装openai: pip install openai>=1.0.0")
        except Exception as e:
            raise Exception(f"客户端初始化失败: {e}")

    def _init_anthropic(self):
        """初始化Anthropic客户端"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print(f"✓ Anthropic客户端初始化成功: {self.model}")
        except ImportError:
            raise ImportError("请安装anthropic: pip install anthropic")
        except Exception as e:
            raise Exception(f"Anthropic客户端初始化失败: {e}")

    def _init_google(self):
        """初始化Google客户端"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            print(f"✓ Google客户端初始化成功: {self.model}")
        except ImportError:
            raise ImportError("请安装google-generativeai: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Google客户端初始化失败: {e}")

    def generate(self, prompt: str) -> Dict:
        """
        调用LLM生成响应

        Returns:
            {
                "response": str,
                "tokens": int,
                "time": float
            }
        """
        start_time = time.time()

        try:
            if self.provider == "openai":
                result = self._generate_openai(prompt)
            elif self.provider == "anthropic":
                result = self._generate_anthropic(prompt)
            elif self.provider == "google":
                result = self._generate_google(prompt)
            else:
                raise ValueError(f"不支持的provider: {self.provider}")

            # 统计
            elapsed = time.time() - start_time
            result["time"] = elapsed
            self.stats["total_calls"] += 1
            self.stats["total_tokens"] += result.get("tokens", 0)
            self.stats["total_time"] += elapsed

            return result

        except Exception as e:
            self.stats["errors"] += 1
            return {
                "response": "",
                "tokens": 0,
                "time": time.time() - start_time,
                "error": str(e)
            }

    def _generate_openai(self, prompt: str) -> Dict:
        """OpenAI API调用（兼容OpenAI和国内大模型）"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个准确、专业的学术标题分类专家。"
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return {
            "response": response.choices[0].message.content.strip(),
            "tokens": response.usage.total_tokens if response.usage else 0
        }

    def _generate_anthropic(self, prompt: str) -> Dict:
        """Anthropic API调用"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return {
            "response": message.content[0].text,
            "tokens": message.usage.input_tokens + message.usage.output_tokens
        }

    def _generate_google(self, prompt: str) -> Dict:
        """Google Gemini API调用"""
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
        )

        return {
            "response": response.text,
            "tokens": 0  # Gemini API不直接返回token数
        }


class LLMExperiment:
    """LLM对比实验管理器"""

    def __init__(self, config_file: str = "llm_config.json"):
        """
        初始化实验

        Args:
            config_file: 配置文件路径
        """
        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 获取Few-shot示例
        self.examples = self._get_few_shot_examples(
            self.config["experiment"]["few_shot_examples"]
        )

        # 初始化启用的LLM客户端
        self.llm_clients = {}
        for name, llm_config in self.config["llms"].items():
            if llm_config.get("enabled", True):
                try:
                    self.llm_clients[name] = UnifiedLLMClient(llm_config)
                except Exception as e:
                    print(f"⚠️  {name} 初始化失败: {e}")
                    print(f"   跳过 {name}")

        if not self.llm_clients:
            raise Exception("没有可用的LLM客户端！请检查配置文件和API密钥。")

        print(f"\n✓ 成功初始化 {len(self.llm_clients)} 个LLM客户端")

    def _get_few_shot_examples(self, num_examples: int) -> List[Dict]:
        """获取Few-shot示例"""
        all_examples = [
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
        return all_examples[:num_examples]

    def _create_prompt(self, title: str) -> str:
        """创建Few-shot Prompt"""
        prompt = """你是一个专业的学术论文标题质量评估专家。判断给定标题是否为正确提取的学术论文标题。

【分类标准】
✓ 正确标题（1）：完整、清晰的学术论文标题，语法正确
✗ 错误标题（0）：包含页码、摘要、章节编号、期刊信息、格式错误等

【参考示例】

"""
        # 添加示例
        for i, ex in enumerate(self.examples, 1):
            label_symbol = "✓" if ex["label"] == 1 else "✗"
            label_text = "正确标题" if ex["label"] == 1 else "错误标题"
            prompt += f"示例 {i}:\n"
            prompt += f"标题：「{ex['title']}」\n"
            prompt += f"判断：{label_symbol} {label_text}\n"
            prompt += f"理由：{ex['reason']}\n\n"

        # 待分类标题
        prompt += "=" * 60 + "\n"
        prompt += "【现在请判断以下标题】\n\n"
        prompt += f"标题：「{title}」\n"
        prompt += f"判断：\n\n"
        prompt += "请只回答\"✓ 正确标题\"或\"✗ 错误标题\"。"

        return prompt

    def _parse_response(self, response: str) -> int:
        """解析LLM响应"""
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
                return 0  # 默认

    def run_experiment(
        self,
        test_titles: List[str],
        test_labels: List[int],
        sample_size: Optional[int] = None
    ) -> Dict:
        """
        运行完整实验

        Returns:
            {
                "llm_name": {
                    "predictions": np.ndarray,
                    "results": {...},  # 评估结果
                    "details": [...]  # 详细预测
                }
            }
        """
        print("\n" + "=" * 80)
        print(" " * 25 + "LLM对比实验")
        print("=" * 80)

        # 采样
        if sample_size and sample_size < len(test_titles):
            print(f"\n使用前 {sample_size} 个样本进行测试")
            test_titles = test_titles[:sample_size]
            test_labels = test_labels[:sample_size]
        else:
            sample_size = len(test_titles)

        print(f"测试样本数: {sample_size}")
        print(f"Few-shot示例数: {len(self.examples)}")
        print(f"参与实验的LLM: {', '.join(self.llm_clients.keys())}")

        # 对每个LLM进行实验
        all_results = {}
        evaluator = ModelEvaluator()

        for llm_name, client in self.llm_clients.items():
            print(f"\n{'='*80}")
            print(f"实验 {llm_name}")
            print(f"{'='*80}")

            predictions = []
            details = []

            for i, title in enumerate(tqdm(test_titles, desc=f"{llm_name} 预测进度")):
                # 创建prompt
                prompt = self._create_prompt(title)

                # 调用LLM
                result = client.generate(prompt)

                # 解析响应
                label = self._parse_response(result["response"])
                predictions.append(label)

                # 保存详细信息
                details.append({
                    "title": title,
                    "true_label": int(test_labels[i]),
                    "pred_label": label,
                    "response": result["response"],
                    "tokens": result.get("tokens", 0),
                    "time": result.get("time", 0.0),
                    "error": result.get("error", None)
                })

                # API限流延迟
                if i < len(test_titles) - 1:
                    time.sleep(self.config["experiment"]["delay_between_calls"])

            # 评估性能
            predictions = np.array(predictions)
            eval_result = evaluator.evaluate_model(
                test_labels,
                predictions,
                model_name=llm_name,
                verbose=True
            )

            # 统计信息
            print(f"\n统计信息:")
            print(f"  总调用次数: {client.stats['total_calls']}")
            print(f"  总Token消耗: {client.stats['total_tokens']}")
            print(f"  总时间: {client.stats['total_time']:.2f}秒")
            print(f"  平均时间: {client.stats['total_time']/client.stats['total_calls']:.2f}秒/样本")
            print(f"  错误次数: {client.stats['errors']}")

            # 保存结果
            all_results[llm_name] = {
                "predictions": predictions,
                "eval_result": eval_result,
                "details": details,
                "stats": client.stats.copy()
            }

        return all_results

    def _convert_to_json_serializable(self, obj):
        """递归地将numpy类型转换为Python原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def save_results(self, results: Dict, output_dir: str = None):
        """保存实验结果"""
        if output_dir is None:
            output_dir = self.config["experiment"]["output_dir"]

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存详细结果（JSON）
        detailed_results = {}
        for llm_name, result in results.items():
            detailed_results[llm_name] = {
                "eval_metrics": self._convert_to_json_serializable(result["eval_result"]),
                "stats": result["stats"],
                "predictions": result["predictions"].tolist(),
                "details": self._convert_to_json_serializable(result["details"])
            }

        json_file = os.path.join(output_dir, f"llm_comparison_{timestamp}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 详细结果已保存: {json_file}")

        # 2. 保存对比报告（文本）
        report_file = os.path.join(output_dir, f"llm_comparison_report_{timestamp}.txt")
        self._generate_report(results, report_file)
        print(f"✓ 对比报告已保存: {report_file}")

        # 3. 保存对比表格（CSV）
        csv_file = os.path.join(output_dir, f"llm_comparison_{timestamp}.csv")
        self._generate_csv(results, csv_file)
        print(f"✓ 对比表格已保存: {csv_file}")

    def _generate_report(self, results: Dict, output_file: str):
        """生成文本格式的对比报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(" " * 35 + "LLM对比实验报告\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Few-shot示例数: {len(self.examples)}\n")
            f.write(f"测试样本数: {self.config['experiment']['sample_size']}\n")
            f.write(f"参与模型数: {len(results)}\n\n")

            # 性能对比表
            f.write("=" * 100 + "\n")
            f.write("性能对比\n")
            f.write("=" * 100 + "\n\n")

            header = f"{'模型':<25} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'平均耗时':>12}\n"
            f.write(header)
            f.write("-" * 100 + "\n")

            for llm_name, result in results.items():
                metrics = result["eval_result"]
                stats = result["stats"]
                avg_time = stats["total_time"] / stats["total_calls"] if stats["total_calls"] > 0 else 0

                line = f"{llm_name:<25} "
                line += f"{metrics['accuracy']*100:>9.2f}% "
                line += f"{metrics['precision']*100:>9.2f}% "
                line += f"{metrics['recall']*100:>9.2f}% "
                line += f"{metrics['f1']*100:>9.2f}% "
                line += f"{avg_time:>11.2f}s\n"
                f.write(line)

            # 找出最佳模型
            f.write("\n" + "=" * 100 + "\n")
            f.write("最佳模型\n")
            f.write("=" * 100 + "\n\n")

            best_acc = max(results.items(), key=lambda x: x[1]["eval_result"]["accuracy"])
            best_f1 = max(results.items(), key=lambda x: x[1]["eval_result"]["f1"])

            f.write(f"最佳准确率: {best_acc[0]} ({best_acc[1]['eval_result']['accuracy']*100:.2f}%)\n")
            f.write(f"最佳F1分数: {best_f1[0]} ({best_f1[1]['eval_result']['f1']*100:.2f}%)\n")

            # 成本分析
            f.write("\n" + "=" * 100 + "\n")
            f.write("成本与效率分析\n")
            f.write("=" * 100 + "\n\n")

            header = f"{'模型':<25} {'总Token':>12} {'总时间':>12} {'错误数':>10} {'成功率':>10}\n"
            f.write(header)
            f.write("-" * 100 + "\n")

            for llm_name, result in results.items():
                stats = result["stats"]
                success_rate = (stats["total_calls"] - stats["errors"]) / stats["total_calls"] * 100 if stats["total_calls"] > 0 else 0

                line = f"{llm_name:<25} "
                line += f"{stats['total_tokens']:>12} "
                line += f"{stats['total_time']:>11.2f}s "
                line += f"{stats['errors']:>10} "
                line += f"{success_rate:>9.1f}%\n"
                f.write(line)

    def _generate_csv(self, results: Dict, output_file: str):
        """生成CSV格式的对比表格"""
        import csv

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入表头
            writer.writerow([
                "模型", "准确率", "精确率", "召回率", "F1分数",
                "总Token", "总时间(秒)", "平均时间(秒)", "错误数", "成功率(%)"
            ])

            # 写入数据
            for llm_name, result in results.items():
                metrics = result["eval_result"]
                stats = result["stats"]
                avg_time = stats["total_time"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
                success_rate = (stats["total_calls"] - stats["errors"]) / stats["total_calls"] * 100 if stats["total_calls"] > 0 else 0

                writer.writerow([
                    llm_name,
                    f"{metrics['accuracy']*100:.2f}",
                    f"{metrics['precision']*100:.2f}",
                    f"{metrics['recall']*100:.2f}",
                    f"{metrics['f1']*100:.2f}",
                    stats['total_tokens'],
                    f"{stats['total_time']:.2f}",
                    f"{avg_time:.2f}",
                    stats['errors'],
                    f"{success_rate:.1f}"
                ])

    def print_comparison_summary(self, results: Dict):
        """打印对比摘要"""
        print("\n" + "=" * 80)
        print(" " * 30 + "实验总结")
        print("=" * 80)

        print(f"\n{'模型':<20} {'准确率':>10} {'F1分数':>10} {'平均耗时':>12}")
        print("-" * 55)

        for llm_name, result in results.items():
            metrics = result["eval_result"]
            stats = result["stats"]
            avg_time = stats["total_time"] / stats["total_calls"] if stats["total_calls"] > 0 else 0

            print(f"{llm_name:<20} {metrics['accuracy']*100:>9.2f}% {metrics['f1']*100:>9.2f}% {avg_time:>11.2f}s")

        print("\n" + "=" * 80)


def main():
    """主函数"""
    print("=" * 80)
    print(" " * 25 + "LLM多模型对比实验")
    print("=" * 80)

    # 检查配置文件
    config_file = "llm_config.json"
    if not os.path.exists(config_file):
        print(f"\n❌ 配置文件不存在: {config_file}")
        print("请先创建配置文件并填写API密钥")
        sys.exit(1)

    # 检查API密钥
    with open(config_file, 'r') as f:
        config = json.load(f)

    has_valid_key = False
    for name, llm_config in config["llms"].items():
        if llm_config.get("enabled", True):
            if "YOUR_" not in llm_config.get("api_key", ""):
                has_valid_key = True
                break

    if not has_valid_key:
        print("\n❌ 请先在 llm_config.json 中填写至少一个有效的API密钥")
        print("\n示例:")
        print('  "api_key": "sk-xxxxxxxxxxxx"')
        sys.exit(1)

    # 加载数据
    print("\n[步骤 1/3] 加载测试数据")
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

    # 初始化实验
    print("\n[步骤 2/3] 初始化实验")
    print("-" * 80)

    try:
        experiment = LLMExperiment(config_file)
    except Exception as e:
        print(f"❌ 实验初始化失败: {e}")
        sys.exit(1)

    # 运行实验
    print("\n[步骤 3/3] 运行实验")
    print("-" * 80)

    # 询问样本数
    sample_size = config["experiment"]["sample_size"]
    user_input = input(f"\n使用默认样本数 {sample_size}？(y/n，输入n可自定义): ").lower()

    if user_input == 'n':
        try:
            sample_size = int(input("请输入样本数: "))
        except:
            print("输入无效，使用默认值")

    try:
        results = experiment.run_experiment(
            test_titles,
            test_labels,
            sample_size=sample_size
        )

        # 打印摘要
        experiment.print_comparison_summary(results)

        # 保存结果
        experiment.save_results(results)

        print("\n" + "=" * 80)
        print(" 实验完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

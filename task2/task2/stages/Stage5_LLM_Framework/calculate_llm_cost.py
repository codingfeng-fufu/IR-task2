"""
calculate_llm_cost.py
=====================
计算LLM实验的Token消耗和成本
"""

import pandas as pd
import json
from typing import Dict, List


class LLMCostCalculator:
    """LLM成本计算器"""

    # 2024年12月最新定价（美元/百万tokens）
    PRICING = {
        "gpt-3.5-turbo": {
            "input": 0.50,   # $0.50 per 1M tokens
            "output": 1.50,  # $1.50 per 1M tokens
            "name": "GPT-3.5-turbo"
        },
        "gpt-4": {
            "input": 30.00,  # $30 per 1M tokens
            "output": 60.00, # $60 per 1M tokens
            "name": "GPT-4"
        },
        "gpt-4-turbo": {
            "input": 10.00,  # $10 per 1M tokens
            "output": 30.00, # $30 per 1M tokens
            "name": "GPT-4-turbo"
        },
        "claude-3-5-haiku": {
            "input": 1.00,   # $1 per 1M tokens
            "output": 5.00,  # $5 per 1M tokens
            "name": "Claude-3.5-Haiku"
        },
        "claude-3-5-sonnet": {
            "input": 3.00,   # $3 per 1M tokens
            "output": 15.00, # $15 per 1M tokens
            "name": "Claude-3.5-Sonnet"
        },
        "claude-3-opus": {
            "input": 15.00,  # $15 per 1M tokens
            "output": 75.00, # $75 per 1M tokens
            "name": "Claude-3-Opus"
        },
        "gemini-1.5-pro": {
            "input": 1.25,   # $1.25 per 1M tokens (>128K)
            "output": 5.00,  # $5 per 1M tokens
            "name": "Gemini-1.5-Pro"
        },
        "gemini-1.5-flash": {
            "input": 0.075,  # $0.075 per 1M tokens
            "output": 0.30,  # $0.30 per 1M tokens
            "name": "Gemini-1.5-Flash"
        }
    }

    def __init__(self):
        self.sample_count = 0
        self.avg_title_length = 0
        self.prompt_tokens_per_sample = 0
        self.output_tokens_per_sample = 0

    def analyze_dataset(self, test_file: str = "data/testSet-1000.xlsx"):
        """分析数据集"""
        print("="*80)
        print("步骤 1: 分析数据集")
        print("="*80)

        try:
            # 读取测试数据
            df = pd.read_excel(test_file)

            # 过滤掉NaN
            titles = df['title given by manchine'].dropna().tolist()
            self.sample_count = len(titles)

            # 计算平均长度
            total_chars = sum(len(str(title)) for title in titles)
            self.avg_title_length = total_chars / len(titles)

            # 统计信息
            title_lengths = [len(str(title)) for title in titles]
            min_len = min(title_lengths)
            max_len = max(title_lengths)

            print(f"✓ 测试集样本数: {self.sample_count}")
            print(f"✓ 平均标题长度: {self.avg_title_length:.1f} 字符")
            print(f"✓ 标题长度范围: {min_len} - {max_len} 字符")

            # 显示示例
            print(f"\n示例标题:")
            for i in range(min(5, len(titles))):
                print(f"  {i+1}. {str(titles[i])[:70]}")

        except Exception as e:
            print(f"⚠️  读取数据失败: {e}")
            print("使用估算值...")
            self.sample_count = 1000
            self.avg_title_length = 60  # 估算平均60字符
            print(f"✓ 估算样本数: {self.sample_count}")
            print(f"✓ 估算平均长度: {self.avg_title_length} 字符")

    def estimate_tokens(self, few_shot_examples: int = 8):
        """估算Token消耗"""
        print(f"\n{'='*80}")
        print("步骤 2: 估算Token消耗")
        print("="*80)

        # Few-shot示例的token估算
        # 每个示例约：标题(60字符≈15tokens) + 标签(10tokens) + 理由(30tokens) = 55tokens
        examples_tokens = few_shot_examples * 55

        # Prompt模板的tokens（系统提示 + 分类标准 + 格式说明）
        template_tokens = 200

        # 待分类标题的tokens（平均60字符 ≈ 15 tokens）
        title_tokens = int(self.avg_title_length / 4)  # 粗略估算：4字符≈1token

        # 输入tokens = 模板 + 示例 + 标题
        self.prompt_tokens_per_sample = template_tokens + examples_tokens + title_tokens

        # 输出tokens（响应：判断 + 理由）≈ 50 tokens
        self.output_tokens_per_sample = 50

        print(f"\nToken组成（每个样本）:")
        print(f"  - Prompt模板: {template_tokens} tokens")
        print(f"  - Few-shot示例 ({few_shot_examples}个): {examples_tokens} tokens")
        print(f"  - 待分类标题: {title_tokens} tokens")
        print(f"  - LLM输出响应: {self.output_tokens_per_sample} tokens")
        print(f"\n单样本Token消耗:")
        print(f"  - 输入: {self.prompt_tokens_per_sample} tokens")
        print(f"  - 输出: {self.output_tokens_per_sample} tokens")
        print(f"  - 合计: {self.prompt_tokens_per_sample + self.output_tokens_per_sample} tokens")

        print(f"\n全部{self.sample_count}个样本Token消耗:")
        total_input = self.prompt_tokens_per_sample * self.sample_count
        total_output = self.output_tokens_per_sample * self.sample_count
        total = total_input + total_output
        print(f"  - 输入: {total_input:,} tokens ({total_input/1000:.1f}K)")
        print(f"  - 输出: {total_output:,} tokens ({total_output/1000:.1f}K)")
        print(f"  - 合计: {total:,} tokens ({total/1000:.1f}K)")

    def calculate_costs(self) -> Dict:
        """计算各个LLM的成本"""
        print(f"\n{'='*80}")
        print("步骤 3: 计算各LLM成本")
        print("="*80)

        total_input_tokens = self.prompt_tokens_per_sample * self.sample_count
        total_output_tokens = self.output_tokens_per_sample * self.sample_count

        results = []

        print(f"\n{'模型':<25} {'输入成本':>12} {'输出成本':>12} {'总成本':>12} {'人民币':>12}")
        print("-" * 80)

        for model_id, pricing in self.PRICING.items():
            # 计算成本（美元）
            input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
            output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
            total_cost = input_cost + output_cost

            # 转换为人民币（汇率7.2）
            total_cny = total_cost * 7.2

            results.append({
                "model": pricing["name"],
                "model_id": model_id,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": total_cost,
                "total_cost_cny": total_cny
            })

            print(f"{pricing['name']:<25} ${input_cost:>10.2f} ${output_cost:>10.2f} ${total_cost:>10.2f} ¥{total_cny:>10.2f}")

        return results

    def generate_recommendations(self, results: List[Dict]):
        """生成推荐方案"""
        print(f"\n{'='*80}")
        print("步骤 4: 推荐方案")
        print("="*80)

        # 按成本排序
        sorted_by_cost = sorted(results, key=lambda x: x["total_cost_usd"])

        print("\n【方案一：极致性价比】")
        cheapest = sorted_by_cost[0]
        print(f"  推荐: {cheapest['model']}")
        print(f"  成本: ${cheapest['total_cost_usd']:.2f} (¥{cheapest['total_cost_cny']:.2f})")
        print(f"  优势: 成本最低")

        print("\n【方案二：免费试用】")
        print(f"  推荐: Gemini-1.5-Flash")
        gemini_flash = next(r for r in results if "Flash" in r["model"])
        print(f"  成本: ${gemini_flash['total_cost_usd']:.2f} (¥{gemini_flash['total_cost_cny']:.2f})")
        print(f"  优势: 几乎免费，适合大规模测试")

        print("\n【方案三：平衡性能与成本】")
        print(f"  推荐: GPT-3.5-turbo + Claude-3.5-Haiku")
        gpt35 = next(r for r in results if r["model_id"] == "gpt-3.5-turbo")
        haiku = next(r for r in results if "Haiku" in r["model"])
        combined_cost = gpt35['total_cost_usd'] + haiku['total_cost_usd']
        print(f"  成本: ${combined_cost:.2f} (¥{combined_cost*7.2:.2f})")
        print(f"  优势: 两个快速、廉价的模型对比")

        print("\n【方案四：追求最佳性能】")
        print(f"  推荐: GPT-4-turbo + Claude-3.5-Sonnet")
        gpt4t = next(r for r in results if "turbo" in r["model_id"])
        sonnet = next(r for r in results if "Sonnet" in r["model"])
        combined_cost2 = gpt4t['total_cost_usd'] + sonnet['total_cost_usd']
        print(f"  成本: ${combined_cost2:.2f} (¥{combined_cost2*7.2:.2f})")
        print(f"  优势: 高性能模型，准确率可能更高")

    def save_report(self, results: List[Dict], output_file: str = "llm_cost_estimation.json"):
        """保存成本估算报告"""
        report = {
            "dataset": {
                "sample_count": self.sample_count,
                "avg_title_length": self.avg_title_length
            },
            "tokens": {
                "input_per_sample": self.prompt_tokens_per_sample,
                "output_per_sample": self.output_tokens_per_sample,
                "total_input": self.prompt_tokens_per_sample * self.sample_count,
                "total_output": self.output_tokens_per_sample * self.sample_count
            },
            "costs": results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 详细报告已保存至: {output_file}")


def main():
    """主函数"""
    print("="*80)
    print(" "*25 + "LLM成本估算工具")
    print("="*80)
    print()

    calculator = LLMCostCalculator()

    # 1. 分析数据集
    calculator.analyze_dataset()

    # 2. 估算tokens
    calculator.estimate_tokens(few_shot_examples=8)

    # 3. 计算成本
    results = calculator.calculate_costs()

    # 4. 生成推荐
    calculator.generate_recommendations(results)

    # 5. 保存报告
    calculator.save_report(results)

    print(f"\n{'='*80}")
    print("成本估算完成!")
    print("="*80)
    print("\n提示:")
    print("  1. 以上成本基于2024年12月的API定价")
    print("  2. 实际成本可能因Token计数方式略有差异(±10%)")
    print("  3. 建议先用小样本(100条)测试，验证实际消耗")
    print("  4. Gemini有免费额度，可优先使用")
    print()


if __name__ == "__main__":
    main()

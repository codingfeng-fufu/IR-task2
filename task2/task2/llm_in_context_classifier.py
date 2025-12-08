"""
llm_in_context_classifier.py
=============================
LLM In-Context Learning 分类器
使用生成式大语言模型进行学术标题分类（零训练，Few-shot）

支持：
1. OpenAI API (GPT-3.5/GPT-4)
2. 本地开源LLM（可选）
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm


class LLMInContextClassifier:
    """
    基于LLM的In-Context Learning分类器

    核心思想：
    - 不训练模型，使用Few-shot示例
    - 通过精心设计的Prompt引导LLM分类
    - 可解释性强（可要求给出理由）
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        examples: Optional[List[Dict]] = None,
        temperature: float = 0.0
    ):
        """
        初始化LLM分类器

        Args:
            provider: API提供商 ("openai", "claude", "local")
            model: 模型名称
            api_key: API密钥（如果使用API）
            examples: Few-shot示例列表
            temperature: 生成温度（0=确定性，1=随机性）
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature

        # 设置API
        if provider == "openai":
            try:
                import openai
                openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not openai.api_key:
                    raise ValueError("需要设置OPENAI_API_KEY环境变量或传入api_key参数")
                self.client = openai
                print(f"✓ 已连接到OpenAI API，使用模型: {model}")
            except ImportError:
                raise ImportError("请先安装openai: pip install openai")

        elif provider == "local":
            print("⚠️  本地模型需要GPU，请确保已安装transformers和torch")
            self._init_local_model()
        else:
            raise ValueError(f"不支持的provider: {provider}")

        # Few-shot示例
        self.examples = examples or self._get_default_examples()
        print(f"✓ 已加载 {len(self.examples)} 个Few-shot示例")

        # 统计信息
        self.stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "errors": 0
        }

    def _get_default_examples(self) -> List[Dict]:
        """
        获取默认的Few-shot示例

        策略：选择典型的正例和负例，覆盖常见错误模式
        """
        return [
            # 正确标题示例
            {
                "title": "Deep Learning for Natural Language Processing: A Survey",
                "label": 1,
                "reason": "完整规范的学术论文标题，主题明确"
            },
            {
                "title": "Machine Translation Using Neural Networks",
                "label": 1,
                "reason": "清晰准确的研究标题，语法正确"
            },
            {
                "title": "A Comparative Study of Sentiment Analysis Methods",
                "label": 1,
                "reason": "标准的学术论文标题格式"
            },

            # 错误标题示例 - 页码模式
            {
                "title": "pp. 123-145 Introduction to Machine Learning",
                "label": 0,
                "reason": "包含页码信息（pp.），非标准标题"
            },

            # 错误标题示例 - 摘要片段
            {
                "title": "Abstract: This paper presents a new method for text classification",
                "label": 0,
                "reason": "包含'Abstract'关键词，是摘要片段而非标题"
            },

            # 错误标题示例 - 格式错误
            {
                "title": "A Novel Approach to Deep Learning......",
                "label": 0,
                "reason": "包含连续点号（......），疑似提取错误"
            },

            # 错误标题示例 - 期刊信息
            {
                "title": "Vol. 25, No. 3, 2024 - Neural Networks",
                "label": 0,
                "reason": "包含期刊卷号信息，非标题内容"
            },

            # 错误标题示例 - 引言片段
            {
                "title": "1. Introduction Recent advances in deep learning",
                "label": 0,
                "reason": "包含章节编号和'Introduction'，是引言片段"
            }
        ]

    def _create_prompt(self, title: str, include_reasoning: bool = True) -> str:
        """
        创建Few-shot Prompt

        Prompt设计策略：
        1. 明确任务定义
        2. 给出清晰的分类标准
        3. 提供多样化的示例
        4. 指定输出格式
        """
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

            if include_reasoning and "reason" in ex:
                prompt += f"理由：{ex['reason']}\n"

            prompt += "\n"

        # 待分类标题
        prompt += "=" * 60 + "\n"
        prompt += "【现在请判断以下标题】\n\n"
        prompt += f"标题：「{title}」\n"
        prompt += f"判断：\n"

        if include_reasoning:
            prompt += "\n请先回答\"✓ 正确标题\"或\"✗ 错误标题\"，然后用一句话简要说明理由。"
        else:
            prompt += "\n请只回答\"✓ 正确标题\"或\"✗ 错误标题\"。"

        return prompt

    def _parse_response(self, response: str) -> Tuple[int, str]:
        """
        解析LLM的响应

        Returns:
            (label, reason)
        """
        response_lower = response.lower()

        # 判断标签
        if "✓" in response or "正确标题" in response or "correct" in response_lower:
            label = 1
        elif "✗" in response or "错误标题" in response or "incorrect" in response_lower:
            label = 0
        else:
            # 尝试从数字解析
            if "1" in response[:10]:
                label = 1
            elif "0" in response[:10]:
                label = 0
            else:
                # 默认为错误（保守策略）
                label = 0
                print(f"⚠️  无法明确解析响应，默认为0: {response[:50]}...")

        # 提取理由
        lines = response.strip().split('\n')
        reason = lines[0] if lines else response

        return label, reason

    def predict_single(self, title: str, verbose: bool = False) -> Dict:
        """
        预测单个标题

        Returns:
            {
                "title": str,
                "label": int (0 or 1),
                "response": str (LLM的完整响应),
                "reason": str (分类理由),
                "confidence": float (置信度，暂时为1.0)
            }
        """
        prompt = self._create_prompt(title)

        if self.provider == "openai":
            return self._predict_openai(prompt, title, verbose)
        else:
            return self._predict_local(prompt, title, verbose)

    def _predict_openai(self, prompt: str, title: str, verbose: bool) -> Dict:
        """使用OpenAI API预测"""
        try:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个准确、专业的学术标题分类专家。请严格按照给定的标准进行判断。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=150
            )

            answer = response.choices[0].message.content.strip()

            # 统计信息
            self.stats["total_calls"] += 1
            if "usage" in response:
                self.stats["total_tokens"] += response["usage"]["total_tokens"]

            # 解析响应
            label, reason = self._parse_response(answer)

            if verbose:
                print(f"\n标题: {title}")
                print(f"响应: {answer}")
                print(f"判断: {label}")

            return {
                "title": title,
                "label": label,
                "response": answer,
                "reason": reason,
                "confidence": 1.0,
                "tokens": response.get("usage", {}).get("total_tokens", 0)
            }

        except Exception as e:
            print(f"❌ API调用失败: {e}")
            self.stats["errors"] += 1

            return {
                "title": title,
                "label": 0,  # 默认为错误
                "response": "",
                "reason": f"Error: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }

    def predict(
        self,
        titles: List[str],
        delay: float = 0.5,
        verbose: bool = False
    ) -> np.ndarray:
        """
        批量预测

        Args:
            titles: 标题列表
            delay: API调用间隔（秒），避免超限
            verbose: 是否打印详细信息

        Returns:
            predictions: numpy数组，形状为(n,)
        """
        predictions = []
        results = []

        print(f"\n{'='*60}")
        print(f"LLM In-Context Learning 分类器")
        print(f"{'='*60}")
        print(f"模型: {self.model}")
        print(f"Few-shot示例数: {len(self.examples)}")
        print(f"待预测样本数: {len(titles)}")
        print(f"Temperature: {self.temperature}")
        print(f"{'='*60}\n")

        for i, title in enumerate(tqdm(titles, desc="预测进度")):
            result = self.predict_single(title, verbose=verbose)
            predictions.append(result["label"])
            results.append(result)

            # 延迟避免API限流
            if i < len(titles) - 1 and self.provider == "openai":
                time.sleep(delay)

        # 保存详细结果
        self.last_results = results

        # 打印统计信息
        print(f"\n{'='*60}")
        print("预测完成统计")
        print(f"{'='*60}")
        print(f"总调用次数: {self.stats['total_calls']}")
        print(f"总Token消耗: {self.stats['total_tokens']}")
        print(f"错误次数: {self.stats['errors']}")
        print(f"预测为正类: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
        print(f"预测为负类: {len(predictions)-sum(predictions)} ({(len(predictions)-sum(predictions))/len(predictions)*100:.1f}%)")
        print(f"{'='*60}\n")

        return np.array(predictions)

    def save_results(self, filepath: str):
        """保存详细预测结果（包含LLM的reasoning）"""
        if hasattr(self, 'last_results'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": self.model,
                    "num_examples": len(self.examples),
                    "statistics": self.stats,
                    "results": self.last_results
                }, f, ensure_ascii=False, indent=2)
            print(f"✓ 详细结果已保存至: {filepath}")
        else:
            print("⚠️  没有可保存的结果，请先运行predict()")

    def save_model(self):
        """保存模型配置（符合统一接口）"""
        config = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "examples": self.examples,
            "stats": self.stats
        }

        filepath = "models/llm_in_context_config.json"
        os.makedirs("models", exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"✓ 模型配置已保存至: {filepath}")

    def load_model(self):
        """加载模型配置（符合统一接口）"""
        filepath = "models/llm_in_context_config.json"

        if not os.path.exists(filepath):
            print(f"⚠️  配置文件不存在: {filepath}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.provider = config.get("provider", self.provider)
        self.model = config.get("model", self.model)
        self.temperature = config.get("temperature", self.temperature)
        self.examples = config.get("examples", self.examples)

        print(f"✓ 已加载配置: {filepath}")

    def get_feature_vectors(self, titles: List[str]) -> np.ndarray:
        """
        获取特征向量（用于t-SNE可视化）

        使用Sentence-BERT提取标题的语义向量
        这些向量可以用于t-SNE降维可视化，展示标题在语义空间中的分布

        Args:
            titles: 标题列表

        Returns:
            embeddings: numpy数组，形状为(n, embedding_dim)
        """
        print("\n提取语义向量用于可视化...")

        # 延迟导入，避免未安装时影响其他功能
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("⚠️  sentence-transformers未安装，使用简单特征")
            print("   安装方法: pip install sentence-transformers")
            # 降级方案：使用TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            features = vectorizer.fit_transform(titles)
            return features.toarray()

        # 初始化或使用已有的embedding模型
        if not hasattr(self, 'embedding_model'):
            print("加载Sentence-BERT模型: all-MiniLM-L6-v2 (轻量级，384维)...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ 模型加载完成")

        # 批量提取嵌入向量
        print(f"正在提取 {len(titles)} 个标题的语义向量...")
        embeddings = self.embedding_model.encode(
            titles,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        print(f"✓ 向量提取完成，维度: {embeddings.shape}")
        return embeddings


def main():
    """示例：如何使用LLM In-Context Learning分类器"""

    print("="*60)
    print("LLM In-Context Learning 分类器 - 使用示例")
    print("="*60)

    # 初始化分类器
    # 方式1：使用OpenAI API（需要API key）
    classifier = LLMInContextClassifier(
        provider="openai",
        model="gpt-3.5-turbo",  # 或 "gpt-4"
        api_key=os.getenv("OPENAI_API_KEY"),  # 从环境变量读取
        temperature=0.0  # 确定性输出
    )

    # 测试数据
    test_titles = [
        "Deep Learning for Computer Vision: A Comprehensive Survey",
        "pp. 45-67 Abstract: This paper presents",
        "A Novel Method for Machine Translation......",
        "Neural Networks for Natural Language Processing",
        "Vol. 25, No. 3 Introduction to AI",
        "Attention Mechanism in Transformer Models",
        "Reference: Smith et al., 2020, Deep Learning",
    ]

    print("\n测试标题:")
    for i, title in enumerate(test_titles, 1):
        print(f"{i}. {title}")

    # 预测
    predictions = classifier.predict(test_titles, delay=0.5, verbose=True)

    # 打印结果
    print("\n" + "="*60)
    print("预测结果汇总")
    print("="*60)
    for i, (title, pred) in enumerate(zip(test_titles, predictions), 1):
        label_text = "✓ 正确标题" if pred == 1 else "✗ 错误标题"
        print(f"{i}. {label_text}: {title}")

    # 保存详细结果
    classifier.save_results("output/llm_predictions_detailed.json")

    # 保存配置
    classifier.save_model()


if __name__ == "__main__":
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  请设置OPENAI_API_KEY环境变量")
        print("示例: export OPENAI_API_KEY='your-api-key-here'")
        print("\n或者在代码中直接传入api_key参数")
    else:
        main()

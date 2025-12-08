"""
predownload_models.py
=====================
预下载所有BERT实验需要的模型到本地缓存
这样运行实验时就不需要从网络下载了

支持的模型:
1. bert-base-uncased (BERT baseline)
2. allenai/scibert_scivocab_uncased (SciBERT - 学术专用)
3. roberta-base (RoBERTa)
"""

import os
import sys
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch


# 所有需要的模型
REQUIRED_MODELS = {
    'bert-base': 'bert-base-uncased',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'roberta': 'roberta-base',
}


def download_model(model_name: str, model_path: str, retry_count: int = 3):
    """下载单个模型(tokenizer + model)"""

    print(f"\n{'='*80}")
    print(f" 📥 下载模型: {model_name}")
    print(f" 路径: {model_path}")
    print(f"{'='*80}")

    for attempt in range(1, retry_count + 1):
        try:
            print(f"\n尝试 {attempt}/{retry_count}...")

            # 1. 下载tokenizer
            print(f"  [1/3] 下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                force_download=False,  # 使用缓存
                resume_download=True   # 支持断点续传
            )
            print(f"  ✓ Tokenizer下载完成")

            # 2. 下载基础模型(用于特征提取)
            print(f"  [2/3] 下载基础模型...")
            base_model = AutoModel.from_pretrained(
                model_path,
                force_download=False,
                resume_download=True
            )
            print(f"  ✓ 基础模型下载完成")

            # 3. 下载分类模型
            print(f"  [3/3] 下载分类模型...")
            classification_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                force_download=False,
                resume_download=True
            )
            print(f"  ✓ 分类模型下载完成")

            # 检查模型大小
            model_size_mb = sum(
                p.numel() * p.element_size()
                for p in classification_model.parameters()
            ) / (1024 * 1024)
            print(f"  📊 模型大小: {model_size_mb:.2f} MB")
            print(f"  📊 参数量: {sum(p.numel() for p in classification_model.parameters()):,}")

            print(f"\n✅ {model_name} 下载成功!")
            return True

        except Exception as e:
            print(f"\n❌ 下载失败 (尝试 {attempt}/{retry_count}): {str(e)}")
            if attempt < retry_count:
                print(f"  ⏳ 5秒后重试...")
                import time
                time.sleep(5)
            else:
                print(f"\n❌ {model_name} 下载失败,已达最大重试次数")
                return False

    return False


def check_model_cached(model_path: str) -> bool:
    """检查模型是否已缓存"""
    try:
        # 尝试加载tokenizer (最快的检查方式)
        AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        return True
    except:
        return False


def get_cache_info():
    """获取HuggingFace缓存信息"""
    cache_dir = os.environ.get('HF_HOME') or os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
    print(f"\n📁 HuggingFace缓存目录: {cache_dir}")

    if os.path.exists(cache_dir):
        # 计算缓存大小
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass

        print(f"📊 当前缓存大小: {total_size / (1024**3):.2f} GB")
    else:
        print(f"⚠️  缓存目录不存在")


def main():
    """主函数"""

    print("\n" + "="*80)
    print(" 🚀 BERT模型预下载工具")
    print("="*80)

    # 显示缓存信息
    get_cache_info()

    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"\n⚠️  未检测到GPU,使用CPU")

    print(f"\n📋 需要下载的模型:")
    for idx, (name, path) in enumerate(REQUIRED_MODELS.items(), 1):
        cached = check_model_cached(path)
        status = "✓ 已缓存" if cached else "✗ 未缓存"
        print(f"  {idx}. {name:12s} -> {path:40s} [{status}]")

    # 询问用户
    print(f"\n{'='*80}")
    choice = input("是否开始下载? (y/n, 默认y): ").strip().lower()
    if choice and choice != 'y':
        print("取消下载")
        return

    # 下载所有模型
    results = {}
    success_count = 0
    total_count = len(REQUIRED_MODELS)

    for model_name, model_path in REQUIRED_MODELS.items():
        # 检查是否已缓存
        if check_model_cached(model_path):
            print(f"\n✓ {model_name} 已缓存,跳过下载")
            results[model_name] = 'cached'
            success_count += 1
            continue

        # 下载模型
        success = download_model(model_name, model_path, retry_count=3)
        results[model_name] = 'success' if success else 'failed'
        if success:
            success_count += 1

    # 打印总结
    print(f"\n\n{'='*80}")
    print(" 📊 下载总结")
    print(f"{'='*80}")

    for model_name, status in results.items():
        status_icon = {
            'cached': '✓ 已缓存',
            'success': '✓ 下载成功',
            'failed': '✗ 下载失败'
        }[status]
        print(f"  {model_name:12s}: {status_icon}")

    print(f"\n成功: {success_count}/{total_count}")

    if success_count == total_count:
        print("\n🎉 所有模型准备就绪!")
        print("现在可以运行实验了: python run_bert_experiments.py")
    else:
        print(f"\n⚠️  有 {total_count - success_count} 个模型下载失败")
        print("请检查网络连接后重试")

    # 显示最终缓存大小
    get_cache_info()


if __name__ == "__main__":
    main()

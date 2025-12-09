"""
测试模型加载是否正常
"""
import torch
import sys
sys.path.insert(0, '/home/u2023312337/task2/task2/stages/Stage4_BERT_Optimization')

from bert_classifier_optimized import OptimizedBERTClassifier

# 测试加载exp1
print("=" * 80)
print("测试 exp1_bert_base_baseline_best.pt")
print("=" * 80)

classifier1 = OptimizedBERTClassifier(
    model_name='bert-base',
    max_length=64,
    model_path='/home/u2023312337/task2/task2/stages/Stage4_BERT_Optimization/models/exp1_bert_base_baseline_best.pt'
)

# 获取初始化时的第一层权重
if hasattr(classifier1, 'model'):
    initial_weight = classifier1.model[0].embeddings.word_embeddings.weight[0][:5].clone()
    print(f"初始化后的权重前5个值: {initial_weight}")

# 加载模型
success = classifier1.load_model()
print(f"加载结果: {success}")
print(f"is_trained: {classifier1.is_trained}")

# 获取加载后的权重
if hasattr(classifier1, 'model'):
    loaded_weight = classifier1.model.bert.embeddings.word_embeddings.weight[0][:5]
    print(f"加载后的权重前5个值: {loaded_weight}")

    # 检查是否改变
    diff = (loaded_weight - initial_weight).abs().sum().item()
    print(f"权重差异: {diff}")

    if diff < 0.001:
        print("⚠️  警告：权重几乎没有变化，可能加载失败！")
    else:
        print("✓ 权重已改变，加载成功")

"""
demo_visualizer.py
==================
Stage1 可视化演示脚本
展示如何使用config.py来设置输出路径
"""

import sys
import os

# 导入config配置
from config import get_output_path, get_model_path, get_data_path
from visualizer import ResultVisualizer
from evaluator import ModelEvaluator
import numpy as np

print("=" * 60)
print("Stage1_Foundation 可视化演示")
print("=" * 60)
print(f"输出目录: {get_output_path('')}")
print(f"模型目录: {get_model_path('')}")
print(f"数据目录: {get_data_path('')}")
print("=" * 60)

# 创建示例数据
test_labels = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0] * 10)
predictions1 = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0] * 10)
predictions2 = np.array([1, 0, 1, 1, 0, 0, 1, 1, 1, 0] * 10)

# 评估模型
evaluator = ModelEvaluator()
result1 = evaluator.evaluate_model(test_labels, predictions1, "Model A", verbose=False)
result2 = evaluator.evaluate_model(test_labels, predictions2, "Model B", verbose=False)

results = [result1, result2]

# 可视化 - 使用config中的路径
visualizer = ResultVisualizer()
visualizer.plot_comparison(
    results, 
    save_path=get_output_path('demo_comparison.png')
)
visualizer.plot_confusion_matrices(
    results,
    save_path=get_output_path('demo_confusion.png')
)

print("\n✓ 演示完成!请检查 output/ 目录:")
print(f"  - {get_output_path('demo_comparison.png')}")
print(f"  - {get_output_path('demo_confusion.png')}")

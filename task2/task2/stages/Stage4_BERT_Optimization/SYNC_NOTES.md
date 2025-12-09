# Stage4 文件同步说明

## 日期
2025-12-08

## 同步内容

### 1. 核心文件完全同步
从主目录复制以下文件到 Stage4，确保逻辑完全一致：

- `run_bert_experiments.py` - BERT 实验批量运行脚本
- `train_bert_optimized_v2.py` - 优化的 BERT 分类器实现

### 2. 修改文件

**evaluate_stage4.py (行22)**
```python
# 修改前：
from bert_classifier_optimized import OptimizedBERTClassifier

# 修改后：
from train_bert_optimized_v2 import OptimizedBERTClassifier
```

### 3. 备份文件
创建了以下备份文件以防需要回滚：
- `run_bert_experiments.py.backup`
- `train_bert_optimized_v2.py.backup`

## 验证结果

✓ `run_bert_experiments.py` 与主目录完全一致
✓ `train_bert_optimized_v2.py` 与主目录完全一致
✓ `evaluate_stage4.py` 和 `run_bert_experiments.py` 都从 `train_bert_optimized_v2` 导入

## 关键差异说明

### bert_classifier_optimized.py vs train_bert_optimized_v2.py

| 特性 | bert_classifier_optimized.py | train_bert_optimized_v2.py |
|------|------------------------------|----------------------------|
| max_length 默认值 | 64 | 96 |
| 自定义分类头 | 支持 (use_custom_head) | 不支持 |
| dropout_rate 默认值 | 0.3 | 0.2 |
| Layer-wise LR | 无 | 有 |
| 对抗训练 | 无 | 有 |
| 混合精度 | 无 | 有 |

**结论**：`train_bert_optimized_v2.py` 包含更多高级优化功能，是训练和评估应该使用的版本。

## 现在可以运行

```bash
cd /home/u2023312337/task2/task2/stages/Stage4_BERT_Optimization

# 评估已训练的模型
python evaluate_stage4.py
```

## 注意事项

1. Stage4 的模型文件路径指向主目录：`/home/u2023312337/task2/task2/models/experiments/`
2. 模型文件已经训练好并复制到 Stage4
3. 所有文件现在使用相同的类定义，不会有加载冲突

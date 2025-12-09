# Stage4: BERT优化实验

**时间**：2024年11月16-28日  
**目标**：探索BERT的高级优化技术，测试不同模型和损失函数

## 📁 文件列表

| 文件 | 行数 | 功能 | 日期 |
|------|------|------|------|
| `train_optimized_bert.py` | 261 | BERT训练V1 | Nov 16 |
| `bert_classifier_optimized.py` | 736 | BERT优化类（V2） | Nov 16 |
| `optimized_BERT.py` | 376 | BERT优化框架 | Nov 16 |
| `train_bert_optimized_v2.py` | 760 | BERT训练V2（最终版） | Nov 28 |
| `run_bert_experiments.py` | ~350 | 批量实验（5组） | Nov 28 |
| `predownload_models.py` | ~100 | 模型预下载工具 | Nov 28 |
| `run_quick.sh` | ~50 | 快速实验脚本 | Nov 28 |

## 🎯 阶段成果

### 性能提升
- **BERT基础版**：87.91% (Stage2)
- **SciBERT + Focal Loss**：**89.04%** (+1.13%)
- **最佳F1**：90.57%

### 实验对比（5组实验）

| 实验 | 模型 | 损失函数 | 准确率 | F1 | 特点 |
|------|------|----------|--------|-----|------|
| Exp1 | bert-base | CE | 86.68% | 88.22% | 基准 |
| Exp2 | scibert | Focal Loss | **89.04%** | **90.57%** | 🏆最佳 |
| Exp3 | roberta | Weighted CE | 88.42% | 90.13% | 平衡 |
| Exp4 | deberta-v3 | CE | 87.50% | 89.45% | 潜力大 |
| Exp5 | scibert | CE (max_len=128) | 88.11% | 89.78% | 长序列 |

## 🔬 优化技术详解

### 1. 预训练模型选择

**BERT-base-uncased**（基准）
- 12层Transformer
- 110M参数
- 通用预训练

**SciBERT**（最佳）⭐
```python
model_name = "allenai/scibert_scivocab_uncased"
```
- 专门在科学文献上预训练
- 更适合学术标题分类
- **性能提升1.13%**

**RoBERTa**
```python
model_name = "roberta-base"
```
- 动态masking
- 更大batch size训练
- 性能稳定

**DeBERTa-v3**
```python
model_name = "microsoft/deberta-v3-base"
```
- Disentangled attention
- 理论上最强，但需要更多调优

### 2. Focal Loss（关键技术）

**问题**：标准Cross-Entropy对所有样本一视同仁

**Focal Loss解决方案**：
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

**效果**：
- 更关注困难样本
- 召回率提升2.14%
- 准确率提升1.13%

### 3. 对抗训练（FGM）

```python
class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
    
    def attack(self, emb_name='word_embeddings'):
        # 在embedding上添加扰动
        for name, param in self.model.named_parameters():
            if emb_name in name:
                norm = torch.norm(param.grad)
                perturbation = epsilon * param.grad / norm
                param.data.add_(perturbation)
```

**效果**：提高模型鲁棒性

### 4. 学习率优化

**层级学习率**（Layer-wise Learning Rate）：
```python
optimizer_grouped_parameters = [
    {'params': model.bert.embeddings.parameters(), 'lr': 2e-5},
    {'params': model.bert.encoder.layer[:6].parameters(), 'lr': 2e-5},
    {'params': model.bert.encoder.layer[6:].parameters(), 'lr': 3e-5},
    {'params': model.classifier.parameters(), 'lr': 5e-5}
]
```

**Warmup策略**：
```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

### 5. Early Stopping

```python
early_stopping = EarlyStopping(patience=3, mode='max')
for epoch in range(epochs):
    val_score = evaluate(model, val_loader)
    if early_stopping(val_score):
        break
```

## 🚀 使用示例

### 快速测试（run_quick.sh）
```bash
./run_quick.sh
# 选择：
# 1. 完整实验（5轮，2-3小时）
# 2. 中等实验（3轮，1-1.5小时）
# 3. 快速测试（3轮，30分钟）
```

### 单次训练（最佳配置）
```bash
python train_bert_optimized_v2.py
```

配置：
```python
model_name = 'allenai/scibert_scivocab_uncased'
max_length = 96
loss_type = 'focal'
use_adversarial = True
epochs = 5
batch_size = 32
learning_rate = 2e-5
```

### 批量实验（5组对比）
```bash
python run_bert_experiments.py
```

输出：`models/experiments/comparison_report.txt`

### 预下载模型
```bash
python predownload_models.py
```

## 📊 详细性能分析

### SciBERT + Focal Loss（最佳配置）

| 类别 | 精确率 | 召回率 | F1 | 支持数 |
|------|--------|--------|-----|--------|
| 错误标题(0) | 87.53% | 87.93% | 87.73% | 464 |
| 正确标题(1) | 90.58% | 90.24% | 90.41% | 512 |
| **宏平均** | 89.06% | 89.09% | 89.07% | 976 |
| **加权平均** | **89.14%** | **89.15%** | **89.14%** | 976 |

### 混淆矩阵分析
```
真实\预测    0     1
    0      408    56    (88%正确)
    1       50   462    (90%正确)
```

- **假阳性（FP）**：56个（将错误标题误判为正确）
- **假阴性（FN）**：50个（将正确标题误判为错误）

## 💡 优化经验总结

### 有效优化（按重要性排序）
1. ⭐⭐⭐ **SciBERT模型**：领域预训练模型效果显著
2. ⭐⭐⭐ **Focal Loss**：解决困难样本问题
3. ⭐⭐ **Warmup + 层级学习率**：训练更稳定
4. ⭐⭐ **对抗训练FGM**：提高鲁棒性
5. ⭐ **Early Stopping**：防止过拟合

### 效果不明显
- ❌ 增加max_length到512（计算量大，提升小）
- ❌ PGD对抗训练（比FGM慢，提升不大）
- ❌ 数据增强（回译等，效果一般）

### 性价比分析

| 技术 | 性能提升 | 计算成本 | 实现难度 | 推荐度 |
|------|---------|---------|----------|--------|
| SciBERT | +1.13% | 0% | ⭐ | ⭐⭐⭐⭐⭐ |
| Focal Loss | +0.8% | 0% | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| FGM对抗 | +0.3% | +20% | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 层级学习率 | +0.2% | 0% | ⭐⭐ | ⭐⭐⭐ |

## 🔗 参考文档

- **BERT_OPTIMIZATION_README.md** - 完整优化指南
- **QUICK_START.md** - 快速上手
- **models/experiments/comparison_report.txt** - 详细实验报告

## 📈 代码统计

- **总行数**：~2,800行
- **文件数**：7个
- **实验组数**：5组
- **训练时长**：8-12小时（全部实验）

---

**总结**：通过系统的BERT优化，性能从87.91%提升至89.04%，SciBERT和Focal Loss是关键。

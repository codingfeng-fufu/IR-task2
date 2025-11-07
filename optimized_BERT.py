"""
bert_classifier_optimized.py
==================
优化版BERT分类器 - 专注于BERT + 全部优化技巧
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os
import random


class TitleDataset(Dataset):
    """增强的数据集类"""

    def __init__(self, titles: List[str], labels: List[int], tokenizer,
                 max_length=64, augment=False):
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.titles)

    def augment_text(self, text: str) -> str:
        """文本增强：随机删除、交换"""
        words = text.split()
        if len(words) <= 2:
            return text

        # 10%概率随机删除一个词
        if random.random() < 0.1:
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)

        # 10%概率随机交换相邻词
        if random.random() < 0.1 and len(words) > 1:
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return ' '.join(words)

    def __getitem__(self, idx):
        title = self.titles[idx]
        label = self.labels[idx]

        # 训练时进行数据增强
        if self.augment:
            title = self.augment_text(title)

        encoding = self.tokenizer(
            title,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class FGM:
    """Fast Gradient Method 对抗训练"""

    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        """生成对抗样本"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EMA:
    """指数移动平均，提升模型稳定性"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class BERTClassifierOptimized:
    """
    优化版BERT分类器 - 专注于分类效果

    核心优化：
    1. 对抗训练（FGM）- 提升鲁棒性
    2. 指数移动平均（EMA）- 提升稳定性
    3. 差异化学习率 - 分类层用更大学习率
    4. Warmup + Cosine学习率调度 - 更好的收敛
    5. 验证集早停 - 防止过拟合
    6. 数据增强 - 扩充训练数据
    7. Focal Loss - 处理类别不平衡（可选）
    8. 梯度裁剪 - 防止梯度爆炸
    """

    def __init__(self, model_name='bert-base-uncased', max_length=64,
                 use_fgm=True, use_ema=True):
        """
        初始化优化版BERT分类器

        参数:
            model_name: BERT模型名称
                - 'bert-base-uncased': 英文标准BERT
                - 'bert-base-cased': 英文区分大小写BERT
                - 'bert-base-chinese': 中文BERT
            max_length: 最大序列长度
            use_fgm: 是否使用FGM对抗训练
            use_ema: 是否使用指数移动平均
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_fgm = use_fgm
        self.use_ema = use_ema

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # 加载tokenizer和模型
        print(f"\n加载BERT模型: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        ).to(self.device)

        # 初始化对抗训练和EMA
        self.fgm = FGM(self.model) if use_fgm else None
        self.ema = EMA(self.model) if use_ema else None

        self.is_trained = False
        self.best_model_path = 'best_bert_model.pt'

    def focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        """
        Focal Loss - 处理类别不平衡，专注于难分类样本

        参数:
            logits: 模型输出
            labels: 真实标签
            alpha: 平衡因子
            gamma: 聚焦参数（gamma越大，对易分类样本的权重越小）
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def train(self, train_titles: List[str], train_labels: List[int],
              val_titles: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None,
              epochs=10, batch_size=16, learning_rate=2e-5,
              warmup_ratio=0.1, weight_decay=0.01,
              patience=3, use_focal_loss=False,
              augment_data=True):
        """
        训练模型（全优化版）

        参数:
            train_titles: 训练标题
            train_labels: 训练标签
            val_titles: 验证标题（如果为None，会自动从训练集划分20%）
            val_labels: 验证标签
            epochs: 训练轮数（建议10-15轮）
            batch_size: 批次大小（16-32为佳，显存不足用8）
            learning_rate: 学习率（BERT推荐2e-5）
            warmup_ratio: 预热比例（0.1表示前10%的步数用于预热）
            weight_decay: 权重衰减（L2正则化）
            patience: 早停耐心值（验证F1多少轮不提升就停止）
            use_focal_loss: 是否使用Focal Loss（类别不平衡时推荐）
            augment_data: 是否数据增强
        """
        print("\n" + "=" * 70)
        print("训练优化版BERT分类器")
        print("=" * 70)
        print(f"模型: {self.model_name}")
        print(f"训练样本: {len(train_titles)}")
        print(f"轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {learning_rate}")
        print(f"\n优化技巧:")
        print(f"  ✓ 对抗训练(FGM): {self.use_fgm}")
        print(f"  ✓ 指数移动平均(EMA): {self.use_ema}")
        print(f"  ✓ 差异化学习率: 是")
        print(f"  ✓ Cosine学习率调度: 是")
        print(f"  ✓ 梯度裁剪: 是")
        print(f"  ✓ Focal Loss: {use_focal_loss}")
        print(f"  ✓ 数据增强: {augment_data}")
        print(f"  ✓ 验证早停: 是")

        # 如果没有提供验证集，自动划分20%
        if val_titles is None:
            from sklearn.model_selection import train_test_split
            train_titles, val_titles, train_labels, val_labels = train_test_split(
                train_titles, train_labels, test_size=0.2,
                random_state=42, stratify=train_labels
            )
            print(f"\n自动划分验证集:")
            print(f"  训练集: {len(train_titles)} 样本")
            print(f"  验证集: {len(val_titles)} 样本")

        # 创建数据集
        train_dataset = TitleDataset(
            train_titles, train_labels, self.tokenizer,
            self.max_length, augment=augment_data
        )
        val_dataset = TitleDataset(
            val_titles, val_labels, self.tokenizer,
            self.max_length, augment=False
        )

        train_loader = TorchDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = TorchDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # 差异化学习率：分类层（classifier）用更大的学习率
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            # 分类层，有weight decay
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'classifier' in n and not any(nd in n for nd in no_decay)],
                'lr': learning_rate * 10,  # 分类层用10倍学习率
                'weight_decay': weight_decay
            },
            # 分类层，无weight decay
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'classifier' in n and any(nd in n for nd in no_decay)],
                'lr': learning_rate * 10,
                'weight_decay': 0.0
            },
            # BERT层，有weight decay
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'classifier' not in n and not any(nd in n for nd in no_decay)],
                'lr': learning_rate,
                'weight_decay': weight_decay
            },
            # BERT层，无weight decay
            {
                'params': [p for n, p in self.model.named_parameters()
                           if 'classifier' not in n and any(nd in n for nd in no_decay)],
                'lr': learning_rate,
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

        # Warmup + Cosine学习率调度
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        print(f"\n学习率调度:")
        print(f"  总步数: {total_steps}")
        print(f"  Warmup步数: {warmup_steps}")
        print(f"  BERT层学习率: {learning_rate}")
        print(f"  分类层学习率: {learning_rate * 10}")

        # 训练循环
        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'=' * 70}")

            # ========== 训练阶段 ==========
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            progress_bar = tqdm(train_loader, desc="训练")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits

                # 计算损失
                if use_focal_loss:
                    loss = self.focal_loss(logits, labels)
                else:
                    loss = F.cross_entropy(logits, labels)

                # 反向传播
                loss.backward()

                # FGM对抗训练
                if self.use_fgm:
                    self.fgm.attack()
                    outputs_adv = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits_adv = outputs_adv.logits
                    if use_focal_loss:
                        loss_adv = self.focal_loss(logits_adv, labels)
                    else:
                        loss_adv = F.cross_entropy(logits_adv, labels)
                    loss_adv.backward()
                    self.fgm.restore()

                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA更新
                if self.use_ema:
                    self.ema.update()

                # 统计
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct / train_total:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            # ========== 验证阶段 ==========
            val_loss, val_acc, val_f1, val_preds, val_true = self.evaluate(
                val_loader, use_focal_loss
            )

            print(f"\nEpoch {epoch + 1} 结果:")
            print(f"  训练 - Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} ({train_acc * 100:.2f}%)")
            print(f"  验证 - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} ({val_acc * 100:.2f}%) | F1: {val_f1:.4f}")

            # 早停检查（基于F1分数）
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0

                # 保存最佳模型（使用EMA参数）
                if self.use_ema:
                    self.ema.apply_shadow()
                    torch.save(self.model.state_dict(), self.best_model_path)
                    self.ema.restore()
                else:
                    torch.save(self.model.state_dict(), self.best_model_path)

                print(f"  ✓ 最佳模型已保存! (F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  - 验证F1未提升 (当前最佳: {best_val_f1:.4f}, 耐心: {patience_counter}/{patience})")

                if patience_counter >= patience:
                    print(f"\n早停触发! 最佳验证F1: {best_val_f1:.4f}")
                    break

        # 加载最佳模型
        print(f"\n加载最佳模型 (验证F1: {best_val_f1:.4f})")
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.is_trained = True

        print("\n✓ 训练完成!")
        print(f"✓ 最佳验证F1分数: {best_val_f1:.4f}")
        return best_val_f1

    def evaluate(self, dataloader, use_focal_loss=False):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits

                if use_focal_loss:
                    loss = self.focal_loss(logits, labels)
                else:
                    loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = f1_score(all_labels, all_preds, average='binary')

        return avg_loss, accuracy, f1, all_preds, all_labels

    def predict(self, titles: List[str], batch_size=16) -> np.ndarray:
        """预测标签"""
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法")

        self.model.eval()
        predictions = []

        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="预测"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        return np.array(predictions)

    def predict_proba(self, titles: List[str], batch_size=16) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法")

        self.model.eval()
        probabilities = []

        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="计算概率"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
                probabilities.append(probs)

        return np.vstack(probabilities)

    def detailed_evaluation(self, titles: List[str], labels: List[int],
                            batch_size=16):
        """详细评估报告"""
        predictions = self.predict(titles, batch_size)
        probabilities = self.predict_proba(titles, batch_size)

        print("\n" + "=" * 70)
        print("详细评估报告")
        print("=" * 70)

        # 分类报告
        print("\n分类报告:")
        print(classification_report(labels, predictions,
                                    target_names=['错误标题', '正确标题'],
                                    digits=4))

        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        print("混淆矩阵:")
        print(f"              预测错误  预测正确")
        print(f"实际错误      {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"实际正确      {cm[1][0]:6d}    {cm[1][1]:6d}")

        # 各类指标
        accuracy = np.mean(predictions == labels)
        f1 = f1_score(labels, predictions, average='binary')

        print(f"\n总体指标:")
        print(f"  准确率(Accuracy): {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  F1分数: {f1:.4f}")

        # 显示一些预测示例
        print(f"\n预测示例（前10个）:")
        print(f"{'标题':<50} {'真实':<8} {'预测':<8} {'置信度':<10}")
        print("-" * 80)

        for i in range(min(10, len(titles))):
            title = titles[i][:47] + "..." if len(titles[i]) > 50 else titles[i]
            true_label = "正确" if labels[i] == 1 else "错误"
            pred_label = "正确" if predictions[i] == 1 else "错误"
            confidence = probabilities[i][predictions[i]]

            # 标记预测错误的样本
            marker = "❌" if predictions[i] != labels[i] else "✓"
            print(f"{marker} {title:<48} {true_label:<8} {pred_label:<8} {confidence:.3f}")

        return {
            'accuracy': accuracy,
            'f1': f1,
            'predictions': predictions,
            'probabilities': probabilities
        }


def main():
    """主函数：演示优化版BERT分类器"""
    from data_loader import DataLoader, create_sample_data

    print("=" * 70)
    print(" 优化版BERT分类器 - 效果优先")
    print("=" * 70)

    # 加载数据
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'positive_titles.txt',
        'negative_titles.txt',
        'test_titles.txt'
    )

    # 如果没有实际文件，使用示例数据
    if len(train_titles) == 0:
        print("\n未找到数据文件，使用示例数据进行演示...")
        train_titles, train_labels, test_titles, test_labels = create_sample_data()

    print(f"\n数据统计:")
    print(f"  训练样本: {len(train_titles)}")
    print(f"  测试样本: {len(test_titles)}")
    print(f"  正样本比例: {sum(train_labels) / len(train_labels):.2%}")
    print(f"  负样本比例: {(len(train_labels) - sum(train_labels)) / len(train_labels):.2%}")

    # 创建优化版分类器
    classifier = BERTClassifierOptimized(
        model_name='bert-base-uncased',
        max_length=64,
        use_fgm=True,  # 使用对抗训练
        use_ema=True  # 使用指数移动平均
    )

    # 训练（使用所有优化技巧）
    best_f1 = classifier.train(
        train_titles,
        train_labels,
        epochs=10,  # 充足的训练轮数
        batch_size=16,  # 合适的批次大小
        learning_rate=2e-5,  # BERT推荐学习率
        warmup_ratio=0.1,  # 10%步数用于warmup
        patience=3,  # 3轮不提升就早停
        use_focal_loss=False,  # 是否用Focal Loss（类别不平衡时设为True）
        augment_data=True  # 数据增强
    )

    # 在测试集上详细评估
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)

    results = classifier.detailed_evaluation(test_titles, test_labels)

    print(f"\n{'=' * 70}")
    print("最终结果总结:")
    print(f"  验证集最佳F1: {best_f1:.4f}")
    print(f"  测试集准确率: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"  测试集F1分数: {results['f1']:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
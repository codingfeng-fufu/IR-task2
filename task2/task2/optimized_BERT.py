"""
bert_classifier_optimized.py (增强版)
==================
优化版BERT分类器 - 添加了完整的模型保存/加载功能
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os
import random
import json


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
    9. 完整的模型保存/加载 - 避免重复训练
    """

    def __init__(self, model_name='bert-base-uncased', max_length=64,
                 use_fgm=True, use_ema=True, model_dir='models/bert'):
        """
        初始化优化版BERT分类器

        参数:
            model_name: BERT模型名称
            max_length: 最大序列长度
            use_fgm: 是否使用FGM对抗训练
            use_ema: 是否使用指数移动平均
            model_dir: 模型保存目录
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_fgm = use_fgm
        self.use_ema = use_ema
        self.model_dir = model_dir

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"使用设备: {self.device} ({torch.cuda.get_device_name(0)})")
            print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print(f"使用设备: {self.device}")

        # 加载tokenizer和模型
        print("加载BERT分词器...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        print("加载BERT分类模型...")
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        ).to(self.device)
        
        # 用于提取特征的模型(不带分类头)
        self.feature_model = BertModel.from_pretrained(model_name).to(self.device)

        # 初始化对抗训练和EMA
        self.fgm = FGM(self.model) if use_fgm else None
        self.ema = EMA(self.model) if use_ema else None

        self.is_trained = False

    def focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        """Focal Loss - 处理类别不平衡"""
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
              augment_data=True, save_model=True):
        """训练模型（全优化版）"""
        print("\n" + "="*60)
        print("训练 BERT 分类器（优化版v2）")
        print("="*60)
        print(f"模型: {self.model_name}")
        print(f"训练样本数: {len(train_titles)}")
        print(f"轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {learning_rate}")
        print(f"预热比例: {warmup_ratio}")
        
        # 如果没有提供验证集，从训练集划分20%
        if val_titles is None:
            from sklearn.model_selection import train_test_split
            train_titles, val_titles, train_labels, val_labels = train_test_split(
                train_titles, train_labels, test_size=0.2, random_state=42, stratify=train_labels
            )
            print(f"自动划分验证集: {len(val_titles)} 样本")

        # 创建数据集
        train_dataset = TitleDataset(
            train_titles, train_labels, self.tokenizer,
            self.max_length, augment=augment_data
        )
        val_dataset = TitleDataset(
            val_titles, val_labels, self.tokenizer,
            self.max_length, augment=False
        )

        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 差异化学习率
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'classifier' in n],
                'lr': learning_rate * 10,  # 分类层用更大学习率
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 'classifier' not in n],
                'lr': learning_rate,
                'weight_decay': weight_decay
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

        # 学习率调度
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(f"总训练步数: {total_steps}")
        print(f"预热步数: {warmup_steps}")

        # 早停
        best_val_f1 = 0.0
        patience_counter = 0

        # 训练循环
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")

            self.model.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            progress_bar = tqdm(train_loader, desc=f"训练中")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss if not use_focal_loss else self.focal_loss(outputs.logits, labels)
                logits = outputs.logits

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 对抗训练
                if self.use_fgm:
                    self.fgm.attack()
                    outputs_adv = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss_adv = outputs_adv.loss if not use_focal_loss else self.focal_loss(outputs_adv.logits, labels)
                    loss_adv.backward()
                    self.fgm.restore()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # EMA更新
                if self.use_ema:
                    self.ema.update()

                # 统计
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_predictions:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # Epoch完成统计
            avg_loss = total_loss / len(train_loader)
            accuracy = correct_predictions / total_predictions

            print(f"\nEpoch {epoch + 1} 完成:")
            print(f"  - 平均损失: {avg_loss:.4f}")
            print(f"  - 训练准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  - 当前学习率: {scheduler.get_last_lr()[0]:.2e}")

            # 验证集评估
            if self.use_ema:
                self.ema.apply_shadow()
            
            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_loader, use_focal_loss)
            
            if self.use_ema:
                self.ema.restore()

            print(f"  - 验证损失: {val_loss:.4f}")
            print(f"  - 验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"  - 验证F1分数: {val_f1:.4f}")

            # 早停检查
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0

                # 保存最佳模型（临时）
                if self.use_ema:
                    self.ema.apply_shadow()
                    temp_path = os.path.join(self.model_dir, 'best_model_temp.pt')
                    os.makedirs(self.model_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), temp_path)
                    self.ema.restore()
                else:
                    temp_path = os.path.join(self.model_dir, 'best_model_temp.pt')
                    os.makedirs(self.model_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), temp_path)

                print(f"  ✓ F1改善（最佳F1: {best_val_f1:.4f}）")
            else:
                patience_counter += 1
                print(f"  - F1未改善（最佳F1: {best_val_f1:.4f}）")

                if patience_counter >= patience:
                    print(f"\n早停触发! 最佳验证F1: {best_val_f1:.4f}")
                    break

        # 加载最佳模型
        print(f"\n加载最佳模型 (验证F1: {best_val_f1:.4f})")
        temp_path = os.path.join(self.model_dir, 'best_model_temp.pt')
        self.model.load_state_dict(torch.load(temp_path))
        self.is_trained = True

        print("\n✓ BERT训练完成!")
        
        # 保存完整模型
        if save_model:
            self.save_model()
        
        return best_val_f1

    def save_model(self):
        """保存完整模型（模型权重 + tokenizer + 配置）"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 保存模型权重
        model_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        torch.save(self.model.state_dict(), model_path)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(self.model_dir)
        
        # 保存配置
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'use_fgm': self.use_fgm,
            'use_ema': self.use_ema,
            'is_trained': self.is_trained
        }
        config_path = os.path.join(self.model_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ 完整模型已保存至: {self.model_dir}")
        print(f"  - 模型权重: {model_path}")
        print(f"  - Tokenizer: {self.model_dir}")
        print(f"  - 配置文件: {config_path}")
    
    def load_model(self):
        """加载完整模型"""
        config_path = os.path.join(self.model_dir, 'config.json')
        model_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            print(f"⚠️  模型文件不存在:")
            if not os.path.exists(config_path):
                print(f"  - 缺失: {config_path}")
            if not os.path.exists(model_path):
                print(f"  - 缺失: {model_path}")
            return False
        
        print(f"加载模型: {self.model_dir}")
        
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.model_name = config['model_name']
        self.max_length = config['max_length']
        self.use_fgm = config['use_fgm']
        self.use_ema = config['use_ema']
        self.is_trained = config['is_trained']
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        
        # 重新初始化FGM和EMA
        self.fgm = FGM(self.model) if self.use_fgm else None
        self.ema = EMA(self.model) if self.use_ema else None
        
        print("✓ 模型加载成功!")
        return True

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
            raise ValueError("模型尚未训练!请先调用train()方法或load_model()方法")

        self.model.eval()
        predictions = []

        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="预测中"):
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
            raise ValueError("模型尚未训练!请先调用train()方法或load_model()方法")

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

    def get_feature_vectors(self, titles: List[str], batch_size=16) -> np.ndarray:
        """
        获取BERT的特征向量([CLS] token的嵌入)
        用于可视化
        
        参数:
            titles: 标题列表
            batch_size: 批次大小
            
        返回:
            特征矩阵
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法或load_model()方法")
        
        self.feature_model.eval()
        embeddings = []
        
        dummy_labels = [0] * len(titles)
        dataset = TitleDataset(titles, dummy_labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取特征"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.feature_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 使用[CLS] token的嵌入(第一个token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)


def main():
    """主函数：演示优化版BERT分类器"""
    from data_loader import DataLoader, create_sample_data

    print("="*70)
    print(" 优化版BERT分类器演示（增强版 - 支持保存/加载）")
    print("="*70)

    # 加载数据
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'positive_titles.txt',
        'negative_titles.txt',
        'test_titles.txt'
    )

    # 如果没有实际文件，使用示例数据
    if len(train_titles) == 0:
        train_titles, train_labels, test_titles, test_labels = create_sample_data()

    # 为了演示，只使用部分数据
    print("\n注意: 为了演示，只使用部分数据")
    train_titles = train_titles[:200]
    train_labels = train_labels[:200]
    test_titles = test_titles[:50]
    test_labels = test_labels[:50]

    # 创建分类器
    classifier = BERTClassifierOptimized(
        model_name='bert-base-uncased',
        max_length=64,
        use_fgm=True,
        use_ema=True,
        model_dir='models/bert'
    )

    # 尝试加载已有模型
    if not classifier.load_model():
        # 如果没有已有模型，则训练新模型
        classifier.train(
            train_titles,
            train_labels,
            epochs=2,
            batch_size=8,
            learning_rate=2e-5,
            save_model=True
        )
    else:
        print("使用已加载的模型进行预测")

    # 进行预测
    print("\n" + "="*60)
    print("在测试集上进行预测")
    print("="*60)

    predictions = classifier.predict(test_titles, batch_size=8)
    probabilities = classifier.predict_proba(test_titles, batch_size=8)

    # 显示一些预测结果
    print("\n预测结果示例:")
    print(f"{'标题':<50} {'真实':<8} {'预测':<8} {'置信度':<10}")
    print("-" * 80)

    for i in range(min(10, len(test_titles))):
        title = test_titles[i][:47] + "..." if len(test_titles[i]) > 50 else test_titles[i]
        true_label = "正确" if test_labels[i] == 1 else "错误"
        pred_label = "正确" if predictions[i] == 1 else "错误"
        confidence = probabilities[i][predictions[i]]

        print(f"{title:<50} {true_label:<8} {pred_label:<8} {confidence:.3f}")

    # 计算准确率
    accuracy = np.mean(predictions == test_labels)
    print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()
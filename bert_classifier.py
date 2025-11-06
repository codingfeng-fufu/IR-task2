"""
bert_classifier.py
==================
BERT分类器实现
使用预训练的BERT模型进行序列分类
"""

import numpy as np
import torch
from typing import List, Dict
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW
from tqdm import tqdm


class TitleDataset(Dataset):
    """PyTorch数据集,用于BERT模型"""
    
    def __init__(self, titles: List[str], labels: List[int], tokenizer, max_length=64):
        """
        初始化数据集
        
        参数:
            titles: 标题列表
            labels: 标签列表
            tokenizer: BERT分词器
            max_length: 最大序列长度
        """
        self.titles = titles
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.titles)
    
    def __getitem__(self, idx):
        title = self.titles[idx]
        label = self.labels[idx]
        
        # 使用BERT分词器编码
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


class BERTClassifier:
    """
    基于BERT的分类器
    使用预训练的BERT模型进行微调
    """
    
    def __init__(self, model_name='bert-base-uncased', max_length=64):
        """
        初始化BERT分类器

        参数:
            model_name: 预训练BERT模型名称
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # 设置设备(GPU或CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"使用设备: {self.device} ({torch.cuda.get_device_name(0)})")
            print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print(f"使用设备: {self.device}")
        
        # 加载分词器
        print("加载BERT分词器...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # 加载分类模型
        print("加载BERT分类模型...")
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,  # 二分类
            output_attentions=False,
            output_hidden_states=False
        ).to(self.device)

        # 用于提取特征的模型(不带分类头)
        self.feature_model = BertModel.from_pretrained(model_name).to(self.device)
        
        # 训练状态
        self.is_trained = False
        
    def train(self, titles: List[str], labels: List[int],
              epochs=5, batch_size=32, learning_rate=2e-5, warmup_steps=500):
        """
        微调BERT模型（优化版v2）

        参数:
            titles: 训练标题列表
            labels: 训练标签列表
            epochs: 训练轮数（增加到5）
            batch_size: 批次大小（保持32以获得更好的泛化）
            learning_rate: 学习率
            warmup_steps: 学习率预热步数
        """
        print("\n" + "="*60)
        print("训练 BERT 分类器（优化版v2）")
        print("="*60)
        print(f"模型: {self.model_name}")
        print(f"训练样本数: {len(titles)}")
        print(f"轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {learning_rate}")
        print(f"预热步数: {warmup_steps}")

        # 创建数据集和数据加载器
        dataset = TitleDataset(titles, labels, self.tokenizer, self.max_length)
        dataloader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # 设置优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # 设置学习率调度器（带预热）
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(f"总训练步数: {total_steps}")

        # 记录最佳损失（用于监控，但不早停）
        best_loss = float('inf')
        
        # 训练循环
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # 使用tqdm显示进度条
            progress_bar = tqdm(dataloader, desc=f"训练中")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 将数据移到设备上
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # 更新学习率

                # 统计
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)

                # 更新进度条（显示当前学习率）
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions/total_predictions:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # 计算平均损失和准确率
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions

            print(f"\nEpoch {epoch + 1} 完成:")
            print(f"  - 平均损失: {avg_loss:.4f}")
            print(f"  - 训练准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  - 当前学习率: {scheduler.get_last_lr()[0]:.2e}")

            # 记录最佳损失（不触发早停）
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  ✓ 损失改善（最佳损失: {best_loss:.4f}）")
            else:
                print(f"  - 损失未改善（最佳损失: {best_loss:.4f}）")
        
        self.is_trained = True
        print("\n✓ BERT训练完成!")
    
    def predict(self, titles: List[str], batch_size=16) -> np.ndarray:
        """
        预测给定标题的标签
        
        参数:
            titles: 待分类的标题列表
            batch_size: 批次大小
            
        返回:
            预测标签数组 (0=错误标题, 1=正确标题)
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练!请先调用train()方法")
        
        self.model.eval()
        predictions = []
        
        # 创建数据加载器(不需要标签)
        dummy_labels = [0] * len(titles)  # 虚拟标签
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
                
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
        
        return np.array(predictions)
    
    def predict_proba(self, titles: List[str], batch_size=16) -> np.ndarray:
        """
        预测给定标题的概率
        
        参数:
            titles: 待分类的标题列表
            batch_size: 批次大小
            
        返回:
            概率数组,形状为 (n_samples, 2)
        """
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
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
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
            raise ValueError("模型尚未训练!请先调用train()方法")
        
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
    """
    主函数:演示BERT分类器的使用
    """
    from data_loader import DataLoader, create_sample_data
    
    print("="*70)
    print(" BERT 分类器演示")
    print("="*70)
    
    # 加载数据
    train_titles, train_labels, test_titles, test_labels = DataLoader.prepare_dataset(
        'positive_titles.txt',
        'negative_titles.txt',
        'test_titles.txt'
    )
    
    # 如果没有实际文件,使用示例数据
    if len(train_titles) == 0:
        train_titles, train_labels, test_titles, test_labels = create_sample_data()
    
    # 为了演示,只使用部分数据(BERT训练较慢)
    print("\n注意: 为了演示,只使用部分数据")
    train_titles = train_titles[:200]
    train_labels = train_labels[:200]
    test_titles = test_titles[:50]
    test_labels = test_labels[:50]
    
    # 创建并训练分类器
    classifier = BERTClassifier(model_name='bert-base-uncased', max_length=64)
    classifier.train(
        train_titles, 
        train_labels,
        epochs=2,  # 较少的epoch用于演示
        batch_size=8,
        learning_rate=2e-5
    )
    
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

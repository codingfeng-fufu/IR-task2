"""
evaluator.py
============
模型评估模块
计算各种评估指标并生成评估报告
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class ModelEvaluator:
    """
    模型评估器
    用于评估和比较不同分类模型的性能
    """
    
    @staticmethod
    def evaluate_model(
        y_true: List[int], 
        y_pred: List[int], 
        model_name: str,
        verbose: bool = True
    ) -> Dict:
        """
        评估单个模型的性能
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称
            verbose: 是否打印详细信息
            
        返回:
            包含评估指标的字典
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f" {model_name} - 评估结果")
            print(f"{'='*70}")
        
        # ===== 计算基本指标 =====
        accuracy = accuracy_score(y_true, y_pred)
        
        # 计算精确率、召回率和F1分数(针对正类,即label=1)
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # 计算宏平均和微平均F1分数
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # 计算每个类别的精确率、召回率和F1
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        if verbose:
            # 打印整体指标
            print("\n【整体指标】")
            print(f"  准确率 (Accuracy):     {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  精确率 (Precision):    {precision:.4f}")
            print(f"  召回率 (Recall):       {recall:.4f}")
            print(f"  F1分数 (F1-Score):     {f1:.4f}")
            print(f"  F1宏平均 (F1-Macro):   {f1_macro:.4f}")
            print(f"  F1微平均 (F1-Micro):   {f1_micro:.4f}")
            
            # 打印每个类别的指标
            print("\n【各类别指标】")
            class_names = ['负样本(错误标题)', '正样本(正确标题)']
            print(f"{'类别':<20} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'样本数':<10}")
            print("-" * 70)
            
            for i, class_name in enumerate(class_names):
                support = np.sum(np.array(y_true) == i)
                print(f"{class_name:<20} {precision_per_class[i]:<12.4f} "
                      f"{recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f} {support:<10}")
            
            # 打印混淆矩阵
            print("\n【混淆矩阵】")
            header = "实际\\预测"
            print(f"{header:<15} {'预测为负':<15} {'预测为正':<15}")
            print("-" * 50)
            print(f"{'实际为负':<15} {cm[0][0]:<15} {cm[0][1]:<15}")
            print(f"{'实际为正':<15} {cm[1][0]:<15} {cm[1][1]:<15}")
            
            # 计算并打印更多指标
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 等同于召回率
            
            print("\n【附加指标】")
            print(f"  真负例 (TN):           {tn}")
            print(f"  假正例 (FP):           {fp}")
            print(f"  假负例 (FN):           {fn}")
            print(f"  真正例 (TP):           {tp}")
            print(f"  特异度 (Specificity):  {specificity:.4f}")
            print(f"  敏感度 (Sensitivity):  {sensitivity:.4f}")
            
            # 打印分类报告
            print("\n【详细分类报告】")
            print(classification_report(
                y_true, 
                y_pred, 
                target_names=class_names,
                digits=4
            ))
        
        # 返回结果字典
        return {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'specificity': specificity,
            'sensitivity': sensitivity
        }
    
    @staticmethod
    def compare_models(results_list: List[Dict]) -> None:
        """
        比较多个模型的性能
        
        参数:
            results_list: 包含多个评估结果的列表
        """
        print("\n" + "="*70)
        print(" 模型性能对比")
        print("="*70)
        
        # 准备对比表格
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'f1_micro']
        metric_names = {
            'accuracy': '准确率',
            'precision': '精确率',
            'recall': '召回率',
            'f1': 'F1分数',
            'f1_macro': 'F1宏平均',
            'f1_micro': 'F1微平均'
        }
        
        # 打印表头
        header = f"{'指标':<15}"
        for result in results_list:
            header += f"{result['model']:<20}"
        print(header)
        print("-" * (15 + 20 * len(results_list)))
        
        # 打印每个指标
        for metric in metrics:
            row = f"{metric_names[metric]:<15}"
            values = []
            for result in results_list:
                value = result[metric]
                values.append(value)
                row += f"{value:<20.4f}"
            print(row)
            
            # 标记最佳值
            best_idx = np.argmax(values)
            best_model = results_list[best_idx]['model']
            print(f"{'  ↑ 最佳':<15}{best_model}")
            print()
        
        # 总结
        print("\n【综合评价】")
        
        # 计算综合得分(简单平均)
        for result in results_list:
            score = np.mean([result[m] for m in metrics])
            print(f"  {result['model']}: 综合得分 = {score:.4f}")
        
        # 找出最佳模型
        best_result = max(results_list, key=lambda x: np.mean([x[m] for m in metrics]))
        print(f"\n  推荐模型: {best_result['model']}")
    
    @staticmethod
    def calculate_error_analysis(
        y_true: List[int],
        y_pred: List[int],
        titles: List[str]
    ) -> Dict:
        """
        进行错误分析
        
        参数:
            y_true: 真实标签
            y_pred: 预测标签
            titles: 标题文本
            
        返回:
            错误分析结果
        """
        # 找出错误预测
        errors = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                errors.append({
                    'index': i,
                    'title': titles[i],
                    'true_label': y_true[i],
                    'pred_label': y_pred[i],
                    'error_type': 'FP' if y_pred[i] == 1 else 'FN'
                })
        
        # 统计错误类型
        fp_count = sum(1 for e in errors if e['error_type'] == 'FP')
        fn_count = sum(1 for e in errors if e['error_type'] == 'FN')
        
        return {
            'total_errors': len(errors),
            'false_positives': fp_count,
            'false_negatives': fn_count,
            'error_details': errors
        }
    
    @staticmethod
    def print_error_analysis(error_analysis: Dict, max_examples: int = 10):
        """
        打印错误分析结果
        
        参数:
            error_analysis: 错误分析结果
            max_examples: 最多显示的错误示例数量
        """
        print("\n" + "="*70)
        print(" 错误分析")
        print("="*70)
        
        print(f"\n总错误数: {error_analysis['total_errors']}")
        print(f"  假正例 (FP - 错误预测为正确): {error_analysis['false_positives']}")
        print(f"  假负例 (FN - 正确预测为错误): {error_analysis['false_negatives']}")
        
        if error_analysis['total_errors'] > 0:
            print(f"\n【错误示例】(显示前{max_examples}个)")
            print(f"{'类型':<6} {'标题':<50} {'真实':<8} {'预测':<8}")
            print("-" * 80)
            
            for i, error in enumerate(error_analysis['error_details'][:max_examples]):
                error_type = error['error_type']
                title = error['title'][:47] + "..." if len(error['title']) > 50 else error['title']
                true_label = "正确" if error['true_label'] == 1 else "错误"
                pred_label = "正确" if error['pred_label'] == 1 else "错误"
                
                print(f"{error_type:<6} {title:<50} {true_label:<8} {pred_label:<8}")


def main():
    """
    主函数:演示评估器的使用
    """
    from data_loader import create_sample_data
    
    print("="*70)
    print(" 模型评估器演示")
    print("="*70)
    
    # 创建示例数据
    _, _, test_titles, test_labels = create_sample_data()
    
    # 模拟三个模型的预测结果
    np.random.seed(42)
    
    # 模型1: 90% 准确率
    pred1 = test_labels.copy()
    flip_indices = np.random.choice(len(pred1), size=int(0.1 * len(pred1)), replace=False)
    for idx in flip_indices:
        pred1[idx] = 1 - pred1[idx]
    
    # 模型2: 85% 准确率
    pred2 = test_labels.copy()
    flip_indices = np.random.choice(len(pred2), size=int(0.15 * len(pred2)), replace=False)
    for idx in flip_indices:
        pred2[idx] = 1 - pred2[idx]
    
    # 模型3: 95% 准确率
    pred3 = test_labels.copy()
    flip_indices = np.random.choice(len(pred3), size=int(0.05 * len(pred3)), replace=False)
    for idx in flip_indices:
        pred3[idx] = 1 - pred3[idx]
    
    # 评估每个模型
    evaluator = ModelEvaluator()
    
    result1 = evaluator.evaluate_model(test_labels, pred1, "模型A (朴素贝叶斯)")
    result2 = evaluator.evaluate_model(test_labels, pred2, "模型B (Word2Vec+SVM)")
    result3 = evaluator.evaluate_model(test_labels, pred3, "模型C (BERT)")
    
    # 比较模型
    evaluator.compare_models([result1, result2, result3])
    
    # 错误分析(以模型2为例)
    error_analysis = evaluator.calculate_error_analysis(test_labels, pred2, test_titles)
    evaluator.print_error_analysis(error_analysis, max_examples=10)


if __name__ == "__main__":
    main()

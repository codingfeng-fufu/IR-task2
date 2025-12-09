"""
修复模型checkpoint中的is_trained标志
"""
import torch
import os
import glob

model_dir = '/home/u2023312337/task2/task2/models/experiments'
model_files = glob.glob(os.path.join(model_dir, '*.pt'))

print(f"找到 {len(model_files)} 个模型文件")

for model_path in model_files:
    print(f"\n处理: {os.path.basename(model_path)}")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 检查is_trained
        old_value = checkpoint.get('is_trained', 'NOT_FOUND')
        print(f"  旧值: is_trained = {old_value}")
        
        # 修改is_trained为True
        checkpoint['is_trained'] = True
        
        # 保存
        torch.save(checkpoint, model_path)
        print(f"  ✓ 已修改为: is_trained = True")
        
    except Exception as e:
        print(f"  ✗ 失败: {e}")

print("\n完成!")

"""
config.py
==========
Stage1_Foundation 配置文件
定义本阶段的输出路径
"""

import os

# 获取当前脚本所在目录
STAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# 输出目录配置
OUTPUT_DIR = os.path.join(STAGE_DIR, 'output')
MODELS_DIR = os.path.join(STAGE_DIR, 'models')

# 数据目录(使用项目根目录的data)
DATA_DIR = os.path.join(STAGE_DIR, '..', '..', 'data')

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 辅助函数
def get_output_path(filename):
    """获取输出文件的完整路径"""
    return os.path.join(OUTPUT_DIR, filename)

def get_model_path(filename):
    """获取模型文件的完整路径"""
    return os.path.join(MODELS_DIR, filename)

def get_data_path(filename):
    """获取数据文件的完整路径"""
    return os.path.join(DATA_DIR, filename)

if __name__ == '__main__':
    print(f"Stage Directory: {STAGE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Data Directory: {DATA_DIR}")

#!/usr/bin/env python3
"""
run_from_stages.py
==================
在 stages 目录运行脚本的辅助工具
自动设置 Python 路径，使各阶段的模块可以互相导入
"""

import sys
import os
from pathlib import Path

# 获取 stages 目录的绝对路径
STAGES_DIR = Path(__file__).parent.absolute()

# 添加所有包含 Python 模块的目录到 sys.path
module_dirs = [
    STAGES_DIR / "Stage1_Foundation",
    STAGES_DIR / "Stage2_Traditional_Models",
    STAGES_DIR / "Stage3_NaiveBayes_Optimization",
    STAGES_DIR / "Stage4_BERT_Optimization",
    STAGES_DIR / "Stage5_LLM_Framework",
    STAGES_DIR / "Main_Scripts",
    STAGES_DIR / "Utils",
]

for dir_path in module_dirs:
    if dir_path.exists() and str(dir_path) not in sys.path:
        sys.path.insert(0, str(dir_path))

print("✓ Python 路径已配置")
print(f"✓ Stages 目录: {STAGES_DIR}")
print("\n可用的运行命令:")
print("  python run_from_stages.py Main_Scripts/main_pipeline.py")
print("  python run_from_stages.py Main_Scripts/evaluate_saved.py")
print("  python run_from_stages.py Stage1_Foundation/check_environment.py")
print("  python run_from_stages.py Stage3_NaiveBayes_Optimization/test_optimized_nb.py")
print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python run_from_stages.py <脚本路径>")
        print("\n示例:")
        print("  python run_from_stages.py Main_Scripts/main_pipeline.py")
        sys.exit(1)

    script_path = STAGES_DIR / sys.argv[1]

    if not script_path.exists():
        print(f"✗ 错误: 脚本不存在: {script_path}")
        sys.exit(1)

    # 保持工作目录在 stages 根目录（而非子目录）
    # 这样相对路径 data/, models/, output/ 可以正常访问
    os.chdir(STAGES_DIR)

    print(f"运行脚本: {script_path.relative_to(STAGES_DIR)}")
    print(f"工作目录: {STAGES_DIR}")
    print("=" * 80)
    print()

    # 读取并执行脚本
    with open(script_path, 'r', encoding='utf-8') as f:
        script_code = f.read()

    # 修改 sys.argv，移除第一个参数（run_from_stages.py）
    sys.argv = [str(script_path)] + sys.argv[2:]

    # 执行脚本
    exec(compile(script_code, str(script_path), 'exec'), {'__name__': '__main__', '__file__': str(script_path)})

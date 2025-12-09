#!/usr/bin/env python3
"""
环境检查脚本 - 验证服务器环境是否配置正确
"""

import sys
import os

def check_python_version():
    """检查 Python 版本"""
    print("=" * 60)
    print("检查 Python 版本...")
    print("=" * 60)
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python 版本符合要求 (>= 3.8)")
        return True
    else:
        print("✗ Python 版本过低，需要 >= 3.8")
        return False

def check_packages():
    """检查必需的包"""
    print("\n" + "=" * 60)
    print("检查必需的包...")
    print("=" * 60)
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'gensim': 'Gensim',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm',
        'openpyxl': 'OpenPyXL'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = module.__version__
            print(f"✓ {name:20s} {version}")
        except ImportError:
            print(f"✗ {name:20s} 未安装")
            all_ok = False
    
    return all_ok

def check_cuda():
    """检查 CUDA 和 GPU"""
    print("\n" + "=" * 60)
    print("检查 CUDA 和 GPU...")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}:")
                print(f"  名称: {torch.cuda.get_device_name(i)}")
                print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
                print(f"  计算能力: {props.major}.{props.minor}")
                
                # 测试 GPU 内存
                try:
                    torch.cuda.set_device(i)
                    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
                    print(f"  可用内存: {free_mem:.2f} GB / {total_mem:.2f} GB")
                except:
                    pass
            
            print("\n✓ GPU 环境正常")
            return True
        else:
            print("⚠️  警告: CUDA 不可用，将使用 CPU 训练")
            print("   这会显著增加训练时间（特别是 BERT）")
            return False
            
    except ImportError:
        print("✗ PyTorch 未安装")
        return False
    except Exception as e:
        print(f"✗ 检查 CUDA 时出错: {str(e)}")
        return False

def check_data_files():
    """检查数据文件"""
    print("\n" + "=" * 60)
    print("检查数据文件...")
    print("=" * 60)
    
    files = {
        'data/positive.txt': '正样本训练集',
        'data/negative.txt': '负样本训练集',
        'data/testSet-1000.xlsx': '测试集'
    }
    
    all_ok = True
    for filepath, description in files.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024**2  # MB
            print(f"✓ {description:20s} {filepath:30s} ({size:.2f} MB)")
        else:
            print(f"✗ {description:20s} {filepath:30s} 不存在")
            all_ok = False
    
    return all_ok

def check_output_dir():
    """检查输出目录"""
    print("\n" + "=" * 60)
    print("检查输出目录...")
    print("=" * 60)
    
    if not os.path.exists('output'):
        os.makedirs('output')
        print("✓ 创建 output/ 目录")
    else:
        print("✓ output/ 目录已存在")
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("✓ 创建 logs/ 目录")
    else:
        print("✓ logs/ 目录已存在")
    
    return True

def test_gpu_computation():
    """测试 GPU 计算"""
    print("\n" + "=" * 60)
    print("测试 GPU 计算...")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠️  跳过 GPU 测试（CUDA 不可用）")
            return True
        
        # 简单的矩阵乘法测试
        print("执行简单的 GPU 计算测试...")
        device = torch.device('cuda')
        
        # 创建测试张量
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # 执行计算
        import time
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ GPU 计算测试通过 (耗时: {elapsed*1000:.2f} ms)")
        
        # 测试 BERT tokenizer
        print("\n测试 BERT tokenizer...")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        test_text = "This is a test sentence for BERT tokenizer."
        tokens = tokenizer(test_text, return_tensors='pt')
        print(f"✓ BERT tokenizer 测试通过")
        print(f"  输入: {test_text}")
        print(f"  Token IDs: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU 计算测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("学术标题分类系统 - 环境检查")
    print("=" * 60)
    
    results = []
    
    # 执行所有检查
    results.append(("Python 版本", check_python_version()))
    results.append(("必需的包", check_packages()))
    results.append(("CUDA 和 GPU", check_cuda()))
    results.append(("数据文件", check_data_files()))
    results.append(("输出目录", check_output_dir()))
    results.append(("GPU 计算", test_gpu_computation()))
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ 所有检查通过！环境配置正确。")
        print("\n可以开始训练:")
        print("  bash run_training.sh")
        print("或:")
        print("  python main_pipeline.py")
        return 0
    else:
        print("\n✗ 部分检查失败，请根据上述信息修复问题。")
        print("\n常见解决方案:")
        print("1. 安装缺失的包: pip install -r requirements_server.txt")
        print("2. 安装 PyTorch (CUDA): pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121")
        print("3. 检查数据文件是否在 data/ 目录下")
        return 1

if __name__ == '__main__':
    sys.exit(main())


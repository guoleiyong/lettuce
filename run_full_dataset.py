# run_full_dataset.py
"""
一键运行完整数据集训练的脚本
"""

import os
import sys
import subprocess
import time


def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn',
        'matplotlib', 'seaborn', 'Pillow', 'albumentations', 'pyyaml'
    ]

    print("检查依赖包...")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package} (未安装)")

    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        install = input("是否自动安装? (y/n): ")
        if install.lower() == 'y':
            for package in missing_packages:
                print(f"正在安装 {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print("请手动安装缺少的依赖包:")
            print(f"pip install {' '.join(missing_packages)}")
            return False

    return True


def check_dataset():
    """检查完整数据集是否存在"""
    print("\n检查数据集...")

    required_files = [
        'Lettuce_MTD_Full/metadata.csv',
        'Lettuce_MTD_Full/plants/'
    ]

    missing_files = []

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ✗ {file_path} (未找到)")

    if missing_files:
        print(f"\n错误: 以下文件或目录未找到:")
        for file in missing_files:
            print(f"  {file}")

        print("\n请确保完整数据集已放置在正确位置:")
        print("1. 将 Lettuce_MTD_Full 文件夹放在项目根目录")
        print("2. 确保 Lettuce_MTD_Full/metadata.csv 存在")
        print("3. 确保 Lettuce_MTD_Full/plants/ 目录存在")

        return False

    # 检查图片数量
    import pandas as pd
    df = pd.read_csv('Lettuce_MTD_Full/metadata.csv')
    print(f"  数据集统计: {len(df)} 条记录")
    print(f"  植物数量: {df['plant_id'].nunique()} 株")
    print(f"  标签分布:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"    {label}: {count}")

    return True


def run_training():
    """运行训练"""
    print("\n" + "=" * 60)
    print("开始训练STMCEN模型（完整数据集）")
    print("=" * 60)

    start_time = time.time()

    try:
        # 导入训练模块
        from scripts.train_model import main as train_main

        # 运行训练
        history, results = train_main()

        end_time = time.time()
        training_time = end_time - start_time

        print(f"\n训练完成!")
        print(f"总训练时间: {training_time / 60:.1f} 分钟")

        return True

    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_plots():
    """生成图表"""
    print("\n生成论文图表...")

    try:
        from scripts.generate_plots import generate_all_plots
        generate_all_plots()
        print("图表生成完成!")
        return True
    except Exception as e:
        print(f"生成图表时出现错误: {e}")
        return False


def main():
    """主函数"""
    print("STMCEN完整数据集训练系统")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        return

    # 检查数据集
    if not check_dataset():
        return

    # 创建输出目录
    os.makedirs("outputs_full", exist_ok=True)
    os.makedirs("checkpoints_full", exist_ok=True)

    # 询问用户操作
    print("\n选择操作:")
    print("1. 训练模型")
    print("2. 生成图表")
    print("3. 全部执行")

    choice = input("请输入选择 (1/2/3): ").strip()

    if choice == '1' or choice == '3':
        if not run_training():
            return

    if choice == '2' or choice == '3':
        generate_plots()

    print("\n" + "=" * 60)
    print("程序执行完成!")
    print("=" * 60)

    print("\n输出文件位置:")
    if os.path.exists("outputs_full"):
        print("  图表和报告: outputs_full/")
    if os.path.exists("checkpoints_full"):
        print("  模型检查点: checkpoints_full/")


if __name__ == "__main__":
    main()
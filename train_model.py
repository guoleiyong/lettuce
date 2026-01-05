# scripts/train_model.py
import torch
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.dataset import create_dataloaders
from utils.trainer import Trainer
from models.stmcen import STMCEN


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataloaders_full(config):
    """为完整数据集准备数据加载器"""
    print("准备完整数据集...")

    metadata_path = config['paths']['metadata']
    image_dir = config['paths']['images_dir']

    # 检查文件是否存在
    if not os.path.exists(metadata_path):
        print(f"错误: 找不到元数据文件 {metadata_path}")
        print("请确保完整数据集已放置在正确位置")
        return None, None, None

    print(f"元数据文件: {metadata_path}")
    print(f"图片目录: {image_dir}")

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=metadata_path,
        image_dir=image_dir,
        config=config
    )

    return train_loader, val_loader, test_loader


def main():
    """主训练函数"""

    # 加载配置
    config_path = 'configs/config.yaml'
    config = load_config(config_path)

    # 设置随机种子
    torch.manual_seed(config['experiment']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['experiment']['seed'])

    # 设备设置
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 准备数据
    print("准备数据...")
    train_loader, val_loader, test_loader = prepare_dataloaders_full(config)

    if train_loader is None:
        print("数据准备失败，退出程序")
        return

    print(f"训练集批次: {len(train_loader)}")
    print(f"验证集批次: {len(val_loader)}")
    print(f"测试集批次: {len(test_loader)}")

    # 创建模型
    print("创建模型...")
    model = STMCEN(
        sensor_dim=config['model']['sensor_dim'],
        num_classes=config['model']['num_classes'],
        window_size=config['model']['window_size'],
        hidden_dim=config['model']['hidden_dim']
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建训练器
    trainer = Trainer(model, config, device)

    # 训练
    print("开始训练...")
    history = trainer.train(train_loader, val_loader, test_loader)

    # 测试
    print("测试模型...")
    test_results = trainer.test(test_loader)

    # 打印测试结果
    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"准确率: {test_results['accuracy']:.4f}")
    print(f"精确率: {test_results['precision']:.4f}")
    print(f"召回率: {test_results['recall']:.4f}")
    print(f"F1分数: {test_results['f1']:.4f}")
    print("=" * 50)

    # 保存结果到文件
    results_file = os.path.join(config['paths']['output_dir'], 'training_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("STMCEN模型训练结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据集: {config['paths']['metadata']}\n")
        f.write(f"训练轮数: {len(history['train_loss'])}\n")
        f.write(f"最佳验证准确率: {max(history['val_acc']):.4f}\n\n")
        f.write("测试结果:\n")
        f.write(f"准确率: {test_results['accuracy']:.4f}\n")
        f.write(f"精确率: {test_results['precision']:.4f}\n")
        f.write(f"召回率: {test_results['recall']:.4f}\n")
        f.write(f"F1分数: {test_results['f1']:.4f}\n")

    print(f"结果已保存到: {results_file}")

    return history, test_results


if __name__ == "__main__":
    main()
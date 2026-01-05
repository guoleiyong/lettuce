# scripts/test_model.py
import torch
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.dataset import LettuceDataset
from models.stmcen import STMCEN
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


def test_model():
    """测试模型"""

    # 加载配置
    config_path = 'configs/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建测试数据集
    test_dataset = LettuceDataset(
        metadata_path=config['paths']['metadata'],
        image_dir=config['paths']['images_dir'],
        window_size=config['model']['window_size'],
        mode='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0  # 在Windows上可能需要设置为0
    )

    # 创建模型
    model = STMCEN(
        sensor_dim=config['model']['sensor_dim'],
        num_classes=config['model']['num_classes'],
        window_size=config['model']['window_size'],
        hidden_dim=config['model']['hidden_dim']
    )

    # 加载预训练权重（如果有）
    checkpoint_path = os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载检查点: {checkpoint_path}")
    else:
        print("未找到检查点，使用随机初始化的模型")

    model.to(device)
    model.eval()

    # 测试
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            sensors = batch['sensors'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(images, sensors)
            preds = torch.argmax(outputs['logits'], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n测试结果:")
    print(f"准确率: {accuracy:.4f}")
    print("混淆矩阵:")
    print(cm)

    # 打印每个类别的准确率
    class_names = ['healthy', 'drought', 'low_light']
    for i in range(len(class_names)):
        class_acc = np.diag(cm)[i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        print(f"{class_names[i]}准确率: {class_acc:.4f}")

    return accuracy, cm


if __name__ == "__main__":
    test_model()
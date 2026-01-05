import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalCrossModalCausalAttention(nn.Module):
    """双向跨模态因果注意力模块"""

    def __init__(self, sensor_dim=4, visual_dim=1280, hidden_dim=256):
        super().__init__()

        # 路径A: 传感器→图像 (环境引导视觉)
        self.sensor_to_visual = nn.ModuleDict({
            'sensor_query': nn.Linear(sensor_dim, hidden_dim),
            'visual_key': nn.Conv2d(visual_dim, hidden_dim, 1),
            'visual_value': nn.Conv2d(visual_dim, hidden_dim, 1),
        })

        # 路径B: 图像→传感器 (视觉校准环境)
        self.visual_to_sensor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sensor_dim),
            nn.Softmax(dim=-1)
        )

        # 输出投影
        self.visual_proj = nn.Conv2d(hidden_dim, visual_dim, 1)

    def forward(self, sensor_features, visual_features):
        """
        参数:
            sensor_features: (B, sensor_dim) 传感器特征
            visual_features: (B, C, H, W) 视觉特征图
        """
        batch_size, C, H, W = visual_features.shape

        # 路径A: 传感器→图像
        # 1. 环境查询
        sensor_query = self.sensor_to_visual['sensor_query'](sensor_features)  # (B, hidden_dim)
        sensor_query = sensor_query.unsqueeze(-1).unsqueeze(-1)  # (B, hidden_dim, 1, 1)

        # 2. 视觉键值
        visual_key = self.sensor_to_visual['visual_key'](visual_features)  # (B, hidden_dim, H, W)
        visual_value = self.sensor_to_visual['visual_value'](visual_features)  # (B, hidden_dim, H, W)

        # 3. 计算注意力权重
        attention_scores = F.cosine_similarity(
            sensor_query,
            visual_key.flatten(2).unsqueeze(1),
            dim=1
        )  # (B, H*W)

        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, H*W)
        attention_weights = attention_weights.view(batch_size, 1, H, W)  # (B, 1, H, W)

        # 4. 调制视觉特征
        modulated_visual = visual_value * attention_weights  # (B, hidden_dim, H, W)
        modulated_visual = self.visual_proj(modulated_visual)  # (B, C, H, W)

        # 全局池化得到特征向量
        visual_vector = F.adaptive_avg_pool2d(modulated_visual, 1).squeeze(-1).squeeze(-1)  # (B, C)

        # 路径B: 图像→传感器
        # 1. 计算传感器重要性权重
        sensor_weights = self.visual_to_sensor(visual_features)  # (B, sensor_dim)

        # 2. 重校准传感器特征
        recalibrated_sensor = sensor_features * sensor_weights  # (B, sensor_dim)

        return {
            'visual_features': visual_vector,
            'sensor_features': recalibrated_sensor,
            'attention_weights': attention_weights,
            'sensor_weights': sensor_weights
        }
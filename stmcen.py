import torch
import torch.nn as nn
import torchvision.models as models
from .bi_cca import BidirectionalCrossModalCausalAttention
from .gstf import GatedSpatioTemporalFusion


class STMCEN(nn.Module):
    """时空多模态因果可解释网络"""

    def __init__(self, sensor_dim=4, num_classes=3, window_size=5, hidden_dim=256):
        super().__init__()

        self.window_size = window_size

        # 图像特征提取器 (MobileNetV2)
        self.image_encoder = models.mobilenet_v2(pretrained=True)
        self.image_encoder.classifier = nn.Identity()

        # 获取中间层特征
        self.visual_feature_layer = self.image_encoder.features[14]  # block_6_expand层

        # 传感器特征提取器
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # 双向跨模态因果注意力
        self.bi_cca = BidirectionalCrossModalCausalAttention(
            sensor_dim=64,
            visual_dim=96,  # MobileNetV2 block_6_expand的输出通道数
            hidden_dim=hidden_dim
        )

        # 门控时空融合
        self.gstf = GatedSpatioTemporalFusion(
            input_dim=96 + 64,  # 图像特征 + 传感器特征
            hidden_dim=hidden_dim
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(96 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def extract_visual_features(self, x):
        """提取视觉特征"""
        # 提取全局特征
        global_features = self.image_encoder(x)

        # 提取中间层空间特征
        spatial_features = None
        for i, layer in enumerate(self.image_encoder.features):
            x = layer(x)
            if i == 14:  # block_6_expand层
                spatial_features = x

        return global_features, spatial_features

    def forward(self, images, sensors):
        """
        参数:
            images: (B, T, C, H, W) 时序图像
            sensors: (B, T, sensor_dim) 时序传感器数据
        """
        batch_size, T, C, H, W = images.shape

        # 存储每个时间步的特征
        visual_features_list = []
        sensor_features_list = []

        # 处理每个时间步
        for t in range(T):
            # 提取图像特征
            img_global, img_spatial = self.extract_visual_features(images[:, t])

            # 提取传感器特征
            sensor_feat = self.sensor_encoder(sensors[:, t])

            # 双向跨模态因果注意力
            bi_cca_output = self.bi_cca(sensor_feat, img_spatial)

            visual_features_list.append(bi_cca_output['visual_features'])
            sensor_features_list.append(bi_cca_output['sensor_features'])

        # 拼接特征
        visual_features = torch.stack(visual_features_list, dim=1)  # (B, T, visual_dim)
        sensor_features = torch.stack(sensor_features_list, dim=1)  # (B, T, sensor_dim)

        # 拼接多模态特征
        multimodal_features = torch.cat([visual_features, sensor_features], dim=-1)  # (B, T, total_dim)

        # 门控时空融合
        fused_features, gstf_info = self.gstf(multimodal_features)  # (B, total_dim)

        # 分类
        logits = self.classifier(fused_features)

        return {
            'logits': logits,
            'attention_weights': bi_cca_output['attention_weights'],
            'sensor_weights': bi_cca_output['sensor_weights'],
            'gate_weights': gstf_info['gate_weights'],
            'multimodal_features': multimodal_features,
            'fused_features': fused_features
        }
import torch
import torch.nn as nn


class GatedSpatioTemporalFusion(nn.Module):
    """门控时空融合模块"""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        # GRU用于时序编码
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        # 门控机制
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, sequence_features):
        """
        参数:
            sequence_features: (B, T, D) 时序特征序列
        """
        batch_size, T, D = sequence_features.shape

        # 1. GRU时序编码
        gru_out, hidden = self.gru(sequence_features)  # gru_out: (B, T, hidden_dim)
        last_hidden = gru_out[:, -1, :]  # (B, hidden_dim)

        # 2. 门控融合
        current_features = sequence_features[:, -1, :]  # (B, D)

        # 拼接历史信息和当前信息
        combined = torch.cat([last_hidden, current_features], dim=-1)  # (B, hidden_dim + D)

        # 计算门控权重
        gate_weights = self.gate_fc(combined)  # (B, hidden_dim)

        # 门控融合
        fused_features = last_hidden * gate_weights + current_features * (1 - gate_weights)

        # 输出投影
        output = self.output_proj(fused_features)  # (B, D)

        return output, {
            'gate_weights': gate_weights,
            'last_hidden': last_hidden,
            'gru_outputs': gru_out
        }
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN(nn.Module):
    def __init__(self, channels_list):

        channels_list: [2, 2, 2] #对应 P3, P4, P5

        super().__init__()

        # 上采样分支（高层特征向低层融合）
        self.conv_p5_to_p4 = nn.Conv2d(channels_list[2], channels_list[1], kernel_size=1)
        self.conv_p4_to_p3 = nn.Conv2d(channels_list[1], channels_list[0], kernel_size=1)

        # 可学习权重（用于加权融合）
        self.weight_p4 = nn.Parameter(torch.ones(2))  # P4和P5->P4的权重
        self.weight_p3 = nn.Parameter(torch.ones(2))  # P3和P4->P3的权重

    def forward(self, p3, p4, p5):
        # --- 上采样分支 ---
        # 第一步：P5 -> P4
        p5_up = F.interpolate(self.conv_p5_to_p4(p5), scale_factor=2, mode='nearest')
        p4_fused = (self.weight_p4[0] * p4 + self.weight_p4[1] * p5_up) / (self.weight_p4.sum() + 1e-4)

        # 第二步：P4 -> P3
        p4_up = F.interpolate(self.conv_p4_to_p3(p4_fused), scale_factor=2, mode='nearest')
        p3_fused = (self.weight_p3[0] * p3 + self.weight_p3[1] * p4_up) / (self.weight_p3.sum() + 1e-4)

        return p3_fused, p4_fused, p5
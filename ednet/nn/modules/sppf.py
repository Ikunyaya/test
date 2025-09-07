import torch
import torch.nn as nn
from .base import Conv  # 从base导入

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    """
    def __init__(self, c1, c2=None, k=5):  # 添加 c2 参数以兼容 YOLO 的调用方式
        super().__init__()
        # 如果只传递了一个参数，将其视为 c1
        if c2 is None:
            c2 = c1
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_ * 4, c2, k=1, s=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        """前向传播"""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
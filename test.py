import torch
import matplotlib.pyplot as plt
from ednet.nn.modules.block import CBAM


def visualize_cbam_attention():
    # 创建一个合理的测试图像（3通道）
    test_image = torch.randn(1, 3, 256, 256)  # 改为3通道而不是1通道

    # 创建CBAM模块，确保通道数匹配
    cbam = CBAM(3)  # 通道数改为3

    # 前向传播
    output = cbam(test_image)

    # 计算差异
    diff = output - test_image
    print("Max difference between input and output:", diff.max().item())

    # 可视化输入和输出
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(test_image[0, 0, :, :].detach().numpy(), cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    axs[1].imshow(output[0, 0, :, :].detach().numpy(), cmap='gray')
    axs[1].set_title('Output after CBAM')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


# 运行可视化
visualize_cbam_attention()
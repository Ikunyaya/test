# test_bifpn.py
import sys

sys.path.append('D:/object_detection/EDNet-main')

try:
    from ednet.nn.modules import BiFPN

    print("✅ BiFPN导入成功！")

    # 测试初始化
    bifpn = BiFPN([160, 320, 640])
    print("✅ BiFPN初始化成功！")
    print(f"BiFPN参数数量: {sum(p.numel() for p in bifpn.parameters())}")

except ImportError as e:
    print(f"❌ 导入失败: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")
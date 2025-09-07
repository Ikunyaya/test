# val.py
from ednet import EDNet
from multiprocessing import freeze_support


def main():
    # 加载预训练模型 (可选择 t, n, s, m, b, l, x)
    model = EDNet("runs/detect/train44/weights/best.pt")

    # 评估模型
    model.val(data="visdrone-det.yaml", split='val')

    # 预测 (更改路径为您的图片路径)
    model('ednet/assets/test1.jpg')


if __name__ == '__main__':
    # 确保多进程支持
    freeze_support()
    # 执行主函数
    main()

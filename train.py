from ednet import EDNet

if __name__ == "__main__":
    # 添加以下两行（可选但建议）
    import multiprocessing

    multiprocessing.freeze_support()

    model = EDNet("ednet-t.yaml")
    #model.train(data="visdrone-det.yaml", epochs=1, imgsz=640, workers=2, batch=8, device=0)
    model.train(
        data="visdrone-det.yaml",
        epochs=200,  # 保持
        imgsz=640,  # 保持
        batch=4,  # 增大batch size（A100可支持）
        workers=4,  # 增加数据加载线程
        device=0,
        lr0=0.01,  # 保持
        lrf=0.01,  # 保持
        cos_lr=True,
        label_smoothing=0.1,
        box=7.5,  # 改为WIoU损失（需修改源码）
        cls=0.5,  # 降低分类损失权重
        dfl=1.5,  # 增加分布焦点损失
        warmup_epochs=3,
        fliplr=0.5,  # 增加水平翻转
        mixup=0.2,  # 增加mixup
        copy_paste=0.3,  # 增加小目标复制
        optimizer='SGD',
        weight_decay=0.0005,  # 降低正则化强度
        amp=True,  # 启用混合精度
        freeze=[0],  # 仅冻结backbone前1层
        overlap_mask=True,  # 启用重叠掩码
        mask_ratio=4,  # 增加掩码分辨率
        close_mosaic=10,  # 最后10轮关闭马赛克增强
    )



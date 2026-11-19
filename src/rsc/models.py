"""模型构建：U-Net 系列（segmentation_models_pytorch）+ SegFormer。

为什么要两个：
  * U-Net + ResNet/EfficientNet 编码器：小数据也能训，边界准，是可靠的下限；
  * SegFormer（MiT 主干 + 轻量 MLP 解码头）：Transformer 的全局感受野对
    "大面积同类地物"（农田、水面、连片屋顶）更稳，是论文里的 SOTA 对照。

两个模型对外接口完全一致：``forward(x) -> logits(B, C, H, W)``，训练代码不必关心用的是谁。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp

    _HAS_SMP = True
except Exception:  # pragma: no cover
    _HAS_SMP = False


# --------------------------------------------------------------------------
# SegFormer 包装：把 1/4 分辨率的 logits 上采样回输入尺寸
# --------------------------------------------------------------------------
class SegFormerWrapper(nn.Module):
    """transformers 版 SegFormer。

    HuggingFace 的 ``SegformerForSemanticSegmentation`` 直接输出 stride=4 的 logits，
    必须自己插值回原图，否则 loss 和标签形状对不上（新手最常踩的坑）。
    """

    def __init__(self, num_classes: int = 6, backbone: str = "nvidia/mit-b2",
                 pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                backbone,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            cfg = SegformerConfig(num_labels=num_classes)
            self.net = SegformerForSemanticSegmentation(cfg)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(pixel_values=x)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------
# 统一工厂
# --------------------------------------------------------------------------
SMP_ARCHS = {
    "unet": "Unet",
    "unetpp": "UnetPlusPlus",
    "deeplabv3plus": "DeepLabV3Plus",
    "pspnet": "PSPNet",
    "fpn": "FPN",
    "manet": "MAnet",
    "pan": "PAN",
    "linknet": "Linknet",
}

# 6GB 显存（RTX 2060/3060）上的实用组合：参数量 / 推荐 crop / batch
ENCODER_HINTS = {
    "resnet34": "轻量首选，crop512/bs8 可跑，迭代快",
    "resnet50": "精度略高，显存 +30%",
    "resnet101": "6GB 需 crop384/bs4 或冻结前几层",
    "efficientnet-b0": "精度/速度平衡好",
    "efficientnet-b4": "Vaihingen 上的经典 SOTA 组合（crop512/bs4 + AMP）",
    "mit_b2": "SegFormer-B2，crop512/bs4 + AMP 可跑",
}


def build_model(
    arch: str = "unet",
    encoder: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    num_classes: int = 6,
    pretrained: bool = True,
) -> nn.Module:
    """构建模型。

    arch: unet / unetpp / deeplabv3plus / pspnet / fpn / segformer
    encoder: ResNet/EfficientNet/MiT 名称（见 ``ENCODER_HINTS``）
    """
    arch = arch.lower()
    if arch == "segformer":
        return SegFormerWrapper(num_classes, backbone=encoder if encoder.startswith("nvidia/")
                                else f"nvidia/{encoder}", pretrained=pretrained,
                                in_channels=in_channels)

    if not _HAS_SMP:
        raise ImportError("需要 pip install segmentation-models-pytorch")

    if arch not in SMP_ARCHS:
        raise KeyError(f"未知架构 {arch!r}，可选 {list(SMP_ARCHS) + ['segformer']}")
    cls = getattr(smp, SMP_ARCHS[arch])
    return cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: nn.Module) -> str:
    n = count_params(model)
    return f"{model.__class__.__name__}: {n / 1e6:.2f} M 可训练参数"

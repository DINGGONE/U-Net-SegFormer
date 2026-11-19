"""Segmentation losses: CE / Dice / Focal and combinations."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """soft Dice（按类计算后取平均），对类别不平衡友好。

    只在"该类别在本 batch 出现过"的类上取平均 —— 否则一个 batch 里没有裸土，
    裸土的 Dice 会被算成 0，把整体 loss 拉高，训练信号就乱了。
    """

    def __init__(self, num_classes: int = 6, ignore_index: int = 255,
                 smooth: float = 1.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = logits.softmax(dim=1)
        valid = (target != self.ignore_index) if self.ignore_index is not None else None
        scores, present = [], []
        for c in range(self.num_classes):
            p = prob[:, c]
            t = (target == c).float()
            if valid is not None:
                p, t = p[valid], t[valid]
            present.append(float(t.sum() > 0))
            inter = (p * t).sum()
            denom = p.sum() + t.sum()
            scores.append((2 * inter + self.smooth) / (denom + self.smooth))
        d = torch.stack(scores)
        w = torch.tensor(present, device=d.device)
        if self.weight is not None:
            w = w * self.weight.to(d.device)
        if w.sum() < 1e-6:          # 整个 batch 全是忽略区
            return d.sum() * 0.0
        return 1 - (d * w).sum() / w.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        valid = target != self.ignore_index
        return loss[valid].mean()


class CombinedLoss(nn.Module):
    """``ce_weight * CE + dice_weight * Dice``，可选类别权重与标签平滑。"""

    def __init__(
        self,
        num_classes: int = 6,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        focal: bool = False,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        if focal:
            self.ce = FocalLoss(2.0, class_weights, ignore_index)
        else:
            self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index,
                                          label_smoothing=label_smoothing)
        self.dice = DiceLoss(num_classes, ignore_index, weight=class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * self.ce(logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss


def build_loss(cfg: dict, class_weights: Optional[torch.Tensor] = None,
               device: str = "cpu") -> CombinedLoss:
    """从配置字典构造损失：支持 ce / dice / ce_dice / focal_dice。"""
    name = cfg.get("name", "ce_dice")
    kw = dict(
        num_classes=cfg.get("num_classes", 6),
        class_weights=class_weights.to(device) if class_weights is not None else None,
        ignore_index=cfg.get("ignore_index", 255),
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )
    if name == "ce":
        return CombinedLoss(ce_weight=1.0, dice_weight=0.0, **kw)
    if name == "dice":
        return CombinedLoss(ce_weight=0.0, dice_weight=1.0, **kw)
    if name == "focal_dice":
        return CombinedLoss(ce_weight=1.0, dice_weight=1.0, focal=True, **kw)
    return CombinedLoss(ce_weight=cfg.get("ce_weight", 1.0),
                        dice_weight=cfg.get("dice_weight", 1.0), **kw)

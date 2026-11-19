"""Confusion matrix and segmentation metrics: mIoU / F1 / OA / Kappa."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes: int = 6, ignore_index: Optional[int] = 255):
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    # ------------------------------------------------------------------
    def update(self, pred: np.ndarray, gt: np.ndarray) -> "ConfusionMatrix":
        """pred/gt: 同形状整型数组（HxW 或 NxHxW），值为类别 id。"""
        pred = np.asarray(pred).ravel()
        gt = np.asarray(gt).ravel()
        assert pred.size == gt.size, "pred 与 gt 元素个数不一致"
        if self.ignore_index is not None:
            keep = (gt >= 0) & (gt < self.num_classes)
            pred, gt = pred[keep], gt[keep]
        # 越界的预测只可能是 bug，钳到 0 而不是静默丢弃
        pred = np.clip(pred, 0, self.num_classes - 1)
        idx = gt.astype(np.int64) * self.num_classes + pred.astype(np.int64)
        self.mat += np.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )
        return self

    def merge(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        """多卡 / 多进程评估时合并混淆矩阵。"""
        self.mat += other.mat
        return self

    def reset(self) -> None:
        self.mat[:] = 0

    # ------------------------------------------------------------------
    # 基础量
    # ------------------------------------------------------------------
    @property
    def tp(self) -> np.ndarray:
        return np.diag(self.mat)

    @property
    def fp(self) -> np.ndarray:
        return self.mat.sum(axis=0) - self.tp

    @property
    def fn(self) -> np.ndarray:
        return self.mat.sum(axis=1) - self.tp

    @property
    def support(self) -> np.ndarray:
        """各类在 GT 中的像素数。"""
        return self.mat.sum(axis=1)

    def iou(self, eps: float = 1e-9) -> np.ndarray:
        denom = self.tp + self.fp + self.fn
        iou = self.tp / np.maximum(denom, eps)
        iou[denom == 0] = np.nan
        return iou

    def f1(self, eps: float = 1e-9) -> np.ndarray:
        prec = self.tp / np.maximum(self.tp + self.fp, eps)
        rec = self.tp / np.maximum(self.tp + self.fn, eps)
        f1 = 2 * prec * rec / np.maximum(prec + rec, eps)
        f1[(self.tp + self.fp + self.fn) == 0] = np.nan
        return f1

    def precision(self, eps: float = 1e-9) -> np.ndarray:
        return self.tp / np.maximum(self.tp + self.fp, eps)

    def recall(self, eps: float = 1e-9) -> np.ndarray:
        return self.tp / np.maximum(self.tp + self.fn, eps)

    def oa(self) -> float:
        total = self.mat.sum()
        return float(self.tp.sum() / total) if total else float("nan")

    def miou(self, ignore_absent: bool = True) -> float:
        iou = self.iou()
        if ignore_absent:
            iou = iou[~np.isnan(iou)]
        return float(np.nanmean(iou))

    def mf1(self, ignore_absent: bool = True) -> float:
        f1 = self.f1()
        if ignore_absent:
            f1 = f1[~np.isnan(f1)]
        return float(np.nanmean(f1))

    def kappa(self) -> float:
        total = self.mat.sum()
        if not total:
            return float("nan")
        po = self.tp.sum() / total
        pe = float((self.mat.sum(axis=0) * self.mat.sum(axis=1)).sum()) / (total * total)
        return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")

    # ------------------------------------------------------------------
    def summary(
        self,
        names: Sequence[str] = None,
        round_to: int = 4,
    ) -> Dict:
        iou, f1 = self.iou(), self.f1()
        per_class = {}
        for i in range(self.num_classes):
            name = names[i] if names and i < len(names) else f"class_{i}"
            per_class[name] = {
                "iou": None if np.isnan(iou[i]) else round(float(iou[i]), round_to),
                "f1": None if np.isnan(f1[i]) else round(float(f1[i]), round_to),
                "precision": round(float(self.precision()[i]), round_to),
                "recall": round(float(self.recall()[i]), round_to),
                "gt_pixels": int(self.support[i]),
                "pred_pixels": int(self.mat[:, i].sum()),
            }
        return {
            "mIoU": round(self.miou(), round_to),
            "mF1": round(self.mf1(), round_to),
            "OA": round(self.oa(), round_to),
            "Kappa": round(self.kappa(), round_to),
            "per_class": per_class,
            "confusion_matrix": self.mat.tolist(),
        }

    def format_table(self, names: Sequence[str] = None) -> str:
        """打印成可直接贴进报告的表。"""
        iou, f1 = self.iou(), self.f1()
        p, r = self.precision(), self.recall()
        head = f"{'类别':<10}{'IoU':>8}{'F1':>8}{'P':>8}{'R':>8}{'GT像素':>12}"
        lines = [head, "-" * len(head)]
        for i in range(self.num_classes):
            name = names[i] if names and i < len(names) else f"class_{i}"
            lines.append(
                f"{name:<10}{iou[i]:>8.3f}{f1[i]:>8.3f}{p[i]:>8.3f}{r[i]:>8.3f}{self.support[i]:>12d}"
            )
        lines.append("-" * len(head))
        lines.append(
            f"{'mIoU':<10}{self.miou():>8.3f}{self.mf1():>8.3f}"
            f"{'':>8}{'':>8}{int(self.mat.sum()):>12d}   OA={self.oa():.3f}  Kappa={self.kappa():.3f}"
        )
        return "\n".join(lines)


def fast_hist(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    """单次调用版（兼容旧代码）。"""
    return ConfusionMatrix(num_classes).update(pred, gt).mat

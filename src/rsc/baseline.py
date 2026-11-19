"""Minimum-distance classifier baseline."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .palette import IGNORE_INDEX, NUM_CLASSES


def features(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8/float → (H,W,F) float32 特征。

    F = R,G,B,ExG,亮度,饱和度,色调-红绿差
    """
    x = np.asarray(rgb, dtype=np.float32)
    if x.max() > 1.5:          # 0~255 输入
        x = x / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    exg = 2 * g - r - b                      # 超绿指数 Excess Green
    bright = (r + g + b) / 3.0
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    rg = r - g
    return np.stack([r, g, b, exg, bright, sat, rg], axis=-1).astype(np.float32)


class MinDistanceClassifier:
    """最小距离分类器；接口故意做成 sklearn 风格，方便直接放进对比实验表。"""

    def __init__(self, num_classes: int = NUM_CLASSES, max_per_class: int = 40000, seed: int = 0):
        self.num_classes = num_classes
        self.max_per_class = max_per_class
        self.rng = np.random.default_rng(seed)
        self.mean_: Optional[np.ndarray] = None      # 特征标准化参数
        self.std_: Optional[np.ndarray] = None
        self.centroids_: Optional[np.ndarray] = None  # (C, F)
        self.priors_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> "MinDistanceClassifier":
        feats, labels = [], []
        for img, msk in zip(images, masks):
            f = features(img).reshape(-1, features(img).shape[-1])
            y = np.asarray(msk).reshape(-1)
            keep = y != IGNORE_INDEX
            feats.append(f[keep])
            labels.append(y[keep])
        F = np.concatenate(feats, 0)
        Y = np.concatenate(labels, 0)

        self.mean_ = F.mean(0)
        self.std_ = F.std(0) + 1e-6
        Z = (F - self.mean_) / self.std_

        cents = np.zeros((self.num_classes, F.shape[1]), np.float32)
        priors = np.zeros(self.num_classes, np.float32)
        for c in range(self.num_classes):
            idx = np.nonzero(Y == c)[0]
            priors[c] = len(idx)
            if len(idx) == 0:
                cents[c] = 0.0
                continue
            if len(idx) > self.max_per_class:
                idx = self.rng.choice(idx, self.max_per_class, replace=False)
            cents[c] = Z[idx].mean(0)
        self.centroids_ = cents
        self.priors_ = priors / max(priors.sum(), 1)
        return self

    # ------------------------------------------------------------------
    def decision(self, rgb: np.ndarray) -> np.ndarray:
        """返回 (C,H,W) 的"负平方距离"，可直接当 logits 用。"""
        assert self.centroids_ is not None, "先 fit()"
        f = features(rgb)
        z = (f - self.mean_) / self.std_
        # ||z||^2 - 2 z·mu + ||mu||^2
        z2 = (z ** 2).sum(-1, keepdims=True)
        m2 = (self.centroids_ ** 2).sum(-1)[None, None, :]
        dot = z @ self.centroids_.T
        d2 = z2 - 2 * dot + m2
        # 加一点类别先验，避免"占地少的类"被完全吞掉
        d2 = d2 - 2.0 * np.log(np.maximum(self.priors_, 1e-6))[None, None, :]
        return (-d2).astype(np.float32)          # (H,W,C)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        return self.decision(rgb).argmax(axis=-1).astype(np.uint8)


def batch_predict_fn(clf: MinDistanceClassifier):
    """把分类器包装成 tiling.sliding_window_predict 需要的 predict_fn。

    输入 (B,3,H,W) float32（0~255） → 输出 (B,C,h,w) logits。
    """

    def _fn(batch_nchw: np.ndarray) -> np.ndarray:
        out = []
        for i in range(batch_nchw.shape[0]):
            hwc = np.transpose(batch_nchw[i], (1, 2, 0))
            out.append(np.transpose(clf.decision(hwc), (2, 0, 1)))
        return np.stack(out, 0)

    return _fn

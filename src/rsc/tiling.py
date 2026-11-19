"""大图切片与滑窗（overlap-tile）推理拼接。

高分影像动辄上万像素，不可能整幅送进网络。标准做法是 overlap-tile：
    1) 按 patch 滑窗取块，块间留 overlap（这里默认 1/4 patch）；
    2) 每块预测得到概率图，乘一个中心高、边缘低的权重窗（Hann）；
    3) 累加加权概率 / 累加权重，再取 argmax —— 这样能消除拼接缝。

纯 numpy 实现，不依赖 torch，train/infer 两侧共用同一套坐标逻辑，
避免"训练时裁的块和推理时裁的块对不上"的经典事故。
"""
from __future__ import annotations

from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

Box = Tuple[int, int, int, int]  # (y0, y1, x0, x1)


# --------------------------------------------------------------------------
# 坐标
# --------------------------------------------------------------------------
def tile_starts(length: int, patch: int, stride: int) -> List[int]:
    """生成覆盖 [0, length) 的滑窗起点，保证最后一块贴边且不越界。"""
    if patch <= 0 or stride <= 0:
        raise ValueError("patch/stride 必须为正")
    if patch >= length:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def iter_boxes(h: int, w: int, patch: int, stride: int) -> Iterator[Box]:
    for y in tile_starts(h, patch, stride):
        for x in tile_starts(w, patch, stride):
            yield (y, y + patch, x, x + patch)


def pad_reflect(img: np.ndarray, patch: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """把 HxW(×C) 图 pad 到至少一个 patch，返回 (pad后图像, 原始尺寸)。

    用 reflect 而不是 zero：零值会在预测边界制造出"黑边水体"。
    """
    h, w = img.shape[:2]
    ph = max(0, patch - h)
    pw = max(0, patch - w)
    if ph == 0 and pw == 0:
        return img, (h, w)
    pad_width = [(0, ph), (0, pw)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad_width, mode="reflect"), (h, w)


# --------------------------------------------------------------------------
# 权重窗
# --------------------------------------------------------------------------
def hann_window(patch: int, min_weight: float = 0.05) -> np.ndarray:
    """二维 Hann 窗，边缘不取到 0（否则边缘像素永远拿不到票）。"""
    w1 = np.hanning(patch)
    win = np.outer(w1, w1).astype(np.float32)
    return np.clip(win, min_weight, None)


# --------------------------------------------------------------------------
# 滑窗推理
# --------------------------------------------------------------------------
def sliding_window_predict(
    image: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    num_classes: int,
    patch: int = 512,
    stride: Optional[int] = None,
    batch_size: int = 4,
    apply_softmax: bool = True,
    weight: str = "hann",
) -> np.ndarray:
    """对 HxWxC(float32, 未归一化) 大图滑窗推理。

    Parameters
    ----------
    image : (H, W, C) 数组，值域随意，归一化交给 predict_fn。
    predict_fn : 接受 (B, C, H, W) float32 → 返回 (B, num_classes, h, w) logits/probs 的 numpy。
    weight : ``hann`` 或 ``uniform``（uniform 等价于平均重叠区）。

    Returns
    -------
    (H, W) uint8 类别图
    """
    h, w = image.shape[:2]
    stride = stride or max(1, patch // 2)
    padded, (h0, w0) = pad_reflect(image, patch)
    H, W = padded.shape[:2]

    if weight == "hann":
        wmap = hann_window(patch)
    elif weight == "uniform":
        wmap = np.ones((patch, patch), np.float32)
    else:
        raise ValueError(weight)

    prob = np.zeros((num_classes, H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    boxes = list(iter_boxes(H, W, patch, stride))
    for i in range(0, len(boxes), batch_size):
        chunk = boxes[i : i + batch_size]
        batch = np.stack(
            [np.transpose(padded[y0:y1, x0:x1], (2, 0, 1)) for (y0, y1, x0, x1) in chunk]
        ).astype(np.float32)
        out = np.asarray(predict_fn(batch), dtype=np.float32)
        if out.ndim != 4:
            raise ValueError(f"predict_fn 应返回 4 维 (B,C,h,w)，实际 {out.shape}")
        if apply_softmax:
            out = out - out.max(axis=1, keepdims=True)
            e = np.exp(out)
            out = e / e.sum(axis=1, keepdims=True)
        for k, (y0, y1, x0, x1) in enumerate(chunk):
            prob[:, y0:y1, x0:x1] += out[k] * wmap
            wsum[y0:y1, x0:x1] += wmap

    prob /= np.maximum(wsum, 1e-6)[None]
    pred = prob.argmax(axis=0).astype(np.uint8)
    return pred[:h0, :w0]


# --------------------------------------------------------------------------
# 训练侧裁剪（带最小有效标签比例过滤）
# --------------------------------------------------------------------------
def crop_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch: int,
    stride: int,
    min_label_ratio: float = 0.0,
    ignore_index: int = 255,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Box]]:
    """把整幅影像+标签切成训练块，可选丢弃"几乎全是忽略区"的块。"""
    h, w = image.shape[:2]
    for box in iter_boxes(h, w, patch, stride):
        y0, y1, x0, x1 = box
        m = mask[y0:y1, x0:x1]
        if min_label_ratio > 0:
            valid = (m != ignore_index).mean()
            if valid < min_label_ratio:
                continue
        yield image[y0:y1, x0:x1], m, box

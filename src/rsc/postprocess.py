"""预测后处理：去碎斑、平滑边界、栅格转矢量、面积统计。

工程经验：分割网络输出的最大问题不是"整体错"，而是：
  * 建筑屋顶上零星几个"水体"像素（碎斑）→ 人工修图会骂人；
  * 道路边缘锯齿 → 叠加到影像上很难看；
  * 小图斑面积 < 最小制图单元（如 4 mm² @ 1:10000）→ 按制图规范必须合并。

所以后处理不是可选项，是交付前的必做步骤。这里的实现尽量零重依赖：
有 OpenCV 用 OpenCV，没有就用纯 numpy 退化实现，保证任何环境都能跑。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # OpenCV 快 10~50 倍，能用就用
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

try:
    from scipy import ndimage as ndi

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

from .palette import IGNORE_INDEX, NUM_CLASSES

# --------------------------------------------------------------------------
# 形态学（纯 numpy 退化实现：3x3 十字/方块结构元）
# --------------------------------------------------------------------------
_K3 = np.ones((3, 3), bool)


def _shift_stack(m: np.ndarray, structure: np.ndarray) -> List[np.ndarray]:
    ph, pw = structure.shape[0] // 2, structure.shape[1] // 2
    padded = np.pad(m, ((ph, ph), (pw, pw)), mode="edge")
    out = []
    for i in range(structure.shape[0]):
        for j in range(structure.shape[1]):
            if structure[i, j]:
                out.append(padded[i : i + m.shape[0], j : j + m.shape[1]])
    return out


def binary_erode(m: np.ndarray, structure: np.ndarray = _K3) -> np.ndarray:
    if _HAS_CV2:
        return cv2.erode(m.astype(np.uint8), structure.astype(np.uint8)) > 0
    return np.logical_and.reduce(_shift_stack(m.astype(bool), structure))


def binary_dilate(m: np.ndarray, structure: np.ndarray = _K3) -> np.ndarray:
    if _HAS_CV2:
        return cv2.dilate(m.astype(np.uint8), structure.astype(np.uint8)) > 0
    return np.logical_or.reduce(_shift_stack(m.astype(bool), structure))


def opening(m: np.ndarray, iterations: int = 1) -> np.ndarray:
    for _ in range(iterations):
        m = binary_dilate(binary_erode(m))
    return m


def closing(m: np.ndarray, iterations: int = 1) -> np.ndarray:
    for _ in range(iterations):
        m = binary_erode(binary_dilate(m))
    return m


# --------------------------------------------------------------------------
# 碎斑过滤
# --------------------------------------------------------------------------
def _label_components(m: np.ndarray) -> Tuple[np.ndarray, int]:
    if _HAS_CV2:
        n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        return lab, n
    if _HAS_SCIPY:
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        return lab, n + 1
    raise RuntimeError("需要 opencv 或 scipy 才能做连通域分析")


def remove_small_objects(
    mask: np.ndarray,
    min_size: int = 64,
    num_classes: int = NUM_CLASSES,
    protect: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """剔除面积小于 min_size 像素的独立图斑，回填为其周围的主导类别。

    protect: 不做处理的类别（如水体细长河道常被误删，可设 protect=(3,)）。
    """
    out = mask.copy()
    protect = set(protect or ())
    for c in range(num_classes):
        if c in protect:
            continue
        lab, n = _label_components(mask == c)
        if n <= 1:
            continue
        counts = np.bincount(lab.ravel())
        # 背景标签 0 不能当碎斑；注意不能写 [1:]（那会误删"第一个小图斑"而不是背景）
        small_ids = np.where(counts < min_size)[0]
        small_ids = small_ids[small_ids != 0]
        if small_ids.size == 0:
            continue
        small = np.isin(lab, small_ids)
        filled = _neighbour_majority(mask, small)
        out[small] = filled[small]
    return out


def _neighbour_majority(mask: np.ndarray, holes: np.ndarray, iters: int = 8) -> np.ndarray:
    """把 holes 区域由外向内"长"回周围已知类别（相当于形态学空洞填充）。

    比取全局众数更合理：河边的小碎斑会被填成水体，而不是被填成占比最大的植被。
    """
    canvas = mask.copy()
    unknown = holes.copy()
    for _ in range(iters):
        if not unknown.any():
            break
        newvals = canvas.copy()
        progress = np.zeros_like(unknown)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_val = np.roll(canvas, (dy, dx), axis=(0, 1))
            shifted_hole = np.roll(holes, (dy, dx), axis=(0, 1))
            take = unknown & (~shifted_hole) & (~progress)
            newvals[take] = shifted_val[take]
            progress |= take
        canvas = newvals
        unknown &= ~progress
    canvas[unknown] = 0
    return canvas


# --------------------------------------------------------------------------
# 平滑
# --------------------------------------------------------------------------
def majority_filter(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """众数滤波：去掉椒盐噪声，代价是细小地物（车、杆）会消失。"""
    if _HAS_CV2:
        return cv2.medianBlur(mask.astype(np.uint8), k)
    # numpy 退化实现：k x k 窗口内计数最多的类别（对 6 类足够快）
    pad = k // 2
    m = np.pad(mask, pad, mode="edge")
    h, w = mask.shape
    stack = np.stack(
        [m[i : i + h, j : j + w] for i in range(k) for j in range(k)], axis=0
    )
    # bincount per-pixel 太慢，用 one-hot 累加
    scores = np.stack([(stack == c).sum(axis=0) for c in range(NUM_CLASSES)], axis=0)
    return scores.argmax(axis=0).astype(np.uint8)


def smooth_boundary(mask: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """对每个类别概率做高斯平滑后再取 argmax？—— 这里输入已是硬标签，
    用"每类二值图高斯模糊 + argmax"近似，可显著减少锯齿。
    """
    if _HAS_CV2:
        blurred = np.stack(
            [
                cv2.GaussianBlur((mask == c).astype(np.float32), (0, 0), sigma)
                for c in range(NUM_CLASSES)
            ]
        )
        return blurred.argmax(axis=0).astype(np.uint8)
    if _HAS_SCIPY:
        blurred = np.stack(
            [ndi.gaussian_filter((mask == c).astype(np.float32), sigma) for c in range(NUM_CLASSES)]
        )
        return blurred.argmax(axis=0).astype(np.uint8)
    return mask


# --------------------------------------------------------------------------
# 一键后处理
# --------------------------------------------------------------------------
def clean_mask(
    mask: np.ndarray,
    min_size: int = 64,
    median_k: int = 0,
    sigma: float = 0.0,
    protect: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    out = mask.copy()
    if median_k and median_k > 1:
        out = majority_filter(out, median_k)
    if sigma and sigma > 0:
        out = smooth_boundary(out, sigma)
    if min_size and min_size > 1:
        out = remove_small_objects(out, min_size, protect=protect)
    return out


# --------------------------------------------------------------------------
# 栅格 → 矢量
# --------------------------------------------------------------------------
def vectorize(mask: np.ndarray, transform=None, crs=None, min_area_px: int = 64):
    """转 GeoJSON FeatureCollection（需要 rasterio）。给 GIS 用。

    mask 中 IGNORE_INDEX 会被跳过；每个图斑带 class_id / class_name 属性。
    """
    from rasterio import features  # 延迟导入：只有导出矢量时才需要

    from .palette import CLASSES, IGNORE_INDEX as IG

    names = {c.id: c.name_en for c in CLASSES}
    feats = []
    data = mask.astype(np.uint8)
    for c in np.unique(data):
        if c == IG:
            continue
        m = (data == c).astype(np.uint8)
        for geom, val in features.shapes(m, mask=m.astype(bool), transform=transform):
            feats.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "class_id": int(c),
                        "class_name": names.get(int(c), "unknown"),
                    },
                }
            )
    return {"type": "FeatureCollection", "crs": str(crs) if crs else None, "features": feats}


# --------------------------------------------------------------------------
# 光谱规则修正：把"被误判成建筑的耕地"改回裸土
# --------------------------------------------------------------------------
def fix_building_fields(
    pred: np.ndarray,
    rgb: np.ndarray,
    rb_thresh: float = 15.0,
    bright_thresh: float = 118.0,
) -> Tuple[np.ndarray, int]:
    """跨域推理时的高频错误：**收获后的农田被大量判成建筑**。

    成因：农田在影像上是"大片、平滑、土黄、中高亮度"，与训练集里的大屋顶/工业厂房
    在低分辨率下非常像；而建筑在真实村庄里是"小、有纹理、颜色杂"。

    这里的修正只做一件事：在**已被判为建筑**的像素里，把同时满足
        R - B > 阈值  且  亮度 > 阈值
    的像素改成裸土。阈值来自本影像的实测统计（农田 R-B≈21、亮度≈126；
    村庄 R-B≈9、亮度≈113），因此对村庄几乎没有误伤。

    返回 ``(修正后的图, 被修改的像素数)``。这是一个**后处理补丁**，
    根治办法是用本地样本微调模型。
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bright = (r + g + b) / 3.0
    field_like = (pred == 1) & ((r - b) > rb_thresh) & (bright > bright_thresh)
    out = pred.copy()
    out[field_like] = 5
    return out, int(field_like.sum())


# --------------------------------------------------------------------------
# 面积统计（行业成果图的核心指标）
# --------------------------------------------------------------------------
def area_stats(
    mask: np.ndarray,
    gsd: float = 1.0,
    names: Optional[List[str]] = None,
) -> List[Dict]:
    """按像元数折算面积。gsd: 地面分辨率(米/像元)，例如 LoveDA 0.3、Sentinel-2 10。"""
    from .palette import CLASSES

    names = names or [c.name for c in CLASSES]
    px_area = float(gsd) ** 2
    total = int((mask != IGNORE_INDEX).sum())
    rows = []
    for c in range(NUM_CLASSES):
        n = int((mask == c).sum())
        rows.append(
            {
                "class_id": c,
                "name": names[c],
                "pixels": n,
                "ratio": n / total if total else 0.0,
                "area_m2": n * px_area,
                "area_ha": n * px_area / 1e4,
                "area_km2": n * px_area / 1e6,
            }
        )
    return rows

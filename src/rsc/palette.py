"""类别体系与配色：全项目唯一真源（single source of truth）。

所有数据集的原生标签都通过 ``LABEL_PRESETS`` 映射到本项目的 6 类体系，
这样模型、指标、制图不必关心底层用的是 LoveDA 还是 Vaihingen。

    0 背景/其他   1 建筑   2 道路   3 水体   4 植被   5 裸土   255 忽略
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

IGNORE_INDEX = 255


@dataclass(frozen=True)
class ClassDef:
    id: int
    name: str          # 中文名（图例用）
    name_en: str       # 英文名（日志/文件名用）
    color: Tuple[int, int, int]   # 制图配色（RGB）


# 配色参考：鲜明、色盲友好、打印后仍可区分
CLASSES: List[ClassDef] = [
    ClassDef(0, "背景/其他", "background", (110, 110, 110)),
    ClassDef(1, "建筑",     "building",   (214, 62, 62)),
    ClassDef(2, "道路",     "road",       (246, 196, 63)),
    ClassDef(3, "水体",     "water",      (46, 108, 208)),
    ClassDef(4, "植被",     "vegetation", (58, 158, 84)),
    ClassDef(5, "裸土",     "bare_soil",  (186, 146, 106)),
]

NUM_CLASSES = len(CLASSES)
CLASS_NAMES = [c.name for c in CLASSES] + ["忽略"]
CLASS_NAMES_EN = [c.name_en for c in CLASSES] + ["ignore"]
PALETTE = np.array([c.color for c in CLASSES], dtype=np.uint8)
# 忽略区在可视化中的颜色（浅灰），不参与指标
IGNORE_COLOR = (232, 232, 232)

# 制图时的类名（更"行业报告"一点，可用于成果图图例）
CARTO_NAMES = ["其他", "建筑用地", "道路交通", "水域", "植被覆盖", "裸土/在建"]

# --------------------------------------------------------------------------
# 不同数据集的标签 → 本项目 6 类
# --------------------------------------------------------------------------
LABEL_PRESETS: Dict[str, Dict[int, int]] = {
    # LoveDA: 0 background, 1 building, 2 road, 3 water, 4 barren, 5 forest, 6 agriculture
    "loveda": {0: 255, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 4},
    # HuggingFace 版 LoveDA（chloechia/loveda）：标签整体右移一位
    #   0 Ignore, 1 Background, 2 Building, 3 Road, 4 Water, 5 Barren, 6 Forest, 7 Agricultural
    "loveda_hf": {0: 255, 1: 255, 2: 1, 3: 2, 4: 3, 5: 5, 6: 4, 7: 4},
    # ISPRS Vaihingen: 0 impervious, 1 building, 2 low veg, 3 tree, 4 car, 5 clutter
    #   不透水面 → 道路类（含广场/硬化地面），车辆面积太小按忽略处理
    "vaihingen": {0: 2, 1: 1, 2: 4, 3: 4, 4: 255, 5: 255},
    # ISPRS Potsdam: 同 Vaihingen 的 6 类定义
    "potsdam": {0: 2, 1: 1, 2: 4, 3: 4, 4: 255, 5: 255},
    # SpaceNet building: 0 background, 1 building（二分类，可直接扩到 6 类）
    "spacenet": {0: 0, 1: 1},
    # 本项目样例数据已经是 0..5
    "raw": {i: i for i in range(NUM_CLASSES)},
}


def build_lut(preset: str = "loveda") -> np.ndarray:
    """构造 256 长度的查找表，用于一次性向量化映射标签。"""
    if preset not in LABEL_PRESETS:
        raise KeyError(f"未知标签预设 {preset!r}，可选: {list(LABEL_PRESETS)}")
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for src, dst in LABEL_PRESETS[preset].items():
        lut[src] = dst
    return lut


def remap_labels(mask: np.ndarray, preset: str = "loveda") -> np.ndarray:
    """把任意数据集的标签图映射到 0..5 + 255。"""
    return build_lut(preset)[np.asarray(mask, dtype=np.uint8)]


# --------------------------------------------------------------------------
# 可视化
# --------------------------------------------------------------------------
def colorize(mask: np.ndarray, ignore_color=IGNORE_COLOR) -> np.ndarray:
    """类别图 → RGB 真彩图（uint8, HxWx3）。"""
    mask = np.asarray(mask)
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for c in CLASSES:
        rgb[mask == c.id] = c.color
    rgb[mask == IGNORE_INDEX] = ignore_color
    rgb[(mask > NUM_CLASSES - 1) & (mask != IGNORE_INDEX)] = ignore_color
    return rgb


def legend_entries(names: List[str] = None):
    """返回 [(color01, label), ...]，颜色归一化到 0~1 供 matplotlib 使用。"""
    names = names or [c.name for c in CLASSES]
    out = []
    for c, n in zip(CLASSES, names):
        out.append((np.array(c.color) / 255.0, n))
    return out

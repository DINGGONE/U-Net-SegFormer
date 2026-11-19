"""rsc — remote sensing land-cover classification toolkit.

模块划分（只有 data/models/losses/engine 依赖 PyTorch，其余为纯 numpy/OpenCV）:

    palette      类别体系与配色（全项目唯一真源）
    metrics      混淆矩阵 / mIoU / F1 / OA / Kappa
    tiling       大图切片与滑窗推理拼接
    data         Dataset + albumentations 增强
    models       U-Net / SegFormer 构建
    losses       CE+Dice / 类别权重
    engine       训练与验证循环
    postprocess  形态学、最小图斑、栅格矢量化
    cartography  制图：配色、图例、比例尺、指北针、海报
"""
from __future__ import annotations

import os as _os

# ---------------------------------------------------------------------------
# Windows + Anaconda 的经典冲突：conda 自带的 MKL 和 PyTorch 自带的 OpenMP
# 同时加载 libiomp5md.dll，进程会直接以 "OMP: Error #15" 中止。
# 必须在 torch / cv2 任一被 import 之前设置，所以放在包的最顶部。
# （非 Windows 或用户已自行设置时不干预）
# ---------------------------------------------------------------------------
if _os.name == "nt" and "KMP_DUPLICATE_LIB_OK" not in _os.environ:
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = "0.1.0"

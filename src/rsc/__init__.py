"""Remote-sensing land-cover segmentation toolkit."""
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

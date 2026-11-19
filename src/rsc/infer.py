"""Inference: sliding-window prediction and GeoTIFF output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .palette import IGNORE_INDEX, NUM_CLASSES, colorize
from .tiling import sliding_window_predict

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
def torch_predict_fn(
    model: torch.nn.Module,
    device: torch.device,
    mean: Sequence[float] = MEAN,
    std: Sequence[float] = STD,
    use_amp: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """把 torch 模型包装成 ``(B,3,H,W) float32[0,255] → (B,C,h,w) logits`` 的函数。"""
    model.eval()
    m = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    amp = bool(use_amp and device.type == "cuda")

    @torch.no_grad()
    def _fn(batch_nchw: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(batch_nchw)).to(device, non_blocking=True)
        x = (x / 255.0 - m) / s
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        return logits.float().cpu().numpy()

    return _fn


def segment_image(
    model: torch.nn.Module,
    rgb: np.ndarray,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    patch: int = 512,
    stride: Optional[int] = None,
    batch_size: int = 4,
) -> np.ndarray:
    """(H,W,3) uint8 → (H,W) uint8 类别图。"""
    return sliding_window_predict(
        rgb,
        torch_predict_fn(model, device),
        num_classes,
        patch=patch,
        stride=stride,
        batch_size=batch_size,
        apply_softmax=True,
        weight="hann",
    )


# --------------------------------------------------------------------------
# 权重读写
# --------------------------------------------------------------------------
def save_checkpoint(path: str | Path, model, optimizer=None, epoch: int = 0,
                    best: float = 0.0, cfg: Optional[Dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": epoch, "best": best,
               "arch": getattr(model, "arch_name", None), "cfg": cfg or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path, device: torch.device, build_fn=None):
    """返回 ``(model, ckpt)``。build_fn(cfg) 用于按存档配置重建网络结构。"""
    try:
        # 显式给 weights_only=False：存档里有 config 字典，且能消掉 torch 2.4 的 FutureWarning
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:  # torch < 2.0 没有这个参数
        ckpt = torch.load(str(path), map_location=device)
    cfg = ckpt.get("cfg", {})
    if build_fn is None:
        from .models import build_model

        mc = cfg.get("model", {})
        model = build_model(
            arch=mc.get("arch", "unet"),
            encoder=mc.get("encoder", "resnet34"),
            num_classes=cfg.get("num_classes", NUM_CLASSES),
            in_channels=cfg.get("in_channels", 3),
            pretrained=False,
        )
    else:
        model = build_fn(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


# --------------------------------------------------------------------------
# GeoTIFF 推理
# --------------------------------------------------------------------------
def predict_geotiff(
    ckpt_path: str | Path,
    src_tif: str | Path,
    dst_tif: str | Path,
    dst_png: Optional[str | Path] = None,
    patch: int = 512,
    stride: Optional[int] = None,
    device: str = "cuda",
    stretch: bool = True,
    fix_fields: bool = False,
) -> Dict:
    """对整幅 GeoTIFF 预测，输出类别 GeoTIFF（uint8, 255=nodata）与彩色 PNG。

    输入影像可以是 3 波段（RGB）或 4 波段（默认取前 3 波段，Vaihingen 为 NIR-R-G 时
    请用 ``--band-order`` 调序或改用 4 通道模型）。
    """
    import rasterio

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model, ckpt = load_checkpoint(ckpt_path, dev)
    num_classes = ckpt.get("cfg", {}).get("num_classes", NUM_CLASSES)

    with rasterio.open(str(src_tif)) as src:
        arr = src.read()                                  # (C,H,W)
        profile = src.profile.copy()
        crs, transform = src.crs, src.transform
        h, w = arr.shape[1], arr.shape[2]

    if arr.shape[0] > 3:
        arr = arr[:3]
    rgb = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    # 16bit/12bit 影像先做百分位拉伸到 0~255，否则归一化后分布完全对不上训练集
    if stretch or rgb.max() > 255:
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)
    rgb = rgb.astype(np.uint8)

    pred = segment_image(model, rgb, dev, num_classes, patch, stride)

    if fix_fields:
        from .postprocess import fix_building_fields

        pred, n_fix = fix_building_fields(pred, rgb)
        print(f"  光谱修正：{n_fix} 个像素由「建筑」改为「裸土」（占全图 {n_fix / pred.size * 100:.1f}%）")

    profile.update(driver="GTiff", count=1, dtype="uint8", compress="lzw",
                   nodata=IGNORE_INDEX)
    Path(dst_tif).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(dst_tif), "w", **profile) as dst:
        dst.write(pred[None].astype(np.uint8))
        dst.set_band_description(1, "landcover_class_0-5_255=nodata")

    if dst_png:
        Path(dst_png).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(colorize(pred)).save(str(dst_png))

    stats = {
        "source": str(src_tif),
        "output": str(dst_tif),
        "shape": [int(h), int(w)],
        "crs": str(crs),
        "transform": list(transform)[:6],
        "num_classes": int(num_classes),
    }
    meta_path = Path(dst_tif).with_suffix(".json")
    meta_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats

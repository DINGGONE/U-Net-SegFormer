"""体检脚本：把任意一幅影像丢进来，告诉你"能不能直接进本项目、要改什么"。

    python scripts/inspect_image.py --src D:\data\我的影像.tif
    python scripts/inspect_image.py --src data\sample\scene\scene_rgb.png

会检查 8 件事：
    1. 能不能读（格式 / 是否损坏）
    2. 尺寸、波段数（模型要 3 波段）
    3. 数据类型、值域（16bit 必须拉伸，否则全黑）
    4. 坐标系 / 地面分辨率 GSD（决定比例尺和面积算得对不对）
    5. 和训练数据的 GSD 差多少（差距大要重新训练，不能硬套）
    6. 有没有旋转（transform 带 b 项要重投影）
    7. 瓦片数估算 & 推理耗时
    8. 结论：能直接推理 / 要先做什么
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# 训练数据的假定分辨率（LoveDA 0.3 m / Vaihingen 0.09 m）
TRAIN_GSD_HINT = 0.3


def is_geotiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def read_geotiff(path: Path):
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(str(path)) as ds:
        gsd_x = abs(ds.transform.a) if ds.transform else None
        gsd_y = abs(ds.transform.e) if ds.transform else None
        info = {
            "width": ds.width, "height": ds.height, "count": ds.count,
            "dtype": ds.dtypes[0], "crs": str(ds.crs) if ds.crs else None,
            "transform": tuple(ds.transform)[:6] if ds.transform else None,
            "gsd_x": gsd_x, "gsd_y": gsd_y, "nodata": ds.nodata,
        }
        # 只读中心一小块做统计，避免把几 GB 的影像读进内存
        h = min(1024, ds.height)
        w = min(1024, ds.width)
        win = Window((ds.width - w) // 2, (ds.height - h) // 2, w, h)   # (col, row, w, h)
        arr = ds.read(window=win).astype(np.float64)
        info["sample"] = arr
    return info


def read_plain(path: Path):
    from PIL import Image

    im = Image.open(path)
    arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[..., None]
    info = {
        "width": im.width, "height": im.height,
        "count": arr.shape[2], "dtype": str(arr.dtype),
        "crs": None, "transform": None, "gsd_x": None, "gsd_y": None, "nodata": None,
        "sample": np.transpose(arr[: min(1024, arr.shape[0]), : min(1024, arr.shape[1])], (2, 0, 1)).astype(np.float64),
    }
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--patch", type=int, default=512)
    args = ap.parse_args()

    path = Path(args.src)
    if not path.exists():
        print(f"[错误] 文件不存在: {path}")
        return 1

    print("=" * 66)
    print(f"  影像体检: {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    print("=" * 66)

    try:
        info = read_geotiff(path) if is_geotiff(path) else read_plain(path)
    except Exception as e:
        print(f"  ❌ 读不了：{type(e).__name__}: {e}")
        print("     建议：用 QGIS 另存为 GeoTIFF；或转成 PNG 再试。")
        return 1

    s = info["sample"]
    print(f"  尺寸      : {info['width']} × {info['height']} px")
    print(f"  波段数    : {info['count']}")
    print(f"  数据类型  : {info['dtype']}")
    print(f"  值域(采样): {s.min():.0f} ~ {s.max():.0f}   均值 {s.mean():.1f}")
    print(f"  坐标系    : {info['crs'] or '无（普通图片，不能叠加 GIS 图层）'}")
    if info["gsd_x"]:
        print(f"  分辨率    : {info['gsd_x']:.3f} m × {info['gsd_y']:.3f} m / 像素")
        coverage = info["width"] * info["gsd_x"] * info["height"] * info["gsd_y"] / 1e6
        print(f"  覆盖面积  : {coverage:.2f} km²  ({coverage * 100:.0f} 公顷)")

    # ---------------- 逐项判定 ----------------
    print("\n  检查结果：")
    problems, notes = [], []

    if info["count"] >= 3:
        print(f"    ✅ 波段够用（取前 3 个波段当 RGB）")
    else:
        problems.append(f"只有 {info['count']} 个波段，模型需要 3 波段（RGB）。"
                        "单波段灰度图可以先复制成 3 通道，但精度会打折。")

    if "float" in str(info["dtype"]) or s.max() > 255:
        print(f"    ⚠️  16bit/浮点影像：推理时会自动做 2%~98% 百分位拉伸（已内置）")
    else:
        print(f"    ✅ 8bit，可直接用")

    if info["gsd_x"]:
        ratio = info["gsd_x"] / TRAIN_GSD_HINT
        if 0.5 <= ratio <= 2.0:
            print(f"    ✅ 分辨率与训练数据（{TRAIN_GSD_HINT} m）接近，可以直接套模型")
        else:
            problems.append(
                f"分辨率 {info['gsd_x']:.2f} m 与训练数据 {TRAIN_GSD_HINT} m 相差 {ratio:.1f} 倍。"
                "直接套模型精度会明显下降，建议重新训练或加多尺度增强。")
        if info["transform"] and abs(info["transform"][1]) > 1e-9:
            problems.append("影像带旋转（transform 的 b 项 ≠ 0），需要先重投影成正北方向。")
    else:
        notes.append("没有地理信息（不是 GeoTIFF），比例尺和面积需要手动给 --gsd 参数。")

    if info["width"] < args.patch or info["height"] < args.patch:
        notes.append(f"尺寸小于 patch({args.patch})，会被反射填充后推理，边缘不可信。")

    # 推理耗时粗估
    stride = args.patch // 2
    import math

    nx = max(1, math.ceil((info["width"] - args.patch) / stride) + 1)
    ny = max(1, math.ceil((info["height"] - args.patch) / stride) + 1)
    print(f"\n  推理规模  : 约 {nx}×{ny} = {nx * ny} 个 {args.patch}×{args.patch} 瓦片")
    print(f"              RTX 2060 上预计 {nx * ny * 0.12:.0f}~{nx * ny * 0.25:.0f} 秒")

    if problems:
        print("\n  ❌ 不能直接推理，需要先处理：")
        for p in problems:
            print(f"     · {p}")
    else:
        print("\n  ✅ 可以直接推理。命令：")
    if notes:
        print("\n  ℹ️  提醒：")
        for n in notes:
            print(f"     · {n}")

    if not problems:
        gsd = f"{info['gsd_x']:.2f}" if info["gsd_x"] else "0.3"
        print(f'\n     "C:\\ProgramData\\anaconda3\\python.exe" scripts\\predict.py ^')
        print(f'         --ckpt runs\\loveda_unet_r34\\best.pt ^')
        print(f'         --src "{path}" --out-dir outputs\\my_area --gsd {gsd} --poster')
    else:
        print("\n  需要先把影像整理成训练数据 → 见 docs/01_数据与标注.md 第 3 节（裁剪）")

    print("\n  ⚠️ 关键前提：权重必须是在【同类数据】上训练的。")
    print("     现在只有合成数据训出来的玩具权重（runs\\gpu_smoke），拿来跑真实影像结果没有意义。")
    print("     先用 LoveDA 训出 runs\\loveda_unet_r34\\best.pt，再跑上面的命令。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

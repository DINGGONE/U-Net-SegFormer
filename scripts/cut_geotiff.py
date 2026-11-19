"""Cut a large GeoTIFF into training tiles.

Usage:
    python scripts/cut_geotiff.py --src a.tif --label a_label.tif --out data/x/train --size 512 --stride 256"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsc.palette import remap_labels  # noqa: E402


def stretch_global(ds, bands=(1, 2, 3), q=(2, 98)) -> tuple:
    """用概视图统计全局拉伸参数，避免每块各自拉伸导致亮度跳变。"""
    h = max(1, ds.height // 16)
    w = max(1, ds.width // 16)
    ov = ds.read(indexes=list(bands), out_shape=(len(bands), h, w), resampling=1)
    ov = ov.astype(np.float32)
    lo = np.percentile(ov, q[0], axis=(1, 2))
    hi = np.percentile(ov, q[1], axis=(1, 2))
    return lo.reshape(-1, 1, 1), np.maximum(hi - lo, 1e-6).reshape(-1, 1, 1)


def main() -> None:
    import rasterio
    from rasterio.windows import Window

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--label", default=None, help="标签 GeoTIFF；不给则只切影像")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=None, help="默认 = size（不重叠）")
    ap.add_argument("--preset", default="vaihingen")
    ap.add_argument("--min-valid", type=float, default=0.3, help="标签有效像素占比下限")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--max-tiles", type=int, default=0, help="0=不限制")
    ap.add_argument("--keep-tif", action="store_true", help="同时保留每块的 GeoTIFF")
    args = ap.parse_args()

    size = args.size
    stride = args.stride or size
    prefix = args.prefix or Path(args.src).stem
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.src) as ds:
        n_band = min(3, ds.count)
        lo, hi = stretch_global(ds, bands=tuple(range(1, n_band + 1)))
        lab_ds = rasterio.open(args.label) if args.label else None
        rows = []
        kept = skipped = 0
        for y in range(0, ds.height - size + 1, stride):
            for x in range(0, ds.width - size + 1, stride):
                win = Window(x, y, size, size)
                arr = ds.read(window=win, indexes=list(range(1, n_band + 1))).astype(np.float32)
                rgb = np.clip((arr - lo) / hi * 255.0, 0, 255).astype(np.uint8)
                if n_band == 1:
                    rgb = np.repeat(rgb, 3, axis=0)
                name = f"{prefix}_{y:05d}_{x:05d}"
                mask = None
                if lab_ds is not None:
                    m = lab_ds.read(1, window=win)
                    mask = remap_labels(m, args.preset)
                    valid = (mask != 255).mean()
                    if valid < args.min_valid:
                        skipped += 1
                        continue
                Image.fromarray(np.transpose(rgb, (1, 2, 0))).save(out / "images" / f"{name}.png")
                if mask is not None:
                    Image.fromarray(mask, mode="L").save(out / "masks" / f"{name}.png")
                if args.keep_tif:
                    tr = ds.window_transform(win)
                    prof = ds.profile.copy()
                    prof.update(count=3, dtype="uint8", height=size, width=size,
                                transform=tr, compress="lzw")
                    with rasterio.open(out / "images" / f"{name}.tif", "w", **prof) as o:
                        o.write(rgb)
                rows.append({"name": name, "row": y, "col": x,
                             "valid_ratio": None if mask is None else round(float(valid), 4)})
                kept += 1
                if args.max_tiles and kept >= args.max_tiles:
                    break
            if args.max_tiles and kept >= args.max_tiles:
                break
        if lab_ds is not None:
            lab_ds.close()

    with open(out / "tiles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "row", "col", "valid_ratio"])
        w.writeheader()
        w.writerows(rows)

    print(f"完成：保留 {kept} 块，跳过 {skipped} 块（标签有效像素 < {args.min_valid}），输出 {out}")
    if not rows:
        print("[warn] 一块都没切出来：检查 --size 是否大于影像尺寸，或 min-valid 是否过高")


if __name__ == "__main__":
    main()

"""把样例 PNG 包成一份带坐标系/分辨率的 GeoTIFF，用来测试"自己的影像"整条流程。

真实场景里你用不到它；它的作用是让你在手头还没有真实影像时，
也能先把 GeoTIFF → 滑窗推理 → 保留坐标 → 成果图 这条链路验证一遍。

    python scripts/make_test_geotiff.py
    python scripts/make_test_geotiff.py --src data/sample/scene/scene_rgb.png --out outputs/test_area.tif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 随便给一个北京附近的 UTM 50N 原点，只为让坐标"看起来是真的"
CRS = "EPSG:32650"
ORIGIN_X, ORIGIN_Y = 442000.0, 4420000.0
GSD = 0.3


def main() -> None:
    import rasterio
    from rasterio.transform import from_origin

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "sample" / "scene" / "scene_rgb.png"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "test_area.tif"))
    ap.add_argument("--gsd", type=float, default=GSD)
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.src).convert("RGB"))
    h, w = rgb.shape[:2]
    transform = from_origin(ORIGIN_X, ORIGIN_Y, args.gsd, args.gsd)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 3, "dtype": "uint8",
        "crs": CRS, "transform": transform, "compress": "lzw",
    }
    with rasterio.open(str(out), "w", **profile) as dst:
        dst.write(np.transpose(rgb, (2, 0, 1)))

    print(f"已生成: {out}")
    print(f"  尺寸   : {w} × {h}")
    print(f"  坐标系 : {CRS}")
    print(f"  分辨率 : {args.gsd} m")
    print(f"  覆盖   : {w * args.gsd:.0f} m × {h * args.gsd:.0f} m = "
          f"{w * h * args.gsd ** 2 / 1e6:.3f} km²")


if __name__ == "__main__":
    main()

"""推理入口：整幅影像 → 分类图（+ 可选成果图）。

    # GeoTIFF（保留坐标系，可拖进 QGIS）
    python scripts/predict.py --ckpt runs/loveda_unet_r34/best.pt ^
        --src data/raw/vaihingen_area15.tif --out-dir outputs/vaihingen

    # 普通 PNG/JPEG（无地理信息，只出彩色图和成果图）
    python scripts/predict.py --ckpt runs/x/best.pt --src data/sample/scene/scene_rgb.png ^
        --out-dir outputs/demo --gsd 0.3 --poster
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsc.cartography import landcover_poster  # noqa: E402
from rsc.infer import load_checkpoint, predict_geotiff, segment_image  # noqa: E402
from rsc.palette import NUM_CLASSES, colorize  # noqa: E402
from rsc.postprocess import clean_mask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", default="outputs/predict")
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--gsd", type=float, default=0.3, help="米/像元，用于比例尺与面积")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-size", type=int, default=80, help="后处理最小图斑（0=不做）")
    ap.add_argument("--fix-fields", action="store_true",
                    help="光谱修正：把误判成建筑的大片耕地改回裸土（跨域推理时建议开）")
    ap.add_argument("--poster", action="store_true", help="同时输出土地利用解译成果图")
    ap.add_argument("--model-name", default="")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.src)
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    stride = args.stride or args.patch // 2

    if src.suffix.lower() in {".tif", ".tiff"}:
        stats = predict_geotiff(args.ckpt, src, out / f"{src.stem}_landcover.tif",
                                out / f"{src.stem}_landcover.png", args.patch, stride,
                                args.device if dev.type == "cuda" else "cpu",
                                fix_fields=args.fix_fields)
        print(f"GeoTIFF 输出: {stats['output']}  CRS={stats['crs']}")
        # 后续制图需要原图，重新读一遍并做同样的拉伸
        import rasterio

        with rasterio.open(str(src)) as ds:
            arr = ds.read()
        rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.float32)
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        mask = np.asarray(Image.open(out / f"{src.stem}_landcover.png"))
        # 彩色 PNG 无法反推类别，重新读 tif
        with rasterio.open(str(out / f"{src.stem}_landcover.tif")) as ds:
            mask = ds.read(1)
    else:
        rgb = np.asarray(Image.open(src).convert("RGB"))
        model, ckpt = load_checkpoint(args.ckpt, dev)
        ncls = ckpt.get("cfg", {}).get("num_classes", NUM_CLASSES)
        mask = segment_image(model, rgb, dev, ncls, args.patch, stride)
        if args.fix_fields:
            from rsc.postprocess import fix_building_fields

            mask, n_fix = fix_building_fields(mask, rgb)
            print(f"光谱修正：{n_fix} 像素 建筑→裸土（{n_fix / mask.size * 100:.1f}%）")
        if args.min_size:
            mask = clean_mask(mask, min_size=args.min_size, median_k=3)
        Image.fromarray(colorize(mask)).save(out / f"{src.stem}_landcover.png")
        np.save(out / f"{src.stem}_landcover.npy", mask)
        print(f"分类图: {out / f'{src.stem}_landcover.png'}")

    if args.poster:
        p = landcover_poster(
            rgb, mask, gsd=args.gsd,
            title="区域土地利用智能解译图",
            subtitle=f"深度学习语义分割 · {src.name}",
            model_name=args.model_name or "U-Net / SegFormer",
            out_png=str(out / f"{src.stem}_poster.png"),
            out_pdf=str(out / f"{src.stem}_poster.pdf"),
            show_image_inset=True,
        )
        print(f"成果图: {p}")


if __name__ == "__main__":
    main()

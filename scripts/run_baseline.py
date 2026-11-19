"""End-to-end pipeline with a minimum-distance classifier (no GPU).

Usage:
    python scripts/run_baseline.py"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsc.baseline import MinDistanceClassifier, batch_predict_fn  # noqa: E402
from rsc.cartography import (  # noqa: E402
    comparison_figure,
    confusion_figure,
    landcover_poster,
    setup_fonts,
)
from rsc.metrics import ConfusionMatrix  # noqa: E402
from rsc.palette import CLASS_NAMES, CARTO_NAMES, NUM_CLASSES, colorize  # noqa: E402
from rsc.postprocess import area_stats, clean_mask  # noqa: E402
from rsc.tiling import sliding_window_predict  # noqa: E402


def load_pairs(split_dir: Path):
    imgs = sorted((split_dir / "images").glob("*.png"))
    out = []
    for p in imgs:
        m = split_dir / "masks" / p.name
        if m.exists():
            out.append((p, m))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "sample"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--gsd", type=float, default=0.3, help="地面分辨率 米/像元")
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--min-size", type=int, default=80, help="后处理最小图斑(像素)")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------- 1. 训练（拟合类别中心）----------------
    train = load_pairs(data / "train")
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p, _ in train]
    msks = [np.asarray(Image.open(m)) for _, m in train]
    print(f"[1/6] 训练瓦片 {len(train)} 张，尺寸 {imgs[0].shape}")
    clf = MinDistanceClassifier().fit(imgs, msks)
    print(f"      类别中心拟合完成，先验占比 = {np.round(clf.priors_, 3)}")

    # ---------------- 2. 验证（滑窗推理）----------------
    val = load_pairs(data / "val")
    cm = ConfusionMatrix(num_classes=NUM_CLASSES)
    predict_fn = batch_predict_fn(clf)
    t_val = time.time()
    for p, m in val:
        img = np.asarray(Image.open(p).convert("RGB"))
        gt = np.asarray(Image.open(m))
        pred = sliding_window_predict(
            img, predict_fn, NUM_CLASSES, patch=args.patch, stride=args.stride, batch_size=4
        )
        cm.update(pred, gt)
    print(f"[2/6] 滑窗验证 {len(val)} 张，耗时 {time.time() - t_val:.1f}s")
    print(cm.format_table(CLASS_NAMES))

    # ---------------- 3. 后处理对比 ----------------
    img0, m0 = val[0]
    raw = sliding_window_predict(
        np.asarray(Image.open(img0).convert("RGB")), predict_fn, NUM_CLASSES,
        patch=args.patch, stride=args.stride,
    )
    gt0 = np.asarray(Image.open(m0))
    cleaned = clean_mask(raw, min_size=args.min_size, median_k=3)
    cm_raw, cm_cln = ConfusionMatrix(NUM_CLASSES), ConfusionMatrix(NUM_CLASSES)
    cm_raw.update(raw, gt0)
    cm_cln.update(cleaned, gt0)
    print(f"[3/6] 后处理：mIoU {cm_raw.miou():.3f} → {cm_cln.miou():.3f}  "
          f"(最小图斑 {args.min_size}px + 中值滤波)")

    # ---------------- 4. 整幅场景推理 + 制图 ----------------
    scene_rgb = np.asarray(Image.open(data / "scene" / "scene_rgb.png").convert("RGB"))
    scene_gt = np.asarray(Image.open(data / "scene" / "scene_mask.png"))
    t_inf = time.time()
    scene_pred = sliding_window_predict(
        scene_rgb, predict_fn, NUM_CLASSES, patch=args.patch, stride=args.stride
    )
    scene_pred = clean_mask(scene_pred, min_size=args.min_size, median_k=3)
    print(f"[4/6] 整幅场景推理 {scene_rgb.shape} 耗时 {time.time() - t_inf:.1f}s")

    cm_scene = ConfusionMatrix(NUM_CLASSES).update(scene_pred, scene_gt)
    print(f"      场景整体 mIoU={cm_scene.miou():.3f}  OA={cm_scene.oa():.3f}")

    # ---------------- 5. 出图 ----------------
    summary = cm_scene.summary(CLASS_NAMES)
    poster = landcover_poster(
        scene_rgb, scene_pred, gsd=args.gsd,
        title="区域土地利用智能解译图",
        subtitle="合成样例数据 · 最小距离分类器基线 · 6 类地物自动提取",
        metrics=summary,
        model_name="MinDistance (baseline)",
        out_png=str(out / "demo_landcover_map.png"),
        out_pdf=str(out / "demo_landcover_map.pdf"),
        show_image_inset=True,
    )
    h, w = scene_pred.shape
    comp = comparison_figure(
        scene_rgb, scene_gt, scene_pred,
        tiles=[(80, 440, 120, 560), (1200, 1560, 2200, 2640)],
        out_path=str(out / "demo_comparison.png"),
    )
    conf = confusion_figure(cm_scene.mat, CARTO_NAMES, str(out / "demo_confusion.png"))
    print(f"[5/6] 成图: {poster}\n      {comp}\n      {conf}")

    # ---------------- 6. 存档 ----------------
    stats = area_stats(scene_pred, gsd=args.gsd, names=CARTO_NAMES)
    payload = {
        "method": "MinDistance baseline (RGB+ExG+brightness+saturation)",
        "dataset": str(data.name),
        "synthetic": True,
        "warning": "合成样例数据，仅验证流程，不得作为实验结果",
        "gsd_m": args.gsd,
        "val_tiles": len(val),
        "metrics_val": cm.summary(CLASS_NAMES),
        "metrics_scene": summary,
        "postprocess": {
            "min_size_px": args.min_size,
            "miou_before": round(cm_raw.miou(), 4),
            "miou_after": round(cm_cln.miou(), 4),
        },
        "area_stats": stats,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "metrics_baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[6/6] 指标存档 {out / 'metrics_baseline.json'}  总耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

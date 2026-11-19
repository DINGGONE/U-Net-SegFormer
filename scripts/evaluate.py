"""Evaluate a checkpoint: confusion matrix, metrics, optional sample points.

Usage:
    python scripts/evaluate.py --ckpt runs/loveda_r34/best.pt --data data/loveda"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsc.cartography import confusion_figure  # noqa: E402
from rsc.engine import validate  # noqa: E402
from rsc.infer import load_checkpoint, torch_predict_fn  # noqa: E402
from rsc.metrics import ConfusionMatrix  # noqa: E402
from rsc.palette import CARTO_NAMES, CLASS_NAMES, NUM_CLASSES, remap_labels  # noqa: E402
from rsc.tiling import sliding_window_predict  # noqa: E402


def point_based_eval(model, data_root: Path, preset: str, points_csv: Path,
                     device, patch: int, stride: int, num_classes: int) -> dict:
    """按样点核查：只对含样点的瓦片做一次推理，然后逐点比对。"""
    rows = list(csv.DictReader(open(points_csv, encoding="utf-8")))
    by_tile = defaultdict(list)
    for r in rows:
        by_tile[r["tile"]].append(r)

    fn = torch_predict_fn(model, device)
    correct, total = 0, 0
    per_class = defaultdict(lambda: {"n": 0, "hit": 0})
    misses = []
    for tile, rs in by_tile.items():
        img_path = data_root / "val" / "images" / f"{tile}.png"
        if not img_path.exists():
            img_path = data_root / "val" / "images" / f"{tile}.tif"
        img = np.asarray(Image.open(img_path).convert("RGB"))
        pred = sliding_window_predict(img, fn, num_classes, patch=patch, stride=stride)
        for r in rs:
            y, x, gt = int(r["row"]), int(r["col"]), int(r["class_id"])
            p = int(pred[y, x])
            total += 1
            per_class[gt]["n"] += 1
            if p == gt:
                correct += 1
                per_class[gt]["hit"] += 1
            elif len(misses) < 12:
                misses.append({"point": r["point_id"], "gt": gt, "pred": p})
    oa = correct / max(total, 1)
    return {
        "n_points": total,
        "oa": round(oa, 4),
        "per_class_oa": {CLASS_NAMES[c]: round(v["hit"] / max(v["n"], 1), 4)
                         for c, v in sorted(per_class.items())},
        "sample_misses": misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--points", default=None, help="验证样点 CSV（可选）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, dev)
    cfg = ckpt.get("cfg", {})
    preset = args.preset or cfg.get("data", {}).get("preset", "raw")
    num_classes = cfg.get("num_classes", NUM_CLASSES)
    out = Path(args.out or Path(args.ckpt).parent)

    # 最终评估用全量验证集（max_images=0 表示不限制）
    cm = validate(model, args.data, preset, num_classes, dev, patch=args.patch,
                  stride=args.stride, max_images=0)
    print(cm.format_table(CLASS_NAMES))
    summary = cm.summary(CLASS_NAMES)

    out.mkdir(parents=True, exist_ok=True)
    confusion_figure(cm.mat, CARTO_NAMES, str(out / "confusion_matrix.png"))
    payload = {"checkpoint": str(args.ckpt), "preset": preset, "pixel_based": summary}
    if args.points:
        pb = point_based_eval(model, Path(args.data), preset, Path(args.points), dev,
                              args.patch, args.stride, num_classes)
        payload["point_based"] = pb
        print(f"\n样点法：{pb['n_points']} 个点，OA={pb['oa']:.3f}")
        for k, v in pb["per_class_oa"].items():
            print(f"  {k:<8} {v:.3f}")
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"\n结果已存 {out / 'metrics.json'} / {out / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()

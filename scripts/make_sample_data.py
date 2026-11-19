"""Generate the synthetic sample dataset.

Usage:
    python scripts/make_sample_data.py"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsc.palette import CARTO_NAMES, CLASSES, NUM_CLASSES  # noqa: E402
from rsc.synth import FIELDS, make_tile  # noqa: E402


def save_pair(rgb: np.ndarray, mask: np.ndarray, img_path: Path, mask_path: Path) -> None:
    img_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(img_path, optimize=True)
    Image.fromarray(mask, mode="L").save(mask_path, optimize=True)


def stratified_points(mask: np.ndarray, n_per_class: int, rng: np.random.Generator, tile: str):
    """分层随机采样：每个类别抽 n 个点，这是遥感精度验证的标准做法。

    比"整幅逐像元算 IoU"更接近行业验收口径（GB/T 及多数行业规范用样点法），
    也便于野外核查 —— 每个点都能落到具体经纬度上去。
    """
    rows = []
    pid = 0
    for cid in range(NUM_CLASSES):
        ys, xs = np.nonzero(mask == cid)
        if len(ys) == 0:
            continue
        k = min(n_per_class, len(ys))
        sel = rng.choice(len(ys), size=k, replace=False)
        for i in sel:
            rows.append(
                {
                    "point_id": f"{tile}_{pid:05d}",
                    "tile": tile,
                    "row": int(ys[i]),
                    "col": int(xs[i]),
                    "class_id": int(cid),
                    "class_name": CLASSES[cid].name_en,
                }
            )
            pid += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "sample"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--points-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    S = args.size

    # ---------- 训练/验证瓦片 ----------
    stats = {}
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        for k in range(n):
            seed = int(rng.integers(0, 10**6))
            urban = (k % 3 == 0)          # 混合城乡场景，模仿 LoveDA 的 urban/rural
            rgb, mask = make_tile(S, seed=seed, urban=urban)
            name = f"{split}_{k:03d}"
            save_pair(rgb, mask, out / split / "images" / f"{name}.png",
                      out / split / "masks" / f"{name}.png")
            stats[name] = {"urban": urban, "seed": seed}

    # ---------- 大场景（滑窗推理 + 制图 demo）----------
    # 2 行 × 3 列拼接成 2048 × 3072 的"区域影像"
    tiles = []
    for k in range(6):
        rgb, mask = make_tile(S, seed=int(rng.integers(0, 10**6)), urban=(k % 2 == 0))
        tiles.append((rgb, mask))
    rows_rgb = [np.concatenate([tiles[r * 3 + c][0] for c in range(3)], axis=1) for r in range(2)]
    rows_msk = [np.concatenate([tiles[r * 3 + c][1] for c in range(3)], axis=1) for r in range(2)]
    scene_rgb = np.concatenate(rows_rgb, axis=0)
    scene_msk = np.concatenate(rows_msk, axis=0)
    save_pair(
        scene_rgb, scene_msk,
        out / "scene" / "scene_rgb.png",
        out / "scene" / "scene_mask.png",
    )
    print(f"scene: {scene_rgb.shape}")

    # ---------- 验证样点 ----------
    rows = []
    for k in range(args.n_val):
        name = f"val_{k:03d}"
        mask = np.asarray(Image.open(out / "val" / "masks" / f"{name}.png"))
        rows.extend(stratified_points(mask, args.points_per_class, rng, name))
    import csv

    with open(out / "sample_points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- 元数据 ----------
    meta = {
        "num_classes": NUM_CLASSES,
        "ignore_index": 255,
        "classes": [
            {"id": c.id, "name": c.name, "name_en": c.name_en, "color": list(c.color),
             "carto_name": CARTO_NAMES[c.id]}
            for c in CLASSES
        ],
        "synthetic": True,
        "note": "合成数据，仅用于打通流程，不得作为实验结果。",
        "tiles": len(stats),
        "scene_shape": list(scene_rgb.shape),
    }
    (out / "classes.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done → {out}  (tiles={len(stats)}, points={len(rows)})")


if __name__ == "__main__":
    main()

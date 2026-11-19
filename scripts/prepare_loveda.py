"""Convert the official LoveDA layout into this project's layout.

Usage:
    python scripts/prepare_loveda.py --src D:/data/LoveDA --dst data/loveda"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsc.palette import remap_labels  # noqa: E402

SPLITS = {"train": ["Train"], "val": ["Val"], "test": ["Test"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LoveDA 解压目录")
    ap.add_argument("--dst", default=str(ROOT / "data" / "loveda"))
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    total = 0
    for split, dirs in SPLITS.items():
        if split == "test" and args.skip_test:
            continue
        for d in dirs:
            img_root = src / d
            if not img_root.exists():
                print(f"[skip] {img_root} 不存在")
                continue
            for domain in ("Urban", "Rural"):
                imgs = sorted((img_root / domain / "images_png").glob("*.png"))
                if not imgs:
                    continue
                out_img = dst / split / "images"
                out_msk = dst / split / "masks"
                out_img.mkdir(parents=True, exist_ok=True)
                out_msk.mkdir(parents=True, exist_ok=True)
                for p in imgs:
                    name = f"{domain.lower()}_{p.stem}"
                    Image.open(p).convert("RGB").save(out_img / f"{name}.png", optimize=True)
                    mpath = img_root / domain / "masks_png" / p.name
                    if mpath.exists():
                        m = np.asarray(Image.open(mpath))
                        if m.ndim == 3:
                            m = m[..., 0]
                        Image.fromarray(remap_labels(m, "loveda"), mode="L").save(
                            out_msk / f"{name}.png", optimize=True)
                    total += 1
                print(f"{split}/{domain}: {len(imgs)} 张")

    # 统计一下类别分布，方便设类别权重与 sanity check
    print(f"\n共处理 {total} 张，输出到 {dst}")
    for split in ("train", "val"):
        msks = sorted((dst / split / "masks").glob("*.png")) if (dst / split / "masks").exists() else []
        if not msks:
            continue
        cnt = np.zeros(8, np.int64)
        for p in msks:
            m = np.asarray(Image.open(p))
            cnt += np.bincount(m.ravel(), minlength=8)
        cnt = cnt[:7]   # 0..6（其中 255 已映射）
        tot = cnt.sum()
        print(f"{split} 类别像素占比: " + ", ".join(f"{i}:{c / tot:.3f}" for i, c in enumerate(cnt)))


if __name__ == "__main__":
    main()

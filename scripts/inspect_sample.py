"""看一眼"样例数据"里到底是什么：影像 / 标签 / 放大局部 / 类别占比。

    python scripts/inspect_sample.py
    python scripts/inspect_sample.py --tile data/sample/val/images/val_001.png

输出 outputs/sample_inspect.png（4 联图），并打印这份数据的元信息。
"""
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

from rsc.cartography import setup_fonts  # noqa: E402
from rsc.palette import CLASSES, CARTO_NAMES, NUM_CLASSES, colorize  # noqa: E402


def densest_window(mask: np.ndarray, size: int, weights=None) -> tuple:
    """找"地物最丰富"的 size×size 窗口，用来做放大展示。

    默认给建筑更高权重 —— 建筑 + 投影最能说明"标签和影像怎么对应"。
    """
    weights = weights or {1: 6.0, 2: 1.5, 3: 1.0, 5: 1.0}
    m = np.zeros(mask.shape, np.float32)
    for c, wt in weights.items():
        m[mask == c] = wt
    cs = np.cumsum(np.cumsum(m, 0), 1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = mask.shape
    best, box = -1, (0, size, 0, size)
    for y in range(0, h - size + 1, 8):
        for x in range(0, w - size + 1, 8):
            s = cs[y + size, x + size] - cs[y, x + size] - cs[y + size, x] + cs[y, x]
            if s > best:
                best, box = s, (y, y + size, x, x + size)
    return box


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default=str(ROOT / "data" / "sample" / "val" / "images" / "val_000.png"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sample_inspect.png"))
    ap.add_argument("--zoom", type=int, default=128)
    args = ap.parse_args()

    img_path = Path(args.tile)
    msk_path = img_path.parent.parent / "masks" / img_path.name
    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    msk = np.asarray(Image.open(msk_path))
    if msk.ndim == 3:
        msk = msk[..., 0]
    h, w = msk.shape

    setup_fonts()
    import matplotlib.pyplot as plt

    y0, y1, x0, x1 = densest_window(msk, args.zoom)
    fig = plt.figure(figsize=(13.5, 8.2), dpi=180)
    fig.suptitle("样例数据是什么：一张合成的高分遥感瓦片 + 配套标签", fontsize=16,
                 fontweight="bold", y=0.975)

    # ① 影像
    ax = fig.add_axes([0.03, 0.36, 0.30, 0.53])
    ax.imshow(rgb)
    ax.set_title(f"① 影像  {w}×{h} px, 8bit RGB（0.3 m/px ≈ {w * 0.3:.0f} m 见方）",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ② 标签
    ax = fig.add_axes([0.35, 0.36, 0.30, 0.53])
    ax.imshow(colorize(msk), interpolation="nearest")
    ax.set_title("② 标签  单通道 PNG，值 0~5", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ③ 类别占比
    ax = fig.add_axes([0.70, 0.40, 0.27, 0.45])
    frac = np.array([(msk == c).sum() for c in range(NUM_CLASSES)]) / msk.size * 100
    order = np.argsort(frac)
    ax.barh(np.arange(NUM_CLASSES), frac[order],
            color=[np.array(CLASSES[i].color) / 255 for i in order])
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_yticklabels([CARTO_NAMES[i] for i in order], fontsize=10)
    ax.set_xlabel("占全图比例 (%)", fontsize=10)
    ax.set_title("③ 类别占比（很偏，真实数据也这样）", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ④ 放大影像
    ax = fig.add_axes([0.03, 0.06, 0.30, 0.24])
    ax.imshow(rgb[y0:y1, x0:x1], interpolation="nearest")
    ax.set_title("④ 放大 4×：白屋顶、深色投影、裸土地表", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_axes([0.35, 0.06, 0.30, 0.24])
    ax.imshow(colorize(msk[y0:y1, x0:x1]), interpolation="nearest")
    ax.set_title("⑤ 同一位置的标签（投影下的像素仍算原地类）", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # ⑥ 说明
    ax = fig.add_axes([0.68, 0.06, 0.30, 0.25])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#f5f6f8",
                               edgecolor="#d8dbe0", linewidth=0.8))
    notes = [
        "这是一份【合成】数据，不是真实卫星影像。",
        "没有传感器、没有拍摄地点和日期。",
        "它按 LoveDA 的规格生成：",
        "  1024×1024 / RGB / 单通道标签 / 0.3 m。",
        "用途：跑通流程、单测、演示。",
        "不可用于：写论文的实验精度。",
    ]
    for i, t in enumerate(notes):
        ax.text(0.06, 0.86 - i * 0.145, t, fontsize=9.6, color="#333333", va="center")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"已生成 {out}")
    print(f"影像: {img_path.name}  {w}×{h} px, {rgb.dtype}, {os_size(img_path):.1f} KB")
    print(f"标签: {msk_path.name}  唯一值 {np.unique(msk).tolist()}")
    print("类别占比: " + ", ".join(f"{CARTO_NAMES[c]}={frac[c]:.1f}%" for c in range(NUM_CLASSES)))


def os_size(p: Path) -> float:
    return p.stat().st_size / 1024


if __name__ == "__main__":
    main()

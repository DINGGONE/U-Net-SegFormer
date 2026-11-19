"""制图：把分割结果做成一张能进报告/答辩的"区域土地利用智能解译图"。

这一层是"算法 + 行业应用"闭环的出口。评审/甲方看的不是 mIoU，而是
"这张图能不能读、有没有图例、有没有面积统计、比例尺对不对"。

包含 4 个成品图函数：
    landcover_poster     区域土地利用智能解译图（主成果图，含图例/面积统计/精度）
    comparison_figure    "影像 | 真值 | 预测" 三联图（论文/答辩常用）
    confusion_figure     混淆矩阵热力图（误差分析）
    history_figure       训练曲线（loss / mIoU），用于说明收敛性

依赖 matplotlib + numpy，不依赖 cv2/rasterio，任何环境都能出图。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # 无界面渲染；必须在 pyplot 之前设置

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle

from .palette import CARTO_NAMES, CLASSES, IGNORE_INDEX, NUM_CLASSES, colorize, legend_entries
from .postprocess import area_stats

CJK_FONTS = ["Microsoft YaHei", "SimHei", "DengXian", "Noto Sans CJK SC", "Source Han Sans SC"]


def setup_fonts() -> str:
    """挑一个能显示中文的字体，返回实际使用的字体名。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return "sans-serif"


# --------------------------------------------------------------------------
# 地图要素
# --------------------------------------------------------------------------
def _nice_length(gsd: float, target_m: float = 300.0) -> float:
    """把比例尺长度取整到 1/2/5 × 10^n 米。"""
    raw = target_m
    exp = np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        cand = m * 10 ** exp
        if cand >= raw * 0.6:
            return float(cand)
    return float(10 ** (exp + 1))


def draw_scalebar(
    ax,
    gsd: float,
    length_m: Optional[float] = None,
    loc: Tuple[float, float] = (0.04, 0.045),
    color: str = "black",
    fontsize: int = 9,
    n_seg: int = 2,
) -> float:
    """在 axes 左下角画分段比例尺，返回所用长度（米）。

    注意：影像轴是 y 轴反向的（imshow 语义），"屏幕下方"对应数据坐标 y 较大处。
    """
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    W, H = x1 - x0, y1 - y0
    length_m = length_m or _nice_length(gsd, target_m=W * gsd * 0.22)
    seg_px = (length_m / n_seg) / gsd
    bx = x0 + loc[0] * W
    by = y1 - loc[1] * H - H * 0.012
    bar_h = H * 0.012

    ax.add_patch(Rectangle((bx, by), seg_px * n_seg, bar_h, facecolor="white",
                           edgecolor=color, linewidth=0.8, zorder=6))
    for i in range(n_seg):
        ax.add_patch(
            Rectangle((bx + i * seg_px, by), seg_px, bar_h,
                      facecolor=color if i % 2 == 0 else "white",
                      edgecolor=color, linewidth=0.8, zorder=7)
        )
    for i in range(n_seg + 1):
        ax.text(bx + i * seg_px, by - bar_h * 0.45,
                f"{int(round(length_m * i / n_seg))}", ha="center", va="bottom",
                fontsize=fontsize, color=color, zorder=8)
    ax.text(bx + seg_px * n_seg + W * 0.008, by + bar_h * 0.5,
            "m", ha="left", va="center", fontsize=fontsize, color=color, zorder=8)
    return length_m


def draw_north_arrow(ax, loc: Tuple[float, float] = (0.94, 0.88), size: float = 0.05, color="black"):
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    W, H = x1 - x0, y1 - y0
    cx = x0 + loc[0] * W
    cy = y1 - loc[1] * H
    s = size * H
    # 屏幕上"向上"= 数据坐标 y 减小；va="bottom" 让文字朝屏幕上方生长
    ax.add_patch(FancyArrow(cx, cy + s * 0.55, 0, -s, width=s * 0.13,
                            head_width=s * 0.34, head_length=s * 0.40,
                            length_includes_head=True, color=color, zorder=8))
    ax.text(cx, cy - s * 0.52, "N", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color, zorder=8)


# --------------------------------------------------------------------------
# 主成果图
# --------------------------------------------------------------------------
def landcover_poster(
    rgb: np.ndarray,
    pred: np.ndarray,
    gsd: float = 0.3,
    title: str = "区域土地利用智能解译图",
    subtitle: str = "基于深度学习语义分割的 6 类地物自动提取",
    metrics: Optional[Dict] = None,
    model_name: str = "SegFormer-B2 / U-Net",
    out_png: str = "outputs/landcover_map.png",
    out_pdf: Optional[str] = None,
    dpi: int = 220,
    show_image_inset: bool = False,
) -> str:
    """出主成果图：左地图 + 右图例/面积统计/精度信息。

    Parameters
    ----------
    rgb : (H,W,3) uint8 原始影像（用作底图可选、及 inset）
    pred : (H,W) 类别图
    gsd : 米/像元，用于比例尺与面积
    metrics : ``ConfusionMatrix.summary()`` 的输出，可为 None
    """
    setup_fonts()
    h, w = pred.shape
    stats = area_stats(pred, gsd=gsd, names=CARTO_NAMES)

    fig = plt.figure(figsize=(15.5, 9.2), dpi=dpi)
    fig.patch.set_facecolor("white")

    # ---- 标题区 ----
    fig.text(0.035, 0.955, title, fontsize=23, fontweight="bold", va="top", color="#1a1a1a")
    fig.text(0.035, 0.905, subtitle, fontsize=11.5, va="top", color="#555555")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.885, 0.885], color="#c8c8c8", linewidth=1.0))

    # ---- 主地图 ----
    ax = fig.add_axes([0.035, 0.075, 0.645, 0.785])
    ax.imshow(colorize(pred), interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.9)
        s.set_color("#333333")

    draw_scalebar(ax, gsd)
    draw_north_arrow(ax)

    if show_image_inset:
        axi = ax.inset_axes([0.76, 0.05, 0.21, 0.21])
        axi.imshow(rgb)
        axi.set_xticks([]); axi.set_yticks([])
        axi.set_title("原影像", fontsize=8, pad=2)
        for s in axi.spines.values():
            s.set_color("white"); s.set_linewidth(1.2)

    # ---- 图例 ----
    axl = fig.add_axes([0.705, 0.545, 0.28, 0.315])
    axl.axis("off")
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    axl.text(0.0, 0.97, "图  例", fontsize=12.5, fontweight="bold", va="top", color="#1a1a1a")
    y = 0.83
    for (color, name), row in zip(legend_entries(CARTO_NAMES), stats):
        axl.add_patch(Rectangle((0.0, y - 0.035), 0.075, 0.072,
                                facecolor=color, edgecolor="#666666", linewidth=0.6))
        axl.text(0.10, y - 0.005, name, fontsize=10.5, va="center", color="#222222")
        axl.text(0.72, y - 0.005, f"{row['ratio'] * 100:5.1f}%", fontsize=10.5,
                 va="center", ha="right", color="#444444", family="DejaVu Sans")
        axl.text(0.99, y - 0.005, f"{row['area_ha']:,.1f} ha", fontsize=9.5,
                 va="center", ha="right", color="#777777", family="DejaVu Sans")
        y -= 0.128

    # ---- 面积统计条形图 ----
    axb = fig.add_axes([0.735, 0.275, 0.235, 0.215])
    ratios = np.array([r["ratio"] * 100 for r in stats])
    colors = [np.array(c.color) / 255 for c in CLASSES]
    order = np.argsort(ratios)
    axb.barh(np.arange(NUM_CLASSES), ratios[order], color=[colors[i] for i in order],
             edgecolor="white", linewidth=0.6)
    axb.set_yticks(np.arange(NUM_CLASSES))
    axb.set_yticklabels([CARTO_NAMES[i] for i in order], fontsize=8.5)
    axb.tick_params(axis="x", labelsize=8)
    axb.set_xlabel("面积占比 (%)", fontsize=8.5)
    axb.set_title("地物构成", fontsize=10.5, fontweight="bold", loc="left", pad=4)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.grid(axis="x", alpha=0.25, linewidth=0.6)

    # ---- 精度 / 信息卡 ----
    axi2 = fig.add_axes([0.705, 0.075, 0.28, 0.155])
    axi2.axis("off")
    axi2.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f5f6f8", edgecolor="#d8dbe0", linewidth=0.8))
    lines = []
    if metrics:
        lines.append(f"mIoU  {metrics.get('mIoU', float('nan')):.3f}     "
                     f"mF1  {metrics.get('mF1', float('nan')):.3f}")
        lines.append(f"OA    {metrics.get('OA', float('nan')):.3f}     "
                     f"Kappa {metrics.get('Kappa', float('nan')):.3f}")
    lines.append(f"覆盖面积  {h * w * gsd ** 2 / 1e4:,.1f} ha    "
                 f"分辨率  {gsd} m")
    lines.append(f"模型  {model_name}")
    for i, ln in enumerate(lines):
        axi2.text(0.05, 0.80 - i * 0.24, ln, fontsize=9.2, color="#333333", va="center")

    fig.text(0.035, 0.022,
             "数据来源：高分遥感影像  |  分类体系：建筑 / 道路 / 水体 / 植被 / 裸土  |  "
             "制图：语义分割模型 + 碎斑过滤后处理",
             fontsize=8, color="#888888")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------
# 三联对比图
# --------------------------------------------------------------------------
def comparison_figure(
    rgb: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    tiles: Sequence[Tuple[int, int, int, int]] = (),
    out_path: str = "outputs/comparison.png",
    titles: Sequence[str] = ("原始影像", "人工标注", "模型预测"),
    dpi: int = 200,
) -> str:
    """影像 / 真值 / 预测 三联图。tiles 可给多个 (y0,y1,x0,x1) 局部放大。

    典型用法：一张全景 + 两行局部放大（说明模型在建筑密集区、水边的好与坏）。
    """
    setup_fonts()
    tiles = list(tiles) or [(0, rgb.shape[0], 0, rgb.shape[1])]
    n = len(tiles) + 1  # 第一行全景 + 若干局部
    ncol = 3
    fig, axes = plt.subplots(n, ncol, figsize=(4.2 * ncol, 3.4 * n), dpi=dpi)
    axes = np.atleast_2d(axes)

    rows = [(rgb, "img"), (gt, "mask"), (pred, "mask")]
    # 第一行：全景
    for j, (arr, kind) in enumerate(rows):
        ax = axes[0, j]
        ax.imshow(arr if kind == "img" else colorize(arr), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[j], fontsize=11, fontweight="bold")
        for s in ax.spines.values():
            s.set_color("#888888"); s.set_linewidth(0.8)

    for i, (y0, y1, x0, x1) in enumerate(tiles, start=1):
        sl = (slice(y0, y1), slice(x0, x1))
        for j, (arr, kind) in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(arr[sl] if kind == "img" else colorize(arr[sl]), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"局部 {i}", fontsize=9)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 混淆矩阵图
# --------------------------------------------------------------------------
def confusion_figure(cm: np.ndarray, names: Sequence[str], out_path: str,
                     normalize: bool = True, dpi: int = 200) -> str:
    setup_fonts()
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm_n = np.divide(cm, np.maximum(row, 1), out=np.zeros_like(cm))
    else:
        cm_n = cm

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=dpi)
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1 if normalize else cm_n.max())
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("预测类别", fontsize=11); ax.set_ylabel("真实类别", fontsize=11)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm_n[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(cm[i, j])}",
                    ha="center", va="center", fontsize=9,
                    color="white" if v > (0.55 if normalize else cm_n.max() * 0.55) else "#333333")
    ax.set_title("混淆矩阵（按真实类别归一化）" if normalize else "混淆矩阵", fontsize=12,
                 fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.text(0.01, 0.01, "对角线 = 该类别的召回率；非对角线 = 主要混淆方向（用于误差分析）",
             fontsize=8, color="#777777")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 训练曲线
# --------------------------------------------------------------------------
def history_figure(history: Dict[str, List[float]], out_path: str, dpi: int = 200) -> str:
    setup_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=dpi)
    ax = axes[0]
    ax.plot(history.get("train_loss", []), label="train loss", color="#d62728")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val loss", color="#1f77b4")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("损失曲线", fontsize=11, fontweight="bold")

    ax = axes[1]
    if "val_miou" in history:
        ax.plot(history["val_miou"], label="val mIoU", color="#2ca02c")
    if "val_mf1" in history:
        ax.plot(history["val_mf1"], label="val mF1", color="#ff7f0e", linestyle="--")
    if "lr" in history:
        ax2 = ax.twinx()
        ax2.plot(history["lr"], color="#999999", alpha=0.6, linewidth=0.9, label="lr")
        ax2.set_ylabel("learning rate", color="#999999")
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax.set_title("验证集精度", fontsize=11, fontweight="bold")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path

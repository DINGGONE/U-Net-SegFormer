"""无标注影像的应急方案：K-means 非监督分类（ISODATA 的现代版）。

适用场景：手上只有影像、没有标注、也还没有训练好的模型。
它不"认识"地物，只是把颜色/纹理相近的像素聚成 k 类，再按光谱经验给每类贴标签。
精度远不如深度学习（目视可用，不能当成果精度），但**10 秒就能出一张合理的图**，
而且能立刻暴露"这张影像到底好不好分"。

    python scripts/unsupervised_baseline.py --src data/raw/实验影像.tif --gsd 0.5
    python scripts/unsupervised_baseline.py --src xxx.tif --k 10 --downsample 4

输出::

    outputs/unsupervised/xxx_clusters.json    每个簇的光谱统计（可核查）
    outputs/unsupervised/xxx_landcover.tif    类别栅格（保留 CRS）
    outputs/unsupervised/xxx_landcover.png    彩色分类图
    outputs/unsupervised/xxx_poster.png       成果图
"""
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
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from rsc.cartography import landcover_poster  # noqa: E402
from rsc.palette import CARTO_NAMES, NUM_CLASSES, colorize  # noqa: E402
from rsc.postprocess import area_stats, clean_mask  # noqa: E402

CLASS_NAMES_EN = ["background", "building", "road", "water", "vegetation", "bare_soil"]


# --------------------------------------------------------------------------
def spectral_features(rgb: np.ndarray) -> np.ndarray:
    """(N,3) 0~255 → (N,F) 光谱特征。"""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    exg = 2 * g - r - b                     # 超绿：植被
    bright = (r + g + b) / 3.0
    mx, mn = x.max(1), x.min(1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)  # 饱和度：屋顶/土壤高，道路/建筑低
    return np.stack([r, g, b, exg, bright, sat], 1)


def cluster_stats(rgb: np.ndarray, labels: np.ndarray, k: int):
    rows = []
    for c in range(k):
        m = labels == c
        n = int(m.sum())
        if n == 0:
            rows.append(None)
            continue
        px = rgb[m].astype(np.float32)
        f = spectral_features(px).mean(0)
        rows.append({
            "cluster": c, "pixels": n,
            "R": round(float(f[0] * 255), 1), "G": round(float(f[1] * 255), 1),
            "B": round(float(f[2] * 255), 1),
            "ExG": round(float(f[3]), 3), "bright": round(float(f[4]), 3),
            "sat": round(float(f[5]), 3),
        })
    return rows


def map_clusters_to_classes(stats, k: int) -> dict:
    """按光谱经验把 k 个簇映射到 6 类。

    规则（无 NIR 时的经验做法）：
      植被  ← ExG 最高
      水体  ← 最暗 + 偏蓝（B 明显大于 R）
      建筑  ← 最亮 + 低饱和（水泥屋顶、白墙）
      裸土  ← 偏红黄 + 中高亮（收割后的农田、裸地）
      道路  ← 中灰 + 极低饱和（高速路面）
      其他  ← 剩下对比度低的
    多个簇落进同一类时，保留"最像"的那个，其余顺延到次优类。
    """
    valid = [s for s in stats if s]
    used = set()
    mapping = {}

    def assign(cluster, cls):
        mapping[cluster] = cls
        used.add(cluster)

    # 1) 植被：ExG 最高的 2 个簇
    for s in sorted(valid, key=lambda x: -x["ExG"])[:2]:
        if s["cluster"] not in used:
            assign(s["cluster"], 4)

    # 2) 水体：必须"蓝通道最大"且很暗 —— 否则深色林地（G>B）会被错当成水
    for s in sorted(valid, key=lambda x: x["bright"]):
        if s["cluster"] in used:
            continue
        if s["B"] >= s["R"] and s["B"] >= s["G"] and s["bright"] < 0.35 and s["ExG"] < 0.06:
            assign(s["cluster"], 3)
            break

    # 3) 建筑：最亮的簇（低饱和优先）
    for s in sorted(valid, key=lambda x: (-x["bright"], x["sat"])):
        if s["cluster"] not in used:
            assign(s["cluster"], 1)
            break

    # 4) 裸土/农田：R 明显大于 B，亮度中高
    for s in sorted(valid, key=lambda x: -(x["R"] - x["B"])):
        if s["cluster"] not in used and s["R"] - s["B"] > 8:
            assign(s["cluster"], 5)
            break

    # 5) 道路：中等亮度、极低饱和
    for s in sorted(valid, key=lambda x: (x["sat"], -x["bright"])):
        if s["cluster"] not in used and s["sat"] < 0.18:
            assign(s["cluster"], 2)
            break

    # 剩余：低饱和归建筑/道路，否则归"其他"
    for s in valid:
        if s["cluster"] in used:
            continue
        if s["sat"] < 0.22 and s["bright"] > 0.42:
            mapping[s["cluster"]] = 1 if s["bright"] > 0.5 else 2
        elif s["ExG"] > 0.0:
            mapping[s["cluster"]] = 4
        else:
            mapping[s["cluster"]] = 0
    return mapping


# --------------------------------------------------------------------------
def main() -> None:
    import rasterio
    from sklearn.cluster import MiniBatchKMeans

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "unsupervised"))
    ap.add_argument("--k", type=int, default=8, help="聚类数，6~10 比较合适")
    ap.add_argument("--downsample", type=int, default=4, help="训练聚类时每隔几个像素取一个")
    ap.add_argument("--gsd", type=float, default=None, help="不填则从 GeoTIFF 读")
    ap.add_argument("--min-size", type=int, default=120)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 读图 ----
    profile = None
    if src.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(str(src)) as ds:
            arr = ds.read()
            profile = ds.profile.copy()
            gsd = args.gsd or abs(ds.transform.a)
            crs = ds.crs
    else:
        arr = np.transpose(np.asarray(Image.open(src).convert("RGB")), (2, 0, 1))
        gsd = args.gsd or 0.5
        crs = None
    if arr.shape[0] > 3:
        arr = arr[:3]
    rgb = np.transpose(arr[:3], (1, 2, 0)).astype(np.uint8)
    h, w = rgb.shape[:2]
    print(f"影像 {w}×{h}，{gsd} m/px，覆盖 {w * h * gsd ** 2 / 1e4:,.0f} 公顷")

    # ---- 聚类 ----
    flat = rgb.reshape(-1, 3)
    step = max(1, args.downsample)
    sample = flat[::step]
    feats = spectral_features(sample)
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    km = MiniBatchKMeans(n_clusters=args.k, random_state=0, n_init=10, batch_size=4096)
    km.fit((feats - mu) / sd)
    print(f"K-means 完成：{args.k} 个簇，样本 {len(sample):,} 像素")

    # ---- 全图分配（按簇中心距离，分块算，避免爆内存） ----
    centers = km.cluster_centers_
    labels = np.empty(len(flat), np.int16)
    block = 2_000_000
    for i in range(0, len(flat), block):
        f = (spectral_features(flat[i:i + block]) - mu) / sd
        d = ((f[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels[i:i + block] = d.argmin(1)
    labels = labels.reshape(h, w)

    stats = cluster_stats(rgb, labels, args.k)
    mapping = map_clusters_to_classes(stats, args.k)

    print("\n  簇  像素占比   R    G    B   ExG   亮度  饱和  → 映射")
    print("  " + "-" * 66)
    for s in stats:
        if not s:
            continue
        cls = mapping.get(s["cluster"], 0)
        print(f"  {s['cluster']:>2}  {s['pixels'] / labels.size * 100:6.2f}%  "
              f"{s['R']:5.1f}{s['G']:5.1f}{s['B']:5.1f}  {s['ExG']:+.2f}  "
              f"{s['bright']:.2f}  {s['sat']:.2f}  → {cls} {CARTO_NAMES[cls]}")

    pred = np.zeros_like(labels, dtype=np.uint8)
    for c, cls in mapping.items():
        pred[labels == c] = cls
    pred = clean_mask(pred, min_size=args.min_size, median_k=3)

    # ---- 保存 ----
    stem = src.stem
    Image.fromarray(colorize(pred)).save(out / f"{stem}_landcover.png")
    np.save(out / f"{stem}_landcover.npy", pred)
    (out / f"{stem}_clusters.json").write_text(
        json.dumps({"k": args.k, "gsd": gsd, "stats": stats, "mapping": mapping},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    if profile is not None:
        profile.update(count=1, dtype="uint8", compress="lzw", nodata=255)
        with rasterio.open(out / f"{stem}_landcover.tif", "w", **profile) as dst:
            dst.write(pred[None])
        print(f"\n  GeoTIFF: {out / f'{stem}_landcover.tif'}  CRS={crs}")

    stats_area = area_stats(pred, gsd=gsd, names=CARTO_NAMES)
    for r in stats_area:
        print(f"  {r['name']:<8} {r['ratio'] * 100:5.1f}%  {r['area_ha']:8.1f} ha")

    landcover_poster(
        rgb, pred, gsd=gsd,
        title="区域土地利用解译图（非监督分类）",
        subtitle=f"K-means {args.k} 簇 · {src.name} · 无标注、无精度评价",
        model_name=f"K-means (k={args.k}, 非监督)",
        out_png=str(out / f"{stem}_poster.png"),
        out_pdf=str(out / f"{stem}_poster.pdf"),
        show_image_inset=True,
    )
    print(f"\n  成果图: {out / f'{stem}_poster.png'}")
    print("  ⚠️ 非监督分类没有精度指标，只能目视判断；正式成果请用训练好的模型。")


if __name__ == "__main__":
    main()

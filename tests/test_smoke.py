"""冒烟测试：不需要 torch，验证"标签映射 / 指标 / 切块拼接 / 后处理"这些最容易错的底层。

运行（任意 python 环境，只要 numpy + pillow）::

    python tests/test_smoke.py
    # 或
    pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsc.metrics import ConfusionMatrix  # noqa: E402
from rsc.palette import NUM_CLASSES, build_lut, remap_labels  # noqa: E402
from rsc.postprocess import area_stats, clean_mask, remove_small_objects  # noqa: E402
from rsc.tiling import iter_boxes, sliding_window_predict, tile_starts  # noqa: E402


# --------------------------------------------------------------------------
def test_label_mapping():
    lut = build_lut("loveda")
    assert lut[1] == 1, "建筑"
    assert lut[2] == 2, "道路"
    assert lut[3] == 3, "水体"
    assert lut[4] == 5, "LoveDA barren → 裸土"
    assert lut[5] == 4 and lut[6] == 4, "forest/agriculture → 植被"
    assert lut[0] == 255, "背景 → 忽略"

    m = np.array([[0, 1], [5, 6]], dtype=np.uint8)
    out = remap_labels(m, "loveda")
    assert out.tolist() == [[255, 1], [4, 4]]

    # Vaihingen: 不透水面算道路，车辆忽略
    assert build_lut("vaihingen")[0] == 2
    assert build_lut("vaihingen")[4] == 255


def test_metrics_perfect_and_known():
    gt = np.array([[0, 0, 1, 1], [2, 2, 2, 2]], np.uint8)
    cm = ConfusionMatrix(6).update(gt, gt)
    assert abs(cm.miou() - 1.0) < 1e-9
    assert abs(cm.oa() - 1.0) < 1e-9
    assert abs(cm.kappa() - 1.0) < 1e-9

    # 已知小例子：4 个像素里 3 对 1 错
    #   pred = [0,0,0,1], true = [0,0,1,1]
    #   class0: tp=2 fp=1 fn=0 → IoU 2/3
    #   class1: tp=1 fp=0 fn=1 → IoU 1/2
    pred = np.array([0, 0, 0, 1], np.uint8)
    true = np.array([0, 0, 1, 1], np.uint8)
    cm = ConfusionMatrix(6).update(pred, true)
    iou = cm.iou()
    assert abs(iou[0] - 2 / 3) < 1e-9, iou[0]
    assert abs(iou[1] - 0.5) < 1e-9, iou[1]
    assert abs(cm.miou() - (2 / 3 + 0.5) / 2) < 1e-9, cm.miou()
    assert abs(cm.oa() - 0.75) < 1e-9


def test_metrics_ignore_index():
    pred = np.array([1, 1, 1], np.uint8)
    gt = np.array([1, 1, 255], np.uint8)
    cm = ConfusionMatrix(6).update(pred, gt)
    assert cm.mat.sum() == 2, "255 不应计入混淆矩阵"
    assert abs(cm.miou() - 1.0) < 1e-9


def test_metrics_f1_matches_formula():
    pred = np.array([0, 1, 1, 2], np.uint8)
    gt = np.array([0, 1, 2, 2], np.uint8)
    cm = ConfusionMatrix(3).update(pred, gt)
    f1 = cm.f1()
    # class1: tp=1 fp=1 fn=0 → P=0.5 R=1 → F1=2/3 ; class2: tp=1 fp=0 fn=1 → 2/3
    assert abs(f1[1] - 2 / 3) < 1e-9, f1[1]
    assert abs(f1[2] - 2 / 3) < 1e-9, f1[2]
    assert abs(cm.recall()[2] - 0.5) < 1e-9, "class2 漏检 1 个 → recall 0.5"


# --------------------------------------------------------------------------
def test_tiles_cover_fully():
    for h, w, patch, stride in [(1000, 1200, 512, 384), (100, 100, 512, 256), (512, 512, 512, 256)]:
        boxes = list(iter_boxes(h, w, patch, stride))
        cover = np.zeros((h, w), np.int32)
        for y0, y1, x0, x1 in boxes:
            cover[y0:y1, x0:x1] += 1
        assert cover.min() >= 1, f"{h}x{w} 有像素没被覆盖"
        assert cover.max() <= 8, "重叠次数异常"


def test_sliding_window_identity():
    """若模型完美（返回 GT 的 one-hot），拼接结果必须逐像元等于 GT。"""
    rng = np.random.default_rng(0)
    h, w = 700, 900
    gt = rng.integers(0, NUM_CLASSES, (h, w)).astype(np.uint8)

    # 假模型：把 GT、坐标编码进输入的三个通道，模型据此"回查"答案。
    # 这样只要拼接坐标有半像素错位，测试就会失败 —— 正是我们要防的 bug。
    yy, xx = np.mgrid[0:h, 0:w]
    pack = np.stack([gt.astype(np.float32) / 10.0, yy.astype(np.float32), xx.astype(np.float32)], -1)

    def index_fn(batch_nchw):
        b = batch_nchw.shape[0]
        outs = np.zeros((b, NUM_CLASSES, batch_nchw.shape[2], batch_nchw.shape[3]), np.float32)
        for i in range(b):
            ys = batch_nchw[i, 1].astype(int)
            xs = batch_nchw[i, 2].astype(int)
            cls = np.rint(batch_nchw[i, 0] * 10).astype(int)
            outs[i][cls, np.arange(cls.shape[0])[:, None], np.arange(cls.shape[1])[None, :]] = 10.0
        return outs

    pred = sliding_window_predict(pack, index_fn, NUM_CLASSES, patch=256, stride=192)
    assert pred.shape == (h, w)
    assert (pred == gt).mean() == 1.0, f"拼接后有 {(pred != gt).mean():.3f} 的像素错位"


def test_sliding_window_small_image():
    img = np.zeros((50, 70, 3), np.float32)
    pred = sliding_window_predict(img, lambda b: np.zeros((len(b), NUM_CLASSES, 256, 256), np.float32),
                                  NUM_CLASSES, patch=256)
    assert pred.shape == (50, 70), "小于 patch 的影像应被 pad 后裁回原尺寸"


# --------------------------------------------------------------------------
def test_remove_small_objects():
    # 一片植被里的 1 像素"建筑"噪点，应被邻域植被吞掉
    m = np.full((64, 64), 4, np.uint8)
    m[40, 40] = 1
    out = remove_small_objects(m, min_size=10)
    assert out[40, 40] == 4, f"碎斑应被邻域类别填充，实际 {out[40, 40]}"
    assert (out == 4).all()

    # 900 像元的建筑图斑不应被删
    m2 = np.full((64, 64), 4, np.uint8)
    m2[10:40, 10:40] = 1
    out2 = remove_small_objects(m2, min_size=10)
    assert (out2 == m2).all(), f"大面积图斑被误删 {(out2 != m2).sum()} 像素"


def test_clean_mask_and_area_stats():
    rng = np.random.default_rng(0)
    m = rng.integers(0, NUM_CLASSES, (128, 128)).astype(np.uint8)
    out = clean_mask(m, min_size=20, median_k=3)
    assert out.shape == m.shape
    stats = area_stats(out, gsd=0.3)
    total = sum(s["pixels"] for s in stats)
    assert total == 128 * 128
    assert abs(sum(s["ratio"] for s in stats) - 1.0) < 1e-6
    # 0.3m 分辨率：1 公顷 = 10000 m2 = 111111 像元
    assert abs(stats[0]["area_ha"] - stats[0]["pixels"] * 0.09 / 1e4) < 1e-9


def test_palette_colorize():
    from rsc.palette import colorize

    m = np.array([[0, 1], [5, 255]], np.uint8)
    rgb = colorize(m)
    assert rgb.shape == (2, 2, 3)
    assert tuple(rgb[0, 0]) == (110, 110, 110)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} 通过")
    sys.exit(1 if failed else 0)

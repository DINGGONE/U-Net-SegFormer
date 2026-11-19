"""生成"样例数据"：结构与 LoveDA / Vaihingen 完全一致的合成高分影像瓦片。

为什么要有它：
  * 真实数据集动辄十几个 GB，网络/磁盘没准备好时，先用它把"裁剪→训练→评估→制图"
    整条链路跑通，代码 bug 当场暴露，不用等数据下完；
  * 单测、CI、答辩演示都需要一份秒级可复现的数据；
  * 类别分布、标签取值、影像尺寸都按 LoveDA 的规范来（1024×1024 RGB + 单通道标签 0~6），
    换成真实数据只需把路径指过去，一行都不用改。

注意：合成数据**没有**真实地物的光谱纹理，指标只能用来验证"流程是否正确"，
不能写进论文当实验结果。真实精度必须用 LoveDA / Vaihingen 的 test 或 val 跑。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# 与 palette.CLASSES 对齐：0 背景 1 建筑 2 道路 3 水体 4 植被 5 裸土
FIELDS = {
    "background": 0,
    "building": 1,
    "road": 2,
    "water": 3,
    "vegetation": 4,
    "bare_soil": 5,
}

# 近似真彩反射率（uint8），用于合成"看起来像遥感影像"的底图
BASE_RGB = {
    0: (118, 120, 112),
    1: (176, 172, 168),
    2: (150, 150, 148),
    3: (58, 86, 104),
    4: (86, 118, 74),
    5: (168, 148, 120),
}


def value_noise(h: int, w: int, scale: int, rng: np.random.Generator) -> np.ndarray:
    """多尺度值噪声：格网随机数 + 双三次插值，模拟地物内部的纹理起伏。"""
    gh, gw = max(2, h // scale), max(2, w // scale)
    grid = rng.random((gh, gw)).astype(np.float32)
    img = Image.fromarray((grid * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def fbm(h: int, w: int, octaves=(64, 32, 16, 8), rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for s in octaves:
        out += amp * value_noise(h, w, s, rng)
        tot += amp
        amp *= 0.5
    return out / tot


def make_tile(size: int, seed: int, urban: bool = False):
    """生成一张 (RGB uint8, 标签 uint8) 瓦片。"""
    rng = np.random.default_rng(seed)
    label = np.zeros((size, size), np.uint8)

    # --- 1. 大尺度景观分区：农田/林地/裸地 ---
    macro = fbm(size, size, octaves=(128, 64, 32), rng=rng)
    veg = macro > np.quantile(macro, 0.30)
    bare = (macro < np.quantile(macro, 0.15))
    label[veg] = FIELDS["vegetation"]
    label[bare] = FIELDS["bare_soil"]

    # 农田条带（模仿 LoveDA 乡村场景的规则纹理）
    if not urban:
        ang = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:size, 0:size]
        proj = xx * np.cos(ang) + yy * np.sin(ang)
        stripe = ((proj % 46) < 5)
        label[stripe & veg] = FIELDS["vegetation"]   # 保持植被，但纹理上做区分

    # --- 2. 河流：正弦曲线加宽 ---
    xs = np.arange(size)
    phase, amp, freq = rng.uniform(0, 2 * np.pi), size * rng.uniform(0.04, 0.09), rng.uniform(0.6, 1.4)
    center = size / 2 + amp * np.sin(2 * np.pi * freq * xs / size + phase)
    width = rng.uniform(14, 34)
    for x in range(size):
        y0 = int(np.clip(center[x] - width / 2, 0, size - 1))
        y1 = int(np.clip(center[x] + width / 2, 0, size - 1))
        label[y0:y1, max(0, x - 1) : x + 1] = FIELDS["water"]

    # --- 3. 道路网（横纵主干道，带轻微随机偏移）---
    for _ in range(rng.integers(2, 4)):
        y = int(rng.uniform(0.1, 0.9) * size)
        rr = int(rng.uniform(4, 9))
        label[max(0, y - rr) : y + rr, :] = FIELDS["road"]
    for _ in range(rng.integers(2, 4)):
        x = int(rng.uniform(0.1, 0.9) * size)
        rr = int(rng.uniform(4, 9))
        label[:, max(0, x - rr) : x + rr] = FIELDS["road"]

    # --- 4. 建筑：街区内的矩形/组合体，带规则朝向 ---
    n_blk = int(rng.integers(26, 46)) if urban else int(rng.integers(8, 16))
    boost = np.ones((size, size), np.float32)   # 每栋楼的亮度差异
    for _ in range(n_blk):
        bw = int(rng.integers(14, 46)) * (1 if urban else 1)
        bh = int(rng.integers(14, 42))
        y = int(rng.integers(0, size - bh - 1))
        x = int(rng.integers(0, size - bw - 1))
        # 只在非水体上盖房，并留出道路
        patch = label[y : y + bh, x : x + bw]
        ok = ~np.isin(patch, [FIELDS["water"], FIELDS["road"]])
        if ok.mean() < 0.6:
            continue
        label[y : y + bh, x : x + bw][ok] = FIELDS["building"]
        boost[y : y + bh, x : x + bw][ok] = rng.uniform(0.90, 1.16)

    # --- 5. 裸土裸露斑块（在建工地）---
    for _ in range(int(rng.integers(2, 6))):
        cy, cx = rng.integers(0, size, 2)
        r = int(rng.uniform(18, 55))
        yy, xx = np.mgrid[0:size, 0:size]
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
        label[m & (label != FIELDS["water"])] = FIELDS["bare_soil"]

    # --- 6. 由标签反推 RGB：调色板 + 纹理 + 阴影 + 噪声 ---
    rgb = np.zeros((size, size, 3), np.float32)
    for k, v in BASE_RGB.items():
        rgb[label == k] = v

    tex = fbm(size, size, octaves=(32, 16, 8, 4), rng=rng) - 0.5
    rgb += tex[..., None] * 26.0          # 地物内部亮度起伏
    rgb += rng.normal(0, 4.0, (size, size, 1))  # 传感器噪声

    # 田块级色彩差异：不同作物、不同墒情 —— 让"同类不同谱"成为难点
    field = fbm(size, size, octaves=(64, 32, 16), rng=rng)
    vegm = label == FIELDS["vegetation"]
    barm = label == FIELDS["bare_soil"]
    rgb[vegm] *= (0.84 + 0.34 * field[vegm])[..., None]
    rgb[barm] *= (0.88 + 0.26 * field[barm])[..., None]
    # 植被的季节性色彩变化（红边效应在 RGB 上表现为 G 通道主导）
    rgb[vegm, 1] += 12
    # 水体浑浊度差异 + 整体偏暗
    watm = label == FIELDS["water"]
    rgb[watm] *= (0.78 + 0.24 * field[watm])[..., None]
    # 建筑按栋高亮
    rgb *= boost[..., None]

    # 光照/大气：横向亮度梯度 + 轻微雾霾，模拟真实成像条件
    gx = np.linspace(0.90, 1.10, size, dtype=np.float32)
    gy = np.linspace(1.04, 0.94, size, dtype=np.float32)
    rgb *= (gx[None, :, None] * gy[:, None, None])
    rgb = rgb * 0.93 + 9.0

    # 建筑投影（东南向）。注意：阴影下的像元仍然属于**地表真实类别**，
    # 而不是建筑 —— 这是分割任务里最容易标注错的地方（也正因如此模型怕阴影）。
    bld = (label == FIELDS["building"]).astype(np.float32)
    dh, dw = max(3, size // 120), max(3, size // 120)
    sh_mask = np.zeros_like(bld)
    sh_mask[dh:, dw:] = bld[:-dh, :-dw]
    sh_mask[bld > 0] = 0
    sh_mask[label == FIELDS["water"]] = 0
    rgb[sh_mask > 0] *= 0.62

    # 整体模糊一点，去掉像素级的"太干净"，更像真实影像
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    rgb_u8 = np.asarray(img, dtype=np.uint8)

    return rgb_u8, label

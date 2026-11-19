"""数据集与增强（albumentations）。

统一的数据约定（所有数据集都先转成这个结构，训练代码就永远不用改）::

    <root>/<split>/images/*.png    3 通道 8bit
    <root>/<split>/masks/*.png     单通道 8bit，值 0..5，255=忽略

LoveDA 原始结构用 ``scripts/prepare_loveda.py`` 一键转过来；
ISPRS/SpaceNet/GEE 导出的 GeoTIFF 用 ``scripts/cut_geotiff.py`` 裁成这个结构。

增强策略要点（遥感分割的"有效增强"和 CV 那套不太一样）：
  * 旋转/翻转必须做满 D4（90° 倍数）—— 航拍影像没有"上"；
  * 任意角度旋转要节制，会把建筑变成斜屋顶，反而不像真实数据；
  * 颜色类增强（亮度/对比度/色调）要强，因为不同时相、不同传感器差异巨大；
  * 少用 Cutout/MixUp —— 高分影像里被挖掉的区域会变成"假地物"，标签对不上。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .palette import IGNORE_INDEX, NUM_CLASSES, remap_labels

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# 增强
# --------------------------------------------------------------------------
def _affine_kwargs(**kw) -> dict:
    """albumentations 1.3 与 1.4 的 A.Affine 参数名不同（mode/cval → border_mode/fill）。

    这里按实际签名过滤，保证两个版本都能跑 —— 否则用户装 1.3.1 就会遇到
    ``TypeError: unexpected keyword argument 'border_mode'``。
    """
    import inspect

    params = inspect.signature(A.Affine.__init__).parameters
    out = {}
    for k, v in kw.items():
        if k in params:
            out[k] = v
        elif k == "border_mode" and "mode" in params:
            out["mode"] = v          # 1.3.x 用 mode=cv2.BORDER_CONSTANT
    return out


def build_transforms(
    split: str,
    size: int = 512,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    strong: bool = True,
) -> A.Compose:
    """split='train' 返回强增强，其余返回仅 resize+normalize。"""
    if split == "train":
        ops = [
            # 先随机裁到略大尺寸再缩回，兼顾尺度不变性（多源影像 GSD 不同时很有用）
            A.RandomScale(scale_limit=(-0.25, 0.25), p=0.3),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, p=1.0),
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(**_affine_kwargs(scale=(0.9, 1.1), translate_percent=0.06,
                                      rotate=(-12, 12), border_mode=0, p=0.35)),
        ]
        if strong:
            ops += [
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
                    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=22, val_shift_limit=18),
                    A.RGBShift(r_shift_limit=18, g_shift_limit=18, b_shift_limit=18),
                ], p=0.6),
                # 阴影/光照突变：高分影像里最常见的干扰
                A.RandomShadow(p=0.15),
                A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.2),
                A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.2),
            ]
        ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    else:
        ops = [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(ops)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class SegDataset(Dataset):
    """标准目录结构的分割数据集。

    Parameters
    ----------
    root : 数据根目录，内含 ``<split>/images`` 与 ``<split>/masks``
    split : train / val / test
    size : 训练裁剪尺寸（验证时为整幅，由 ``tile_inference`` 负责切）
    preset : 标签映射预设（loveda / vaihingen / potsdam / spacenet / raw）
    augment : 是否使用增强（验证/测试请设 False）
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        size: int = 512,
        preset: str = "raw",
        augment: Optional[bool] = None,
        strong_aug: bool = True,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        return_name: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.size = size
        self.preset = preset
        self.return_name = return_name
        img_dir = self.root / split / "images"
        msk_dir = self.root / split / "masks"
        if not img_dir.exists():
            raise FileNotFoundError(f"找不到 {img_dir}；请先运行 prepare_loveda.py / cut_geotiff.py")
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        self.images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
        self.masks = [msk_dir / p.name for p in self.images]
        missing = [str(m) for m in self.masks if not m.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} 个标签缺失，例如 {missing[0]}")

        augment = (split == "train") if augment is None else augment
        self.transform = build_transforms(split if augment else "val", size, mean, std, strong_aug)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        img = np.asarray(Image.open(self.images[idx]).convert("RGB"))
        msk = np.asarray(Image.open(self.masks[idx]))
        if msk.ndim == 3:                       # 有些数据集标签是 3 通道 PNG
            msk = msk[..., 0]
        msk = remap_labels(msk, self.preset)

        out = self.transform(image=img, mask=msk)
        item = {"image": out["image"], "mask": out["mask"].long()}
        if self.return_name:
            item["name"] = self.images[idx].stem
        return item


# --------------------------------------------------------------------------
# DataLoader
# --------------------------------------------------------------------------
def build_loaders(
    root: str | Path,
    preset: str = "raw",
    size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    balance_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = SegDataset(root, "train", size, preset, augment=True)
    val_ds = SegDataset(root, "val", size, preset, augment=False)

    sampler = None
    shuffle = True
    if balance_sampler:
        # 城乡场景差异大时，可按"建筑占比"重采样，缓解乡村场景建筑过少
        w = np.array([_building_ratio(ds_i) for ds_i in train_ds.masks])
        w = np.clip(w, 0.02, None) ** -0.5
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), True)
        shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )
    return train_loader, val_loader


def _building_ratio(mask_path: Path) -> float:
    m = np.asarray(Image.open(mask_path))
    return float((m == 1).mean()) if m.size else 0.0


# --------------------------------------------------------------------------
# 类别权重（处理"植被/道路多、裸土少"的不平衡）
# --------------------------------------------------------------------------
@torch.no_grad()
def compute_class_weights(
    dataset: SegDataset,
    num_classes: int = NUM_CLASSES,
    max_images: int = 200,
    power: float = 0.5,
    cap: float = 8.0,
) -> torch.Tensor:
    """按像素频率的 -power 次方给权重，再归一化到均值 1。

    实践建议：power=0.5（平方根）比 1.0 稳，过强的权重会让小类过曝成"斑点噪声"。
    """
    counts = np.zeros(num_classes, np.float64)
    step = max(1, len(dataset) // max_images)
    for i in range(0, len(dataset), step):
        m = np.asarray(Image.open(dataset.masks[i]))
        if m.ndim == 3:
            m = m[..., 0]
        m = remap_labels(m, dataset.preset)
        v = m[(m >= 0) & (m < num_classes)]
        counts += np.bincount(v.ravel(), minlength=num_classes)
    freq = counts / max(counts.sum(), 1)
    w = np.where(freq > 0, (1.0 / np.maximum(freq, 1e-8)) ** power, 0.0)
    w = np.clip(w, 0, cap)
    w = w / w[w > 0].mean()
    return torch.tensor(w, dtype=torch.float32)

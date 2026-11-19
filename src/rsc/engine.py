"""Training and validation loops."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import SegDataset, build_loaders, compute_class_weights
from .losses import build_loss
from .metrics import ConfusionMatrix
from .models import build_model, describe
from .palette import CLASS_NAMES, NUM_CLASSES, remap_labels
from .tiling import sliding_window_predict


# --------------------------------------------------------------------------
# 学习率
# --------------------------------------------------------------------------
def poly_lambda(total_steps: int, warmup_steps: int = 0, power: float = 0.9):
    """warmup + 多项式衰减（SegFormer/U-Net 都用得上的稳妥策略）。"""

    def _fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, (1 - p) ** power)

    return _fn


def build_optimizer(model, lr: float, weight_decay: float = 1e-4,
                    backbone_lr_scale: float = 0.1):
    """主干用小学习率、解码头用大学习率（迁移学习的常规做法）。"""
    backbone, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if ("encoder" in n or "backbone" in n or n.startswith("net.")) else head).append(p)
    groups = [{"params": head, "lr": lr}]
    if backbone:
        groups.append({"params": backbone, "lr": lr * backbone_lr_scale})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


# --------------------------------------------------------------------------
# 单 epoch
# --------------------------------------------------------------------------
def train_one_epoch(
    model,
    loader: DataLoader,
    criterion,
    optimizer,
    scaler,
    schedulers,
    device: torch.device,
    grad_accum: int = 1,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total, n = 0.0, 0
    use_amp = scaler is not None and scaler.is_enabled()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for s in schedulers:
                s.step()

        total += float(loss.detach()) * grad_accum
        n += 1

    return total / max(n, 1)


# --------------------------------------------------------------------------
# 验证
# --------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model,
    root: str | Path,
    preset: str = "raw",
    num_classes: int = NUM_CLASSES,
    device: torch.device = torch.device("cpu"),
    patch: int = 512,
    stride: Optional[int] = None,
    max_images: int = 150,
    return_cm: bool = False,
):
    """在 ``<root>/val`` 上做滑窗推理并累计混淆矩阵。

    ``max_images`` 很重要：LoveDA 的 val 有 1669 幅 1024² 影像，每幅滑窗 9 次，
    6GB 卡上一个 epoch 的验证要 40 分钟 —— 训练根本跑不动。默认只取 150 幅
    （按排序均匀抽取，保证覆盖不同场景），够稳定又够快；最终评估再用全量。
    """
    from .infer import torch_predict_fn

    model.eval()
    fn = torch_predict_fn(model, device)
    ds = SegDataset(root, "val", size=patch, preset=preset, augment=False)
    idx = list(range(len(ds)))
    if max_images and len(idx) > max_images:
        idx = list(np.linspace(0, len(idx) - 1, max_images).astype(int))

    cm = ConfusionMatrix(num_classes)
    for i in idx:
        path, mpath = ds.images[i], ds.masks[i]
        img = np.asarray(Image.open(path).convert("RGB"))
        gt = np.asarray(Image.open(mpath))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt = remap_labels(gt, preset)
        pred = sliding_window_predict(img, fn, num_classes, patch=patch, stride=stride)
        cm.update(pred, gt)
    return (cm, cm.miou()) if return_cm else cm.miou()


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def fit(cfg: Dict, data_root: str | Path, out_dir: str | Path,
        resume: Optional[str] = None) -> Dict:
    """按配置训练，返回 history（含每 epoch 指标），最优权重存 ``out/best.pt``。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg.get("num_classes", NUM_CLASSES)
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    # ---- 数据 ----
    mc, dc, tc = cfg.get("model", {}), cfg.get("data", {}), cfg.get("train", {})
    size = dc.get("crop_size", 512)
    train_loader, _ = build_loaders(
        data_root, preset=dc.get("preset", "raw"), size=size,
        batch_size=dc.get("batch_size", 4), num_workers=dc.get("num_workers", 4),
        balance_sampler=dc.get("balance_sampler", False),
    )
    train_ds: SegDataset = train_loader.dataset

    # ---- 模型 ----
    model = build_model(
        arch=mc.get("arch", "unet"), encoder=mc.get("encoder", "resnet34"),
        encoder_weights=mc.get("encoder_weights", "imagenet"),
        in_channels=dc.get("in_channels", 3), num_classes=num_classes,
        pretrained=mc.get("pretrained", True),
    ).to(device)
    model.arch_name = f"{mc.get('arch', 'unet')}-{mc.get('encoder', 'resnet34')}"
    print(describe(model))

    # ---- 损失（类别权重按训练集统计）----
    weights = None
    if tc.get("class_weight", True):
        weights = compute_class_weights(train_ds, num_classes)
        print(f"类别权重: {np.round(weights.numpy(), 3)}")
    criterion = build_loss({**tc.get("loss", {}), "num_classes": num_classes}, weights, str(device))

    # ---- 优化器 / 调度 ----
    epochs = tc.get("epochs", 40)
    accum = tc.get("grad_accum", 1)
    opt = build_optimizer(model, tc.get("lr", 6e-4), tc.get("weight_decay", 1e-4),
                          tc.get("backbone_lr_scale", 0.1))
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * epochs
    warmup = int(steps_per_epoch * tc.get("warmup_epochs", 1))
    lam = poly_lambda(total_steps, warmup, tc.get("poly_power", 0.9))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and tc.get("amp", True)))
    except (AttributeError, TypeError):  # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and tc.get("amp", True)))

    start_epoch, best = 0, -1.0
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch = ck.get("epoch", 0) + 1
        best = ck.get("best", -1.0)
        print(f"从 {resume} 恢复，epoch={start_epoch}, best mIoU={best:.4f}")

    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_mf1": [], "lr": []}
    patience, bad = tc.get("early_stop_patience", 12), 0
    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        tr_loss = train_one_epoch(model, train_loader, criterion, opt, scaler, [sched],
                                  device, accum, tc.get("grad_clip", 1.0))
        cm, miou = validate(model, data_root, dc.get("preset", "raw"), num_classes, device,
                            patch=dc.get("crop_size", 512), stride=dc.get("val_stride"),
                            max_images=dc.get("val_max_images", 150), return_cm=True)
        mf1 = cm.mf1()
        history["train_loss"].append(round(tr_loss, 4))
        history["val_loss"].append(None)             # 验证 loss 可选，未计算
        history["val_miou"].append(round(miou, 4))
        history["val_mf1"].append(round(mf1, 4))
        history["lr"].append(opt.param_groups[0]["lr"])

        print(f"[{epoch + 1:03d}/{epochs}] loss={tr_loss:.4f}  mIoU={miou:.4f}  "
              f"mF1={mf1:.4f}  lr={opt.param_groups[0]['lr']:.2e}  "
              f"{time.time() - t0:.0f}s")

        if miou > best:
            best, bad = miou, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "best": best,
                        "cfg": cfg, "arch": model.arch_name},
                       out / "best.pt")
            (out / "best_summary.json").write_text(
                json.dumps(cm.summary(CLASS_NAMES), ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            bad += 1
            if bad >= patience:
                print(f"早停：{patience} 个 epoch 无提升，best mIoU={best:.4f}")
                break

    torch.save({"model": model.state_dict(), "epoch": epoch, "best": best, "cfg": cfg},
               out / "last.pt")
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + list(history))
        for i in range(len(history["train_loss"])):
            w.writerow([i] + [history[k][i] for k in history])
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"训练完成，best mIoU={best:.4f}，权重 {out / 'best.pt'}，耗时 {(time.time() - t0) / 60:.1f} min")
    return history

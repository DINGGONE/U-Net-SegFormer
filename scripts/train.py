"""训练入口。

    # 1) 先确认环境能跑（用样例数据，1 epoch 出结果）
    python scripts/train.py --config configs/loveda_unet_r34.yaml --data data/sample ^
        --preset raw --epochs 2 --out runs/smoke

    # 2) 正式训练
    python scripts/train.py --config configs/loveda_unet_r34.yaml --data data/loveda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rsc.engine import fit  # noqa: E402


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="数据根目录（含 train/val）")
    ap.add_argument("--out", default=None, help="输出目录，默认 runs/<exp_name>")
    ap.add_argument("--preset", default=None, help="覆盖标签映射预设")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--crop-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None, help="Windows 上卡住就设 0")
    ap.add_argument("--val-max-images", type=int, default=None,
                    help="每 epoch 验证多少张（默认 150；小数据集可设 0 = 全量）")
    ap.add_argument("--val-stride", type=int, default=None,
                    help="验证滑窗步长（默认 256；设 512 更快但略粗）")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不加载 ImageNet 预训练权重（离线/无法联网时用，精度会低一些）")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（只用于跑通流程）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    over = {}
    if args.preset:
        over.setdefault("data", {})["preset"] = args.preset
    if args.epochs:
        over.setdefault("train", {})["epochs"] = args.epochs
    if args.batch_size:
        over.setdefault("data", {})["batch_size"] = args.batch_size
    if args.crop_size:
        over.setdefault("data", {})["crop_size"] = args.crop_size
    if args.lr:
        over.setdefault("train", {})["lr"] = args.lr
    if args.num_workers is not None:
        over.setdefault("data", {})["num_workers"] = args.num_workers
    if args.val_max_images is not None:
        over.setdefault("data", {})["val_max_images"] = args.val_max_images
    if args.val_stride is not None:
        over.setdefault("data", {})["val_stride"] = args.val_stride
    if args.no_pretrained:
        over.setdefault("model", {})["encoder_weights"] = None
        over["model"]["pretrained"] = False
    deep_update(cfg, over)

    if args.cpu:
        import torch

        torch.cuda.is_available = lambda: False  # 仅用于冒烟测试

    out = Path(args.out or ROOT / "runs" / cfg.get("exp_name", "exp"))
    history = fit(cfg, args.data, out, resume=args.resume)

    # 训练曲线
    try:
        from rsc.cartography import history_figure

        history_figure(history, str(out / "curves.png"))
        print(f"曲线: {out / 'curves.png'}")
    except Exception as e:  # 缺字体等小问题不该中断训练
        print(f"[warn] 曲线绘制失败: {e}")

    (out / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps({"out": str(out), "best_miou": max(history["val_miou"], default=None)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

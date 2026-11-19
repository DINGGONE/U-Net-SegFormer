"""从 HuggingFace 拉 LoveDA 的一个子集（多线程、可断点续跑）。

为什么不用官方的 Zenodo 压缩包：
  * 4.0 GB 单流下载，实测 0.6 MB/s，还要 100+ 分钟，而且 Zenodo 会限流（403）；
  * HF 上镜像成了单张 PNG，可以**按需取子集**、**多线程并发**，1200 张图大约 15~25 分钟；
  * 单张失败只需重下那一张，不会前功尽弃。

数据来源：https://huggingface.co/datasets/chloechia/loveda
它的标签是**偏移过的**（0=Ignore, 1=Background, 2=Building, 3=Road, 4=Water, 5=Barren,
6=Forest, 7=Agricultural），本脚本已按 palette.LABEL_PRESETS["loveda_hf"] 转成本项目的 0~5+255。

    python scripts/fetch_loveda_hf.py --n-train 1200 --n-val 400
    python scripts/fetch_loveda_hf.py --n-train 200 --n-val 80 --out data/loveda_small
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from rsc.palette import remap_labels  # noqa: E402

REPO = "chloechia/loveda"
API = f"https://huggingface.co/api/datasets/{REPO}"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def http_get(url: str, timeout: int = 40, retries: int = 3) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # 网络抖动很常见，重试几次
            last = e
            time.sleep(1.5 * (i + 1))
    raise last  # type: ignore[misc]


def list_files() -> dict:
    data = json.loads(http_get(API))
    names = [s["rfilename"] for s in data.get("siblings", [])]
    buckets = {"train_img": [], "train_msk": [], "val_img": [], "val_msk": []}
    for n in names:
        low = n.lower()
        if low.endswith(".png"):
            if "train images" in low:
                buckets["train_img"].append(n)
            elif "train masks" in low:
                buckets["train_msk"].append(n)
            elif "val images" in low:
                buckets["val_img"].append(n)
            elif "val masks" in low:
                buckets["val_msk"].append(n)
    for k, v in buckets.items():
        buckets[k] = sorted(v, key=lambda s: int("".join(c for c in s.split("/")[-1] if c.isdigit()) or 0))
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "loveda"))
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)

    print("读取文件列表 ...")
    b = list_files()
    print(f"  可用: train {len(b['train_img'])} 对, val {len(b['val_img'])} 对")

    plan = []
    for split, imgs, msks, n in (("train", b["train_img"], b["train_msk"], args.n_train),
                                 ("val", b["val_img"], b["val_msk"], args.n_val)):
        msk_map = {m.split("/")[-1]: m for m in msks}
        picked = imgs[:n]
        for name in picked:
            base = name.split("/")[-1]
            mname = msk_map.get(base)
            if mname is None:
                continue
            plan.append((split, name, mname, base))

    todo = [(s, i, m, n) for (s, i, m, n) in plan
            if not (out / s / "masks" / n).exists()]
    print(f"  待下载 {len(todo)} 对（已存在 {len(plan) - len(todo)} 对）")

    def fetch_one(item):
        split, iname, mname, base = item
        img_bytes = http_get(BASE + urllib.parse.quote(iname))
        msk_bytes = http_get(BASE + urllib.parse.quote(mname))
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mk = Image.open(io.BytesIO(msk_bytes))
        arr = np.asarray(mk)
        if arr.ndim == 3:
            arr = arr[..., 0]
        mapped = remap_labels(arr, "loveda_hf")
        im.save(out / split / "images" / base, optimize=True)
        Image.fromarray(mapped, mode="L").save(out / split / "masks" / base, optimize=True)
        return base

    t0 = time.time()
    done, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, it): it for it in todo}
        for fut in as_completed(futs):
            done += 1
            try:
                fut.result()
            except Exception as e:
                failed.append((futs[fut][3], str(e)[:60]))
            if done % 50 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / max(el, 1)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(todo)}  已用 {el / 60:.1f} min  "
                      f"速度 {rate * 60:.0f} 对/min  剩余 {eta / 60:.1f} min")

    if failed:
        print(f"\n  失败 {len(failed)} 对，例如 {failed[:3]}")
        print("  重跑本脚本会自动补下缺失的部分。")

    # ---- 统计 ----
    print("\n  类别像素占比：")
    for split in ("train", "val"):
        msks = sorted((out / split / "masks").glob("*.png"))
        if not msks:
            continue
        cnt = np.zeros(7, np.int64)
        for p in msks[:: max(1, len(msks) // 300)]:   # 抽样统计
            m = np.asarray(Image.open(p))
            cnt += np.bincount(m.ravel(), minlength=7)[:7]
        tot = cnt.sum()
        print(f"   {split} ({len(msks)} 张): " +
              " ".join(f"{i}:{c / tot * 100:.1f}%" for i, c in enumerate(cnt)))

    (out / "README.txt").write_text(
        f"来源: https://huggingface.co/datasets/{REPO}\n"
        f"子集: train={args.n_train}, val={args.n_val}\n"
        "标签已映射到本项目 0~5（0 其他 1 建筑 2 道路 3 水体 4 植被 5 裸土），255=忽略\n"
        "训练时用 --preset raw\n", encoding="utf-8")
    print(f"\n完成，输出到 {out}；训练时加 --preset raw")


if __name__ == "__main__":
    main()

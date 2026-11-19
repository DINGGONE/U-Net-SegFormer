"""下载 PyTorch 的 CUDA 版 wheel（支持断点续传），解决"国内下载 2.4 GB 老断"的问题。

背景：
  * `pip install torch` 默认从 PyPI 装的是 **CPU 版**（约 200 MB），能跑但不吃显卡；
  * 要吃显卡必须装 **CUDA 版**（2.4 GB），官方源 download.pytorch.org 在国内经常几十 KB/s；
  * 阿里云镜像有完整 wheel，实测约 0.3 MB/s（2.4 GB 约 2 小时），但断了就得重来。

所以这个脚本做了三件 pip 不做的事：
  1. **断点续传**：断了重跑，从上次的字节位置继续（HTTP Range）；
  2. **失败重试**：网络抖动自动重连，最多 50 次；
  3. **进度可视**：实时显示 MB/s、已下载百分比、剩余时间。

用法::

    # 1) 下载（可以随时 Ctrl+C，下次重跑会接着下）
    python scripts/get_torch_cuda.py

    # 2) 下完自动安装
    python scripts/get_torch_cuda.py --install

    # 3) 换个镜像 / 换个版本
    python scripts/get_torch_cuda.py --mirror https://mirror.sjtu.edu.cn/pytorch-wheels --cu 121
"""
from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

MIRRORS = [
    "https://mirrors.aliyun.com/pytorch-wheels",
    "https://mirror.sjtu.edu.cn/pytorch-wheels",
]
PKGS = ("torch", "torchvision")
VERSIONS = {"torch": "2.4.1", "torchvision": "0.19.1"}


def wheel_name(pkg: str, ver: str, cu: str, py: str) -> str:
    return f"{pkg}-{ver}%2Bcu{cu}-{py}-{py}-win_amd64.whl"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, out: Path, retries: int = 50) -> bool:
    """带断点续传的下载。返回 True 表示文件完整。"""
    ctx = ssl.create_default_context()
    for attempt in range(1, retries + 1):
        have = out.stat().st_size if out.exists() else 0
        headers = {"User-Agent": "pip/24.2"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                total = None
                if r.status == 206:
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr:
                        total = int(cr.split("/")[-1])
                elif r.status == 200:
                    have = 0            # 服务器不支持续传，从头来
                    out.unlink(missing_ok=True)
                    total = int(r.headers.get("Content-Length") or 0)

                mode = "ab" if have else "wb"
                t0, n0 = time.time(), have
                with open(out, mode) as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - t0 > 3:
                            spd = (have - n0) / (time.time() - t0)
                            pct = f"{have / total * 100:5.1f}%" if total else "  ?  "
                            eta = f"{(total - have) / spd / 60:5.1f} min" if total and spd > 0 else "  ?  "
                            print(f"\r    {pct}  {human(have)} / {human(total or 0)}  "
                                  f"{spd / 1e6:5.2f} MB/s  剩余 {eta}", end="")
                            t0, n0 = time.time(), have
            if total and have >= total:
                print(f"\r    100.0%  {human(have)}  完成" + " " * 30)
                return True
            print()
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"\n    [重试 {attempt}/{retries}] {type(e).__name__}: {str(e)[:70]}")
            time.sleep(min(2 * attempt, 15))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "Downloads" / "torch_wheels"))
    ap.add_argument("--cu", default="121", help="CUDA 版本，如 121 / 118")
    ap.add_argument("--mirror", default=MIRRORS[0])
    ap.add_argument("--install", action="store_true", help="下载完成后自动 pip install")
    args = ap.parse_args()

    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  下载 PyTorch CUDA 版 wheel（支持断点续传）")
    print(f"  Python 标签 : {py}   （不匹配就装不上，本机 {sys.version.split()[0]}）")
    print(f"  CUDA        : {args.cu}")
    print(f"  镜像        : {args.mirror}")
    print(f"  保存到      : {out}")
    print("=" * 64)

    files = []
    for pkg in PKGS:
        name = wheel_name(pkg, VERSIONS[pkg], args.cu, py)
        url = f"{args.mirror}/cu{args.cu}/{name}"
        path = out / name.replace("%2B", "+")
        print(f"\n[{pkg}] {url}")
        if path.exists() and path.stat().st_size > 0:
            print(f"    已存在 {human(path.stat().st_size)}，尝试续传...")
        if not download(url, path):
            print(f"    失败：{url}")
            print("    建议：换个镜像重试  --mirror https://mirror.sjtu.edu.cn/pytorch-wheels")
            return 1
        files.append(path)

    print("\n全部下载完成：")
    for f in files:
        print(f"    {f}  ({human(f.stat().st_size)})")

    cmd = [sys.executable, "-m", "pip", "install", *[str(f) for f in files]]
    print("\n安装命令（复制即可）：\n    " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if args.install:
        print("\n正在安装...")
        return subprocess.call(cmd)
    print("\n加上 --install 可以直接安装。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

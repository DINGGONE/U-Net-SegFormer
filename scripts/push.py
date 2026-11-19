"""Push the repository to GitHub (HTTPS with retries, SSH fallback).

Usage:
    python scripts/push.py                 # add + commit + push over HTTPS, then try SSH
    python scripts/push.py --https         # HTTPS only
    python scripts/push.py --ssh           # SSH only
    python scripts/push.py --force         # force push (remote has a partial web upload)
    python scripts/push.py --remote URL    # set the remote first
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

GIT_CANDIDATES = (
    "git",
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
)
SSH_DIR = Path.home() / ".ssh"
KEY = SSH_DIR / "id_ed25519"


def sh(args, cwd=None, timeout=120) -> tuple[int, str]:
    try:
        p = subprocess.run(args, cwd=str(cwd or ROOT), text=True, encoding="utf-8",
                           errors="replace", stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, (p.stdout or "").strip()
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def git(*args, timeout=180) -> tuple[int, str]:
    for exe in GIT_CANDIDATES:
        if shutil.which(exe) or Path(exe).exists():
            rc, out = sh([exe, *args], timeout=timeout)
            if rc != 127:
                return rc, out
    return 127, "git not found"


def git_dir() -> Path | None:
    for exe in GIT_CANDIDATES:
        if Path(exe).exists():
            return Path(exe).parent.parent
        found = shutil.which(exe)
        if found:
            return Path(found).parent.parent
    return None


def to_ssh_url(url: str) -> str:
    m = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    return f"git@{'github.com'}:{m.group(1)}/{m.group(2)}.git" if m else url


def net_error(out: str) -> bool:
    low = out.lower()
    return any(k in low for k in ("connection was reset", "could not connect",
                                  "failed to connect", "timed out", "recv failure"))


def push(force: bool, retries: int, label: str) -> tuple[bool, str]:
    args = ["push", "-u", "origin", "HEAD:main"]
    if force:
        args.insert(1, "--force")
    out = ""
    for i in range(1, retries + 1):
        print(f"\n  {label} 推送第 {i}/{retries} 次 ...")
        rc, out = git(*args, timeout=600)
        if rc == 0:
            return True, out
        for line in out.splitlines()[-2:]:
            print("    " + line)
        if not net_error(out) or i == retries:
            break
        wait = 4 * i
        print(f"    连接被重置，{wait} 秒后重试 ...")
        time.sleep(wait)
    return False, out


def ssh_setup_and_push(url: str, force: bool) -> bool:
    gd = git_dir()
    if gd is None:
        print("  [X] 找不到 Git 安装目录，无法使用 SSH")
        return False
    ssh = gd / "usr" / "bin" / "ssh.exe"
    keygen = gd / "usr" / "bin" / "ssh-keygen.exe"

    print("\n" + "-" * 58)
    print("  改用 SSH 方式（HTTPS 被重置时通常能通）")
    print("-" * 58)

    if not KEY.exists():
        print("  1) 生成本机 SSH 密钥 ...")
        SSH_DIR.mkdir(parents=True, exist_ok=True)
        rc, out = sh([str(keygen), "-t", "ed25519", "-C", "github", "-f", str(KEY), "-N", ""])
        if rc not in (0, 1):
            print("  [X] 生成密钥失败：", out[:200])
            return False
        print("     已生成", KEY)
    else:
        print("  1) 已有 SSH 密钥", KEY)

    pub = Path(str(KEY) + ".pub").read_text(encoding="utf-8").strip()
    try:
        subprocess.run("clip", input=pub, text=True, check=False)
        print("  2) 公钥已复制到剪贴板")
    except Exception:
        print("  2) 请手动复制下面的公钥：\n")
    if not shutil.which("clip"):
        print("     " + pub)

    print("\n  3) 接下来需要把公钥加到 GitHub（只需一次）：")
    print("     · 浏览器已打开页面，点 New SSH key")
    print("     · Title 随便填（如 my-pc），Key 里 Ctrl+V 粘贴")
    print("     · 点 Add SSH key")
    try:
        os.startfile("https://github.com/settings/keys")
    except Exception:
        print("     打开：https://github.com/settings/keys")
    try:
        input("\n  粘贴完成后，按回车继续 ...")
    except EOFError:
        pass

    ssh_url = to_ssh_url(url)
    git("remote", "set-url", "origin", ssh_url)
    print(f"  4) 远端已切换为 {ssh_url}")

    print("  5) 测试 SSH 连接 ...")
    rc, out = sh([str(ssh), "-T", "-o", "StrictHostKeyChecking=accept-new",
                  "-o", "BatchMode=yes", "git@github.com"], timeout=30)
    ok = "successfully authenticated" in (out or "").lower()
    print("     " + (out.splitlines()[0] if out else "(无输出)"))
    if not ok:
        print("  [!] SSH 认证未通过，可能是公钥还没加到 GitHub，或粘贴错了。")
        return False

    return push(force, 1, "SSH")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--https", action="store_true", help="只走 HTTPS")
    ap.add_argument("--ssh", action="store_true", help="只走 SSH")
    args = ap.parse_args()

    print("=" * 58)
    print("  上传到 GitHub")
    print("=" * 58)

    rc, ver = git("--version")
    if rc != 0:
        print("  [X] 没有找到 Git。先装：https://git-scm.com/download/win")
        return 1
    print(f"  {ver}")

    if args.remote:
        git("remote", "remove", "origin")
        git("remote", "add", "origin", args.remote)
        print(f"  远端设为 {args.remote}")

    rc, url = git("remote", "get-url", "origin")
    if rc != 0:
        print("  [X] 还没有远端。先指定：")
        print('      python scripts/push.py --remote https://github.com/用户名/仓库.git')
        return 1
    url = url.strip()
    print(f"  远端：{url}")

    git("add", "-A")
    rc, out = git("commit", "-m", "update: 同步本地改动")
    if rc == 0:
        print("  已提交本地改动")
    elif "nothing to commit" in out or "无文件要提交" in out:
        print("  没有新改动")
    else:
        print("  [!] 提交失败：" + out[:300])
        print('      先设置：git config user.name "名字" / user.email "邮箱"')
        return 1

    ok = False
    if not args.ssh:
        ok, out = push(args.force, args.retries, "HTTPS")
    if ok:
        print("\n" + "=" * 58)
        print("  [OK] 上传成功")
        print("  " + url.replace(".git", ""))
        print("=" * 58)
        return 0

    if args.https:
        print("\n  HTTPS 推送失败。网络原因，可改用：python scripts/push.py --ssh")
        return 1

    print("\n  HTTPS 走不通（连接被重置），自动切换到 SSH ...")
    if ssh_setup_and_push(url, args.force):
        print("\n" + "=" * 58)
        print("  [OK] 上传成功（SSH）")
        print("  " + url.replace(".git", ""))
        print("=" * 58)
        return 0

    print("\n" + "-" * 58)
    print("  两条路都没通。最后两个办法：")
    print("  1) 推到 Gitee，再用 GitHub 的 Import repository 导入：")
    print("     python scripts/push.py --remote https://gitee.com/你的名/rs-landcover.git --force")
    print("  2) 继续用网页拖拽上传（没有提交历史）")
    return 1


if __name__ == "__main__":
    sys.exit(main())

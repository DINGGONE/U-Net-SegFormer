"""Push the repository to GitHub, with retries and clear diagnostics.

Usage:
    python scripts/push.py                  # add + commit + push
    python scripts/push.py --force          # force push (remote only has a partial upload)
    python scripts/push.py --remote URL     # set/override the remote first
"""
from __future__ import annotations

import argparse
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


def git(*args: str, check: bool = False) -> tuple[int, str]:
    for exe in GIT_CANDIDATES:
        try:
            p = subprocess.run([exe, *args], cwd=str(ROOT), text=True, encoding="utf-8",
                               errors="replace", stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
            return p.returncode, p.stdout or ""
        except FileNotFoundError:
            continue
    return 127, "git not found"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    print("=" * 58)
    print("  上传到 GitHub")
    print("=" * 58)

    rc, ver = git("--version")
    if rc != 0:
        print("  [X] 没有找到 Git。先装：https://git-scm.com/download/win")
        return 1
    print(f"  {ver.strip()}")

    if args.remote:
        git("remote", "remove", "origin")
        git("remote", "add", "origin", args.remote)
        print(f"  远端已设为 {args.remote}")

    rc, url = git("remote", "get-url", "origin")
    if rc != 0:
        print("  [X] 还没有关联远端仓库。先运行：")
        print('      python scripts/push.py --remote https://github.com/用户名/仓库名.git')
        return 1
    print(f"  远端：{url.strip()}")

    # ---- 暂存与提交 ----
    git("add", "-A")
    rc, out = git("commit", "-m", "update: 同步本地改动")
    if rc == 0:
        print("  已提交本地改动")
    elif "nothing to commit" in out or "无文件要提交" in out:
        print("  没有新改动，直接推送")
    else:
        print("  [!] 提交失败：")
        print("  " + out.strip()[:400])
        print('      先设置身份：git config user.name "名字" / user.email "邮箱"')
        return 1

    # ---- 推送（带重试）----
    push_args = ["push", "-u", "origin", "HEAD:main"]
    if args.force:
        push_args.insert(1, "--force")

    for i in range(1, args.retries + 1):
        print(f"\n  推送中（第 {i}/{args.retries} 次）...")
        rc, out = git(*push_args)
        if rc == 0:
            print("\n" + "=" * 58)
            print("  [OK] 上传成功")
            print(f"  {url.strip().replace('.git', '')}")
            print("=" * 58)
            return 0
        tail = out.strip().splitlines()[-3:]
        for t in tail:
            print("    " + t)

        low = out.lower()
        net = ("connection was reset" in low or "could not connect" in low
               or "timed out" in low or "recv failure" in low or "failed to connect" in low)
        if net and i < args.retries:
            wait = 5 * i
            print(f"    网络问题，{wait} 秒后重试...")
            time.sleep(wait)
        elif not net:
            break

    # ---- 失败诊断 ----
    print("\n" + "-" * 58)
    print("  推送失败。按报错对照：")
    low = out.lower()
    if "connection was reset" in low or "could not connect" in low or "timed out" in low:
        print("  → GitHub 连不上（国内常见），不是你配置错。三条路：")
        print("    1) 有代理：git config --global http.proxy http://127.0.0.1:7890 后重跑本脚本")
        print("    2) 推到 Gitee，再用 GitHub 的 Import repository 导入：")
        print("       git remote add gitee https://gitee.com/你的名/rs-landcover.git")
        print("       git push -u gitee main")
        print("    3) 继续用网页拖拽上传（缺点：没有提交历史）")
    elif "rejected" in low or "non-fast-forward" in low:
        print("  → 远端已有网页提交，和本地历史不一致。用强制推送覆盖：")
        print("       python scripts/push.py --force")
    elif "could not read username" in low or "terminal prompts disabled" in low or "prompt script" in low:
        print("  → 需要登录授权，但当前窗口弹不出登录框。")
        print("     · 请直接双击根目录的 一键上传.bat（在你自己电脑的窗口里跑，才能弹出浏览器登录）")
        print("     · 或者先配好令牌：git config --global credential.helper manager")
    elif "authentication" in low or "403" in low:
        print("  → 认证失败：密码处要填 Personal Access Token（Settings > Developer settings > Tokens，勾 repo）")
    else:
        print("  → 把上面的原始报错整段复制下来排查")
    return 1


if __name__ == "__main__":
    sys.exit(main())

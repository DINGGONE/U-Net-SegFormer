"""第一次上传 GitHub 的向导（中文交互，逐个确认，不会用命令行也能跑）。

用法：
    双击项目根目录的 `上传到GitHub.bat`
    或   python scripts/upload_github.py

它会依次做：
    1. 找 git（没装会告诉你去哪下、下哪个）
    2. 配置提交者姓名/邮箱（只做一次）
    3. 检查有没有误把大数据集/权重加进来（>50MB 直接拦下）
    4. git init / add / commit
    5. 问你要 GitHub 仓库地址，关联并推送
    6. 推送完打印仓库链接

安全说明：不会上传 data/loveda、runs/、wheels/、.torch、data/raw 这些目录（.gitignore 已挡）。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

MAX_FILE_MB = 50
GIT_DOWNLOAD = "https://git-scm.com/download/win"
PORTABLE_HINT = r".tools\PortableGit\cmd\git.exe"


# --------------------------------------------------------------------------
def line(ch="─", n=64):
    print(ch * n)


def title(t):
    print()
    line("═")
    print(f"  {t}")
    line("═")


def step(n, total, t):
    print(f"\n[{n}/{total}] {t}")


def run(args, check=True, capture=True):
    """跑一条 git 命令，返回 (returncode, 输出文本)。"""
    try:
        p = subprocess.run(
            args, cwd=str(ROOT), text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )
        out = (p.stdout or "") if capture else ""
        if check and p.returncode != 0:
            print(f"  ! 命令失败: {' '.join(args)}")
            if out.strip():
                print("  " + out.strip()[:800].replace("\n", "\n  "))
        return p.returncode, out
    except FileNotFoundError:
        return 127, "找不到可执行文件"


def find_git() -> str | None:
    for cand in ("git", r"C:\Program Files\Git\cmd\git.exe",
                 r"C:\Program Files (x86)\Git\cmd\git.exe",
                 str(ROOT / PORTABLE_HINT),
                 str(Path.home() / "AppData/Local/Programs/Git/cmd/git.exe")):
        try:
            p = subprocess.run([cand, "--version"], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            if p.returncode == 0:
                return cand
        except (FileNotFoundError, OSError):
            continue
    return None


def ask(prompt: str, default: str = "") -> str:
    s = input(f"  {prompt}" + (f" [{default}]" if default else "") + "：").strip()
    return s or default


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=None, help="直接指定仓库地址（跳过交互）")
    ap.add_argument("--no-push", action="store_true", help="只提交到本地，不推送")
    args = ap.parse_args()

    title("把 rs-landcover 上传到 GitHub")
    print("  这是一个交互向导，每一步都会问你，直接回车用默认值即可。")
    print(f"  项目目录：{ROOT}")

    # ---------- 1. git ----------
    step(1, 6, "检查 Git")
    git = find_git()
    if not git:
        print("  ✗ 没找到 Git —— 这是必须的，先装它（2 分钟）：")
        print(f"      1) 打开 {GIT_DOWNLOAD}")
        print("      2) 下载 64-bit Git for Windows Setup，一路 Next 安装")
        print("      3) 装完【关掉所有命令行窗口】，重新双击本文件")
        print()
        print("  或者更简单：用 GitHub Desktop（图形界面，不用命令行）")
        print("      见 docs/05_第一次上传GitHub.md 的「路径 A」")
        return 1
    _, ver = run([git, "--version"])
    print(f"  ✓ {ver.strip()}   ({git})")

    # ---------- 2. 初始化仓库（必须先于 config，否则 git config 会报 not in a git directory）----------
    step(2, 6, "初始化仓库")
    if (ROOT / ".git").exists():
        print("  ✓ 已经是 git 仓库，跳过 git init")
    else:
        rc, _ = run([git, "init", "-b", "main"])
        if rc != 0:                      # 老版本 git 不支持 -b
            run([git, "init"])
            run([git, "checkout", "-b", "main"], check=False)
        print("  ✓ 已初始化，分支 main")

    # ---------- 3. 提交者身份 ----------
    step(3, 6, "配置提交者身份（写进每次提交记录里）")
    _, name = run([git, "config", "user.name"], check=False)
    _, email = run([git, "config", "user.email"], check=False)
    if not name.strip():
        name = ask("你的名字（建议用 GitHub 用户名）")
        rc, err = run([git, "config", "user.name", name], check=False)
        print(f"  ✓ 已设置为 {name}" if rc == 0 else f"  ! 设置失败：{err.strip()[:120]}")
    else:
        print(f"  ✓ 姓名：{name.strip()}")
    if not email.strip():
        email = ask("你的邮箱（GitHub 注册邮箱）")
        rc, err = run([git, "config", "user.email", email], check=False)
        print(f"  ✓ 已设置为 {email}" if rc == 0 else f"  ! 设置失败：{err.strip()[:120]}")
    else:
        print(f"  ✓ 邮箱：{email.strip()}")

    # 中文文件名不转义，日志里能直接看懂
    run([git, "config", "core.quotepath", "false"], check=False)

    # ---------- 3. 初始化 ----------
    # ---------- 4. 安全检查 + 暂存 ----------
    step(4, 6, "检查要上传的文件（挡住大数据集和权重）")
    run([git, "add", "-A"])
    _, staged = run([git, "diff", "--cached", "--name-only"], check=False)
    files = [f for f in staged.splitlines() if f.strip()]
    if not files:
        print("  没有新文件需要提交（可能之前已经提交过）")
    else:
        print(f"  暂存 {len(files)} 个文件")

    big = []
    for f in files:
        p = ROOT / f
        if p.is_file() and p.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            big.append((f, p.stat().st_size / 1024 / 1024))
    if big:
        print(f"\n  ✗ 有 {len(big)} 个文件超过 {MAX_FILE_MB} MB，GitHub 会拒绝：")
        for f, mb in big[:8]:
            print(f"      {mb:7.1f} MB  {f}")
        print("\n  处理办法：确认它们在 .gitignore 里，然后执行")
        print("      git rm -r --cached .")
        print("      git add .")
        return 1
    total_mb = sum((ROOT / f).stat().st_size for f in files if (ROOT / f).is_file()) / 1024 / 1024
    print(f"  总体积 {total_mb:.1f} MB，没有超限文件 ✓")

    # ---------- 5. 提交 ----------
    step(5, 6, "创建提交")
    rc, out = run([git, "commit", "-m",
                   "feat: 高分遥感6类地物分类流水线（U-Net/SegFormer + 评估 + 制图 + GEE）"],
                  check=False)
    if rc == 0:
        print("  ✓ 提交完成")
    elif "nothing to commit" in out or "无文件要提交" in out:
        print("  ✓ 没有新改动，跳过")
    else:
        print("  ✗ 提交失败，原始输出：")
        print("  " + (out.strip()[:800] or "(无输出)"))
        print("\n  最常见原因：没配置提交者身份。执行下面两行后重来：")
        print('      git config user.name "你的名字"')
        print('      git config user.email "你的邮箱"')
        return 1

    # ---------- 6. 关联远程并推送 ----------
    if args.no_push:
        step(6, 6, "跳过推送（--no-push）")
        print("  本地已完成，之后可以手动 git remote add / git push")
        return 0

    step(6, 6, "关联 GitHub 仓库并推送")
    print("  先在浏览器打开 https://github.com/new 建一个空仓库：")
    print("    · Repository name 填  rs-landcover")
    print("    · 不要勾 Add a README / .gitignore / license（本地已有）")
    print("    · 点 Create repository，复制页面上的 https://github.com/你/rs-landcover.git")

    url = args.remote or ask("粘贴仓库地址（https://github.com/.../rs-landcover.git）")
    if not re.match(r"^(https://|git@)", url or ""):
        print("  ✗ 地址格式不对，应该以 https://github.com/ 开头")
        return 1

    rc, _ = run([git, "remote", "get-url", "origin"], check=False)
    if rc == 0:
        run([git, "remote", "set-url", "origin", url], check=False)
    else:
        run([git, "remote", "add", "origin", url], check=False)

    print("\n  开始推送。如果弹出登录窗口，选 “Sign in with your browser” 授权即可。")
    print("  （若要求输入密码，要填 Personal Access Token，不是 GitHub 登录密码）\n")
    rc, out = run([git, "push", "-u", "origin", "main"], check=False)

    if rc == 0:
        line("═")
        print("  ✓ 上传成功！")
        web = url.replace(".git", "")
        print(f"  你的仓库地址：{web}")
        print("  接下来去仓库页面做三件事：")
        print("    1) 右上 About 填描述，Topics 加 remote-sensing / semantic-segmentation")
        print("    2) 看 Actions 有没有变绿（自动跑单测）")
        print("    3) Releases → v0.1.0，把 runs/loveda_r34/best.pt 作为附件传上去")
        line("═")
        return 0

    print("  ✗ 推送失败。原始输出：")
    print("  " + (out.strip()[:900] or "(无输出)"))

    low = (out or "").lower()
    print("\n  按报错判断原因：")
    if "connection was reset" in low or "recv failure" in low or "timed out" in low:
        print("    → 【网络被重置】GitHub 在国内经常连不上，这不是你配置错了。三种办法：")
        print("      ① 有代理（Clash/V2Ray）就设一下再推：")
        print("         git config --global http.proxy http://127.0.0.1:7890")
        print("         git push -u origin main")
        print("      ② 没代理：先推到 Gitee（国内秒传），再用 GitHub 的 Import repository 导入")
        print("         见 docs/05_第一次上传GitHub.md 的「网络不通怎么办」")
        print("      ③ 换个网络（手机热点）重试一次")
    elif "authentication" in low or "403" in low or "could not read username" in low:
        print("    → 【认证失败】密码位置要填 Personal Access Token：")
        print("      https://github.com/settings/tokens → Generate new token (classic) → 勾 repo")
    elif "rejected" in low or "non-fast-forward" in low:
        print("    → 【远端已有提交】先合并再推：")
        print("      git pull --rebase origin main")
        print("      git push -u origin main")
    else:
        print("    → 把上面原始输出整段发给懂行的人；或改用 GitHub Desktop（见 docs/05）")
    return 1


if __name__ == "__main__":
    sys.exit(main())

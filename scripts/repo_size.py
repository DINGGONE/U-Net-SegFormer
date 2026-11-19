"""按 .gitignore 规则统计"即将提交"的文件数量与体积（发布前自检用）。

    python scripts/repo_size.py
"""
from __future__ import annotations

import fnmatch
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_rules() -> tuple[list[str], list[str]]:
    pats, negs = [], []
    gi = ROOT / ".gitignore"
    if not gi.exists():
        return pats, negs
    for line in gi.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        (negs if line.startswith("!") else pats).append(line.lstrip("!"))
    return pats, negs


def ignored(rel: str, pats: list[str], negs: list[str]) -> bool:
    hit = False
    name = rel.rsplit("/", 1)[-1]
    for p in pats:
        base = p.rstrip("/")
        if (fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, base)
                or rel.startswith(base + "/") or fnmatch.fnmatch(name, p)):
            hit = True
    for n in negs:
        if fnmatch.fnmatch(rel, n) or rel.startswith(n.rstrip("/") + "/"):
            hit = False
    return hit


def main() -> None:
    pats, negs = load_rules()
    files, total = [], 0
    for p in ROOT.rglob("*"):
        if p.is_dir() or ".git" in p.parts:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if ignored(rel, pats, negs):
            continue
        sz = p.stat().st_size
        files.append((rel, sz))
        total += sz

    files.sort(key=lambda x: -x[1])
    print(f"将提交 {len(files)} 个文件，共 {total / 1024 / 1024:.2f} MB\n")
    print("最大的 10 个：")
    for rel, sz in files[:10]:
        print(f"  {sz / 1024:8.0f} KB  {rel}")

    d = defaultdict(lambda: [0, 0])
    for rel, sz in files:
        d[rel.split("/")[0]][0] += 1
        d[rel.split("/")[0]][1] += sz
    print("\n按目录：")
    for k, (n, s) in sorted(d.items(), key=lambda x: -x[1][1]):
        print(f"  {k:<22}{n:4d} 个  {s / 1024 / 1024:6.2f} MB")

    big = [f for f in files if f[1] > 50 * 1024 * 1024]
    print(f"\n> 50 MB 的文件：{len(big)} 个（GitHub 单文件上限 100 MB）")
    if big:
        for rel, sz in big:
            print(f"  !! {sz / 1024 / 1024:7.1f} MB  {rel}")
        sys.exit(1)


if __name__ == "__main__":
    main()

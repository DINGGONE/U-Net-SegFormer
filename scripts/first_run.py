"""新手向导：双击 `一键运行.bat` 后实际执行的就是这个脚本。

它按顺序做 6 件事，每一步都会用中文告诉你在干什么、成功长什么样：

    1. 报告环境（Python 版本、有没有 GPU、有没有 torch）
    2. 检查并自动安装最小依赖（numpy / pillow / matplotlib / scipy）
    3. 跑 10 个自检测试
    4. 生成样例数据（没有才生成）
    5. 跑传统基线，出 3 张图 + 1 份指标
    6. 打印"接下来做什么"，并自动打开成果图

也可以手动运行::

    python scripts/first_run.py           # 完整流程
    python scripts/first_run.py --check   # 只检查环境，不跑流程
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    # line_buffering：保证本脚本的中文提示与子进程输出按真实先后顺序显示
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PY = sys.executable
MIN_DEPS = ["numpy", "PIL", "matplotlib", "scipy"]
PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"


def line(ch: str = "=", n: int = 62) -> None:
    print(ch * n)


def title(text: str) -> None:
    print()
    line()
    print(f"  {text}")
    line()


def step(n: int, total: int, text: str) -> None:
    print(f"\n[{n}/{total}] {text}")


def run(cmd, **kw) -> int:
    """跑子进程并返回退出码；输出直接透传到屏幕，方便新手看到报错。"""
    return subprocess.call(cmd, cwd=str(ROOT), **kw)


def check_env() -> dict:
    info = {"python": sys.version.split()[0], "exe": PY, "torch": None, "gpu": None,
            "missing": []}
    for mod in MIN_DEPS:
        try:
            __import__(mod)
        except Exception:
            info["missing"].append(mod)
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    return info


def print_env(info: dict) -> None:
    print(f"  Python    : {info['python']}")
    print(f"  解释器     : {info['exe']}")
    ok = "已安装" if not info["missing"] else "缺少 " + ", ".join(info["missing"])
    print(f"  基础依赖   : {ok}")
    if info["torch"]:
        print(f"  PyTorch   : {info['torch']}   显卡: {info['gpu'] or '未检测到（只能 CPU）'}")
    else:
        print("  PyTorch   : 未安装 —— 暂时不影响下面 1~4 步，")
        print("              只影响'训练模型'（第 5 步会告诉你怎么装）")


def ensure_deps(info: dict) -> bool:
    if not info["missing"]:
        print("  依赖齐全，跳过安装。")
        return True
    pkgs = {"PIL": "pillow"}
    names = [pkgs.get(m, m) for m in info["missing"]]
    print(f"  缺少 {', '.join(names)}，正在自动安装（约 1~3 分钟，走清华镜像）...")
    rc = run([PY, "-m", "pip", "install", *names, "-i", PIP_INDEX])
    if rc != 0:
        print("\n  [失败] 自动安装没成功。请手动执行下面这行，再重新双击本文件：")
        print(f'      "{PY}" -m pip install {" ".join(names)} -i {PIP_INDEX}')
        return False
    print("  安装完成。")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查环境")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="不自动打开图片")
    args = ap.parse_args()

    title("高分遥感地物分类 · 新手向导")
    print("  目标：20 秒内让你看到第一张成果图，并确认整条流水线是通的。")
    print("  注意：这一步不需要 GPU，也不需要下载任何数据集。")

    # ---------- 1. 环境 ----------
    step(1, 5, "检查环境")
    info = check_env()
    print_env(info)
    if args.check:
        print("\n  仅检查模式，结束。")
        return 0

    # ---------- 2. 依赖 ----------
    step(2, 5, "检查依赖")
    if not ensure_deps(info):
        return 1
    info = check_env()          # 装完再确认一次

    # ---------- 3. 自检 ----------
    step(3, 5, "运行自检测试（10 项，应全部 PASS）")
    if args.skip_test:
        print("  已跳过。")
    else:
        rc = run([PY, str(ROOT / "tests" / "test_smoke.py")])
        if rc != 0:
            print("\n  [失败] 自检没通过。请把上面的红色报错整段复制下来。")
            return 1

    # ---------- 4. 样例数据 ----------
    step(4, 5, "准备样例数据")
    scene = ROOT / "data" / "sample" / "scene" / "scene_rgb.png"
    if scene.exists():
        print(f"  样例数据已存在，跳过生成：{scene.parent}")
    else:
        print("  正在生成 11 张 1024×1024 瓦片 + 1 幅 2048×3072 场景，约 20 秒...")
        rc = run([PY, str(ROOT / "scripts" / "make_sample_data.py")])
        if rc != 0:
            print("  [失败] 样例数据生成失败。")
            return 1

    # ---------- 5. 基线 + 出图 ----------
    step(5, 5, "跑传统基线并出图（分类 → 评估 → 后处理 → 制图）")
    rc = run([PY, str(ROOT / "scripts" / "run_baseline.py")])
    if rc != 0:
        print("  [失败] 基线流程出错。")
        return 1

    # ---------- 总结 ----------
    out = ROOT / "outputs"
    title("全部完成")
    print("  刚才发生了什么（按顺序）：")
    print("    ① 用最小距离分类器'训练'了一遍（传统方法基线）")
    print("    ② 在 3 张验证瓦片上算了 mIoU / F1 / OA")
    print("    ③ 用滑窗对 2048×3072 的整幅场景做了预测")
    print("    ④ 后处理去掉碎斑，统计面积，画成成果图")
    print()
    print("  你现在应该能看到 3 张图：")
    print(f"    {out / 'demo_landcover_map.png'}")
    print("        → 区域土地利用智能解译图（这是最终要交的成果）")
    print(f"    {out / 'demo_comparison.png'}")
    print("        → 左：原影像   中：人工标注   右：模型预测（对比看误差）")
    print(f"    {out / 'demo_confusion.png'}")
    print("        → 混淆矩阵（哪两类最容易混淆，一眼看出）")
    print()
    print("  下一步（按顺序，别跳）：")
    print("    第 1 步  ✓ 你已经完成了（跑通流水线）")
    print("    第 2 步  看 docs/01_数据与标注.md，下载 LoveDA 数据集")
    print("    第 3 步  装 PyTorch（见 docs/00_新手一步步.md 第 4 节）")
    print("    第 4 步  先用样例数据训 2 个 epoch，确认能跑")
    print("    第 5 步  换 LoveDA 正式训练")
    print()

    if not args.no_open:
        for name in ("demo_landcover_map.png", "demo_comparison.png"):
            p = out / name
            if p.exists():
                try:
                    os.startfile(str(p))       # Windows 用默认看图程序打开
                except Exception:
                    pass
        print("  图片已自动打开。关掉图片窗口后，回到命令行窗口按任意键结束。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

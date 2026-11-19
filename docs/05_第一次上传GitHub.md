# 第一次上传到 GitHub（零基础版）

> 你已经有完整的项目了，不需要再写任何代码。**这一步只是把文件夹变成 GitHub 仓库。**
> 两条路，选一条走完就行：**路径 A 用图形界面（推荐第一次）**，路径 B 用一条命令。

---

## 先搞清楚：要传的是什么

项目目录：

```
C:\Users\Administrator\Documents\deepseek-harness\default-workspace\rs-landcover
```

里面已经有这些**成品**（不用你新建任何文件）：

| 文件 / 目录 | 是什么 |
|---|---|
| `README.md` | 项目主页，含成果图、精度表、快速开始 |
| `src/rsc/` | 核心库：调色板、指标、切片、模型、损失、训练、推理、制图 |
| `scripts/` | 15 个命令行工具（训练/评估/推理/裁剪/下载…） |
| `configs/` | 3 套实验配置 |
| `docs/` | 4 篇文档 + `images/` 成果图 |
| `tests/` | 10 个单测 |
| `gee/` | GEE 脚本 |
| `LICENSE`、`CITATION.cff`、`.gitignore`、`.github/workflows/` | 开源仓库标配 |
| `data/sample/` | 合成样例数据（18.8 MB，让别人 clone 下来就能跑） |

**不会上传**（`.gitignore` 已挡住）：`data/loveda`（1.3 GB）、`runs/`（权重 560 MB）、`wheels/`（2.3 GB）、`.torch/`、`data/raw/`（你的原始影像）、所有 `*.tif`。

最终会传 **85 个文件、约 24.5 MB**。

---

# 路径 A：GitHub Desktop（推荐，全程点鼠标）

适合：完全不想碰命令行的人。缺点：装一个 ~120 MB 的客户端。

### A1. 装客户端

打开 <https://desktop.github.com/download/> → 下载 → 安装 → 打开。

### A2. 登录

启动后点 **Sign in to GitHub.com** → 浏览器会弹出授权页 → 点 **Authorize desktop** → 回到客户端。

（没有账号就先在 <https://github.com/signup> 注册，邮箱验证一下。）

### A3. 把项目加进来

菜单 **File → Add local repository…**

在弹出的框里点 **Choose…**，选中：

```
C:\Users\Administrator\Documents\deepseek-harness\default-workspace\rs-landcover
```

点 **Add repository**。

> 如果提示 *"This directory does not appear to be a Git repository. Would you like to create a repository here instead?"*
> —— 点 **create a repository**，Name 填 `rs-landcover`，点 **Create repository**。

### A4. 检查待提交列表（关键一步）

左侧会列出几百个文件。**往下翻，确认没有这些**：

```
data/loveda/...       ← 如果有，说明 .gitignore 没生效，别提交
runs/...b est.pt
wheels/
data/raw/
```

正常情况你只会看到 `src/`、`scripts/`、`docs/`、`data/sample/`、`README.md` 等。

### A5. 提交

左下角 **Summary** 填：

```
feat: 高分遥感6类地物分类流水线（U-Net/SegFormer + 评估 + 制图 + GEE）
```

点 **Commit to main**（如果弹出要你填作者名/邮箱，填 GitHub 用户名和注册邮箱）。

### A6. 发布到 GitHub

点顶部 **Publish repository**：

- Name：`rs-landcover`
- Description：`高分遥感影像6类地物语义分割：U-Net/SegFormer 训练、mIoU 评估、滑窗推理与土地利用解译图`
- ⬜ **取消勾选 "Keep this code private"**（想要公开作品集就必须取消）
- 点 **Publish repository**

等进度条走完 → 点 **View on GitHub**，浏览器会打开你的仓库。**完事了。**

---

# 路径 B：命令行（装 Git + 双击一个 bat）

适合：想顺便学会 git 命令的人。

### B1. 装 Git

打开 <https://git-scm.com/download/win> → 下载 **64-bit Git for Windows Setup** → 安装时**一路 Next**（默认选项都对）。

装完**关掉所有已打开的命令行窗口**（不关的话新装的 git 不生效）。

验证：新开一个 cmd，输入

```bat
git --version
```

出现 `git version 2.x` 就成功。

### B2. 网页上建一个空仓库

1. 打开 <https://github.com/new>
2. **Repository name**：`rs-landcover`
3. 选 **Public**
4. ⚠️ **不要勾** `Add a README file`、`Add .gitignore`、`Choose a license`（本地都有了，勾了会冲突）
5. 点 **Create repository**
6. 复制页面上的地址，形如：`https://github.com/你的用户名/rs-landcover.git`

### B3. 双击 `上传到GitHub.bat`

位置：

```
rs-landcover\上传到GitHub.bat
```

它会用中文一步步问你：

| 提示 | 你填什么 |
|---|---|
| 你的名字 | GitHub 用户名 |
| 你的邮箱 | GitHub 注册邮箱 |
| 粘贴仓库地址 | B2 复制的那串 `https://github.com/...` |

然后它会自动：初始化 → 检查大文件 → 提交 → 推送。

### B4. 认证

推送时会弹窗或提示登录：

- **弹窗**：选 **Sign in with your browser** → 浏览器点 Authorize → 完成
- **要求输入密码**：⚠️ 填 **Personal Access Token**，不是登录密码
  1. <https://github.com/settings/tokens> → **Generate new token (classic)**
  2. 勾选 **`repo`** 权限 → 生成
  3. 复制 `ghp_xxxx`，粘到密码框（输入时不显示，粘完回车）

看到 `✓ 上传成功` 就好了。

---

# 两条路都走完之后：5 件事

| # | 在哪 | 做什么 |
|---|---|---|
| 1 | 仓库右上 ⚙️ **About** | 填 Description；**Topics** 加 `remote-sensing` `semantic-segmentation` `land-cover` `pytorch` `u-net` `gis` |
| 2 | **Actions** 标签页 | 看 `smoke-tests` 是否变绿（自动跑 10 个单测，约 1 分钟） |
| 3 | README 页面 | 确认 3 张图都能显示（如果显示裂图，说明 `docs/images/` 没传上去） |
| 4 | **Releases** → Draft a new release | Tag `v0.1.0`，把 `runs/loveda_r34/best.pt` 拖进去当附件（权重不进 git） |
| 5 | 仓库 **Settings → General** | 确认 License 显示为 MIT |

---

# 以后改了代码怎么更新？

**路径 A（GitHub Desktop）**：打开客户端 → 它会自动显示改动的文件 → 填 Summary → **Commit to main** → **Push origin**。

**路径 B（命令行）**：再双击一次 `上传到GitHub.bat` 就行（它会自动 add/commit/push，没有新改动会提示跳过）。

或者手动：

```bat
git add .
git commit -m "fix: 修正 xxx"
git push
```

---

# 报错对照表

| 现象 | 原因 | 解决 |
|---|---|---|
| `git 不是内部或外部命令` | 装完没重开窗口 | 关掉 cmd，重新双击 bat |
| 客户端提示 "not a Git repository" | 还没初始化 | 点 **create a repository** |
| `Authentication failed` | 填了登录密码 | 改用 Personal Access Token |
| `remote origin already exists` | 重复关联 | 脚本已自动处理；手动则 `git remote set-url origin <地址>` |
| `Updates were rejected` | 远端有 README 等初始文件 | `git pull --rebase origin main` 后重推 |
| `file is 105.00 MB; this exceeds GitHub's file size limit` | 大文件混进来了 | 见下方"大文件" |
| 中文文件名变成 `\344\275\240...` | git 默认转义 | 已自动设置 `core.quotepath false` |
| 推送卡在 0% 很久 | 网络 | 换网络重试；或先 `git config --global http.proxy` 设代理 |

---

# 网络不通怎么办（GitHub 连不上时）

症状：`git push` 报

```
Failed to connect to github.com:443 after 21085 ms: Could not connect to server
Recv failure: Connection was reset
RPC failed; curl 56
```

**这不是你配置错了**，是国内直连 GitHub 的常见情况。三条路，按推荐顺序：

## 方案 1：网页拖拽上传（最快，5 分钟）

只要**浏览器能打开 GitHub** 就能用。

1. 浏览器打开你的仓库 `https://github.com/你的用户名/仓库名`
2. 点 **Add file → Upload files**
3. 打开资源管理器，进入 `rs-landcover` 文件夹，**全选里面的文件和文件夹**（Ctrl+A），拖进网页虚线框
   - 注意：不要拖 `data\loveda`、`runs`、`wheels`、`.torch`、`data\raw`（这些不该上传）
   - 网页单次最多 100 个文件、单文件 25 MB；本项目 88 个文件 / 24.5 MB，正好能一次传完
4. 下方 Commit changes 里填说明，点 **Commit changes**

> 缺点：没有提交历史（只有一个 commit）。适合"先传上去再说"。

## 方案 2：Gitee 中转 + GitHub 导入（最稳，保留完整历史）

1. 注册 <https://gitee.com>，新建仓库 `rs-landcover`（不要初始化）
2. 本地加一个 gitee 远端并推送：

```bat
git remote add gitee https://gitee.com/你的用户名/rs-landcover.git
git push -u gitee main
```

3. 回到 GitHub → **New repository** → 页面底部点 **Import a repository**
4. 填 Gitee 仓库地址 `https://gitee.com/你的用户名/rs-landcover.git` → **Begin import**

GitHub 的服务器会自己去 Gitee 拉代码（不经过你的网络），1~2 分钟完成，提交历史完整保留。

## 方案 3：有代理就用代理

如果你装了 Clash / V2Ray（常见本地端口 7890）：

```bat
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
git push -u origin main
```

推送成功后可移除：`git config --global --unset http.proxy`

## 顺带一条：改仓库名

如果建仓库时名字被自动生成成了 `-U-Net-SegFormer-` 这种带横杠的怪名字，
去 **Settings → Repository name** 改成 `rs-landcover`，然后本地更新远端地址：

```bat
git remote set-url origin https://github.com/你的用户名/rs-landcover.git
```

---

# 关于大文件

**永远不要** `git add` 数据集和模型权重。三种正确做法：

| 内容 | 做法 |
|---|---|
| `best.pt`（95 MB） | Release 附件上传，README 给下载链接 |
| LoveDA（1.3 GB） | 不传，README 写下载地址（Zenodo / HF） |
| 你的原始影像 | 不传（可能涉密），自己留本地 |

GitHub 限制：单文件 ≤ 100 MB，仓库建议 ≤ 1 GB。当前 24.5 MB 完全没压力。

---

# 一句话总结

- **不想折腾** → 装 GitHub Desktop → Add local repository → 取消 private → Publish
- **想学命令** → 装 Git → 双击 `上传到GitHub.bat` → 粘仓库地址 → 浏览器授权

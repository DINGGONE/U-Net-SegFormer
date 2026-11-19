# 03 · 制图与 GEE 闭环

模型输出一张 `uint8` 类别图，**不等于**交付物。甲方/评审要的是"能读的图 + 能查的数 + 能核的面积"。
这一篇讲怎么把 `pred` 变成成果，以及怎么用 GEE 把项目从"跑个模型"升级成"行业应用"。

---

## 1. 一张合格的解译图必须有的 7 个要素

| # | 要素 | 为什么 | 实现 |
|---|---|---|---|
| 1 | 主图（分类底色） | 成果主体 | `colorize(pred)` |
| 2 | 图例（名称 + 配色 + 面积占比） | 没有图例的图等于没画 | `legend_entries` + 面积表 |
| 3 | 比例尺（分段，取整到 1/2/5） | 判读距离与地块大小 | `draw_scalebar`（按 `gsd` 自动算） |
| 4 | 指北针 | 制图规范硬要求 | `draw_north_arrow` |
| 5 | 标题 + 副标题（数据/方法/时间） | 让别人知道这是什么 | `title` / `subtitle` |
| 6 | 面积统计（ha / 占比 / 条形图） | "行业应用"的落点 | `postprocess.area_stats` |
| 7 | 精度信息（mIoU/OA + 模型名） | 敢不敢用的依据 | `metrics` 参数 |

外加两个加分项：**原影像缩略图**（便于对照判读）与**数据来源/坐标系/分辨率脚注**。

### 用法

```python
from rsc.cartography import landcover_poster
from rsc.metrics import ConfusionMatrix
import json

summary = json.load(open("outputs/metrics_baseline.json"))["metrics_scene"]
landcover_poster(
    rgb, pred, gsd=0.09,                       # 9 cm 数据
    title="XX 区土地利用智能解译图",
    subtitle="2024 年 6 月 · 高分二号融合影像 · 深度学习语义分割",
    metrics=summary,
    model_name="SegFormer-B2 (mIoU 0.61)",
    out_png="outputs/area15_poster.png",
    out_pdf="outputs/area15_poster.pdf",       # PDF 矢量文字，打印/答辩用
    show_image_inset=True,
)
```

命令行等价：

```bat
python scripts\predict.py --ckpt runs\loveda_unet_r34\best.pt --src data\raw\area15.tif ^
    --out-dir outputs\area15 --gsd 0.09 --poster --model-name "U-Net-ResNet34"
```

---

## 2. 配色：不要用默认彩虹色

`src/rsc/palette.py` 里定的这 6 个颜色是有讲究的（也符合国内土地利用图习惯）：

| 类别 | 色值 | 理由 |
|---|---|---|
| 其他 | `#6E6E6E` 灰 | 中性、退后 |
| 建筑 | `#D63E3E` 红 | 建设用地惯用红/品红，视觉最突出 |
| 道路 | `#F6C443` 黄 | 交通线性要素惯用黄 |
| 水体 | `#2E6CD0` 蓝 | 水域惯用蓝 |
| 植被 | `#3A9E54` 绿 | 植被惯用绿 |
| 裸土 | `#BA926A` 土黄 | 自然土色，与建筑红有明度区分 |

原则：**同类同色系、异类异色相**；避开纯红绿搭配（红绿色盲不可辨）；打印成黑白后仍能靠明度区分。

---

## 3. 后处理与"制图综合"

制图规范里有一个概念叫**最小上图单元**：小于某个面积（如 1:10000 下的 4 mm²）的图斑不允许单独出现。
对应到像素就是：

```
最小图斑(px) = 最小上图面积(m²) / gsd²
例：1:10000 的 4 mm² → 实地 400 m²；gsd 0.09 m → 400 / 0.0081 ≈ 49,400 px ≈ 222×222
```

所以 `--min-size` 要按成图比例尺算，不是拍脑袋。常用值：

| 成图比例尺 | 最小上图面积 | gsd 0.09 m | gsd 0.3 m | gsd 10 m |
|---|---|---|---|---|
| 1:2000 | 1 mm² = 4 m² | 494 px | 44 px | <1 px |
| 1:10000 | 4 mm² = 400 m² | 49,400 px | 4,444 px | 4 px |
| 1:50000 | 4 mm² = 10,000 m² | 1.2 M px | 111,111 px | 100 px |

小比例尺（1:50000）时中值滤波 + 大 min_size 就够了；大比例尺（1:2000）要慎用形态学，
否则**独栋小房子会被吃掉**——这正是建筑提取业务最不能接受的。

---

## 4. 矢量化：把栅格变成 GIS 能用的东西

```python
from rsc.postprocess import vectorize
import rasterio, json

with rasterio.open("outputs/area15/area15_landcover.tif") as ds:
    mask = ds.read(1)
    fc = vectorize(mask, transform=ds.transform, crs=ds.crs)
json.dump(fc, open("outputs/area15/area15_landcover.geojson", "w", encoding="utf-8"),
          ensure_ascii=False)
```

得到 GeoJSON 后可以在 QGIS 里：
- 按 `class_name` 符号化，套用同一套配色；
- `dissolve` 成连片图斑，算真实面积（比按像元数算更准，能消掉阶梯边界）；
- 与二调/三调数据叠加做变化检测。

> 矢量化时若有 `min_area` 需求，用 `shapely` 按 `geometry.area` 过滤，比栅格形态学更可控。

---

## 5. GEE 闭环：三种玩法（按投入从低到高）

### 玩法 A：GEE 只做底图与验证（1 小时，性价比最高）

1. `gee/export_s2_patches.js` 导出研究区 Sentinel-2 时相合成（10 m，5 波段）；
2. 本地 `predict.py` 推理 → 解译图；
3. 在 GEE 里把解译结果 `ee.Image.loadGeoTIFF` 叠到卫星底图上做**目视校核**（`Map.addLayer` 加透明度）。

**话术**："用 GEE 做多时相合成，避免了单景云污染；结果回叠到 GEE 底图做快速质检。"
这一步几乎零成本，但让答辩 PPT 立刻有"行业平台"的味道。

### 玩法 B：GEE 做样本与验证点（半天）

1. `gee/sample_points.js` 分层随机抽 60 点/类；
2. 导出 CSV → 野外/目视核查 → `obs_class` 回填；
3. `python scripts\evaluate.py --points xxx.csv` 出**样点法精度 + 95% 置信区间**。

**话术**："精度评价采用 Olofsson 推荐的分层随机抽样，OA = 0.xx ± 0.0x（95% CI），
每类样点数 ≥ 60。"——这句话比"mIoU 0.72"专业得多。

### 玩法 C：GEE 在线推理（1~2 天，加分项）

把训练好的模型转成 TF SavedModel，走 Vertex AI 端点，在 GEE 里 `ee.Model.fromVertexAI` 直接对
`ee.Image` 做 `predict`，输出整景分类图（无需下载影像）。

```
PyTorch ──torch.onnx.export──> ONNX ──tf2onnx/onnx2tf──> SavedModel ──> GCS ──> Vertex AI Endpoint
                                                                                │
                                                     ee.Model.fromVertexAI(project, region, endpoint)
                                                                                │
                                                       image.predict(model).clip(aoi) → 在线解译图
```

坑：GEE 的 `predict` 对**输入尺寸和波段顺序**很敏感，务必在模型里固定输入 512×512、
输出 logits 后 `argmax` 再 `byte()`；大区域要按 tile 分批 `Export.image.toDrive`。

---

## 6. 交付物清单（按这个交，不会缺东西）

| # | 交付物 | 文件 | 验收点 |
|---|---|---|---|
| 1 | 分类模型权重 | `runs/<exp>/best.pt` + `best_summary.json` | 可复现（含 config_used.yaml） |
| 2 | 精度报告 | `metrics.json` + `confusion_matrix.png` | mIoU/F1/OA/逐类 + 样点法 |
| 3 | 分类成果栅格 | `*_landcover.tif`（uint8, 255=nodata） | 保留 CRS/transform，QGIS 可开 |
| 4 | 分类成果矢量 | `*_landcover.geojson` | 属性含 class_id/class_name |
| 5 | **区域土地利用智能解译图** | `*_poster.png/pdf`（A1 或 A0，300 dpi） | 7 要素齐全 |
| 6 | 面积统计表 | `area_stats` 导出的 CSV | 与图例一致 |
| 7 | 对比图集 | `comparison.png`（含坏例） | 好例/坏例都有 |
| 8 | 训练曲线 | `curves.png` | loss 收敛、mIoU 上升 |
| 9 | 演示视频 | 3 min（数据→训练→推理→成图） | 能离线播放 |
| 10 | 技术报告 | 8~15 页 | 数据、方法、实验、误差分析、结论 |

---

## 7. "算法 + 行业应用"的叙事结构

答辩/面试时，按这四层讲，比罗列模型结构有效得多：

1. **痛点**（30 秒）：土地利用变更调查靠人工判读，1 个人 1 天只能处理 2~3 km²，且标准不统一。
2. **方法**（1 分钟）：语义分割自动提取 6 类地物，滑窗支持任意大区域，后处理保证制图规范。
3. **证据**（1 分钟）：mIoU 0.xx / 建筑 F1 0.xx；样点法 OA 0.xx ± 0.0x；单景 2048×3072 推理 5 s。
4. **闭环**（1 分钟）：GEE 多时相合成 → 模型解译 → 面积统计 → 成果图 → 变化检测（下一期）。
   **"不是做了个模型，是做了条产线。"**

可量化的"行业价值"表述（按你自己的数字填）：

> 单幅 10 km² 影像，人工判读约需 3 人日；本流程从推理到出图约 15 分钟（含后处理与制图），
> 效率提升约 2 个数量级；面积统计与矢量成果可直接对接国土变更调查的数据格式。

---

## 8. 下一期可以加的东西（写进"展望"）

- **变化检测**：双时相输入（Siamese U-Net），输出"新增建筑/减少植被"，业务价值比单时相分类更高；
- **多源融合**：DSM/nDSM 作为第 4 通道（Vaihingen 上建筑 IoU 通常 +3~5）；
- **半监督/自训练**：用模型给无标签区域打伪标签，缓解"标注贵"；
- **不确定性**：MC Dropout 或 Deep Ensemble 输出置信度图，低置信区交人工复核——这才是能落地的形态。

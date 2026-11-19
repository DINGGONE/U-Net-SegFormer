// ============================================================================
// GEE：分层随机验证样点生成（精度评价 + 野外核查用）
// ----------------------------------------------------------------------------
// 用途：从"参考真值图层"（土地利用矢量 / 已有分类图 / 人工解译结果）里，
//       按类别分层抽取样点，导出带经纬度的 CSV，作为 mIoU/F1 与野外核查的依据。
//
// 为什么必须分层随机：
//   * 纯随机抽样会让"植被/农田"占满样本，建筑/裸土只有几十个点，精度估计的
//     方差大到没法用（Olofsson et al. 2014 的标准做法就是分层随机）；
//   * 分层后每类至少 50~100 个点，才能保证 ±5% 的置信区间。
//
// 用法：改下面 3 个参数 → Run → Tasks 里下载 CSV → 交给 scripts/evaluate.py --points
// ============================================================================

// ---------- 1. 参数 ----------
var REF_ASSET = 'users/yourname/landuse_ref';   // 参考图层：面状，必须有 class_id 字段
var CLASS_FIELD = 'class_id';
var N_PER_CLASS = 60;                            // 每类样点数（面积加权时可改成按比例）
var MIN_DIST_M = 100;                            // 样点最小间距，避免空间自相关
var OUT_NAME = 'validation_points';

// 6 类体系（与 src/rsc/palette.py 保持一致）
var CLASSES = [
  {id: 0, name: 'background'},
  {id: 1, name: 'building'},
  {id: 2, name: 'road'},
  {id: 3, name: 'water'},
  {id: 4, name: 'vegetation'},
  {id: 5, name: 'bare_soil'}
];

var ref = ee.FeatureCollection(REF_ASSET);

// ---------- 2. 逐类分层抽样 ----------
var perClass = CLASSES.map(function (c) {
  var sub = ref.filter(ee.Filter.eq(CLASS_FIELD, c.id));
  // stratifiedSample 在面内生成随机点；numPoints 按类固定
  var pts = sub.map(function (f) {
    return ee.Feature(f.geometry());
  }).stratifiedSample({
    numPoints: N_PER_CLASS,
    classBand: null,
    region: sub.geometry(),
    scale: 10,
    seed: 42,
    geometries: true,
    tileScale: 4
  });
  return pts.map(function (p) {
    return p.set({
      class_id: c.id,
      class_name: c.name,
      ref_class: c.id,        // 参考类别（用于算精度）
      obs_class: -1,          // 野外/目视核查填这一列（-1 = 未核查）
      point_id: ee.String(c.name).cat('_').cat(p.id())
    });
  });
});

// 合并 + 去掉彼此过近的点（贪心去重，GEE 里简单实现）
var all = ee.FeatureCollection(perClass).flatten();
var pts = all.randomColumn('rnd', 42).sort('rnd');

// 简易最小间距过滤：把点做成 100m 缓冲区，去掉与之重叠的后续点
var kept = ee.FeatureCollection([]);
// 注意：GEE 客户端循环效率低，样点 < 1000 时才这么干
var n = Math.min(pts.size().getInfo(), 1000);
var list = pts.toList(n);
var usedGeom = [];
for (var i = 0; i < n; i++) {
  var f = ee.Feature(list.get(i));
  var g = f.geometry();
  var tooClose = false;
  for (var j = 0; j < usedGeom.length; j++) {
    if (g.distance(usedGeom[j], 1).lt(MIN_DIST_M).getInfo()) { tooClose = true; break; }
  }
  if (!tooClose) { usedGeom.push(g); kept = kept.merge(ee.FeatureCollection([f])); }
}

// ---------- 3. 加经纬度并导出 ----------
var out = kept.map(function (f) {
  var c = f.geometry().centroid(1).coordinates();
  return f.set({
    lon: c.get(0),
    lat: c.get(1),
    row: null, col: null,       // 与本地 sample_points.csv 的列对齐
    checked: 0,
    note: ''
  });
});

print('样点总数', out.size());
print('各类样点数', out.aggregate_histogram('class_name'));

Export.table.toDrive({
  collection: out,
  description: OUT_NAME,
  folder: 'rs_landcover_points',
  fileFormat: 'CSV',
  selectors: ['point_id', 'class_id', 'class_name', 'ref_class', 'obs_class',
              'lon', 'lat', 'checked', 'note']
});

// ---------- 4. 精度评价（拿到 obs_class 之后在本地算）----------
// 混淆矩阵 + 用户精度/生产者精度/OA 的 95% 置信区间，本地脚本示例：
//   python scripts/evaluate.py --ckpt runs/x/best.pt --data data/loveda \
//       --points outputs/validation_points.csv
// 面积加权的总体精度（Olofsson 推荐）需要各类面积权重 W_i：
//   OA = Σ W_i * (correct_i / n_i)，CI ≈ 1.96 * sqrt(Σ W_i² * p_i(1-p_i)/(n_i-1))

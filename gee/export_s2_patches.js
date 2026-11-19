// ============================================================================
// GEE：导出 Sentinel-2 10m 训练瓦片（用于"自建数据集"路线 + 区域解译图底图）
// ----------------------------------------------------------------------------
// 在 https://code.earthengine.google.com 新建脚本，粘贴运行。
// 流程：去云合成 → 选 5 个波段 → 指定研究区 → 按 512×512 网格切块导出到 Drive。
//
// 为什么要用 GEE：
//   1) 免费、免下载、全球覆盖，Landsat/Sentinel 全在云端；
//   2) 可以做"时序合成"，把多时相压成一张无云影像（比单景质量高得多）；
//   3) 导出 GeoTIFF 后直接喂给 scripts/cut_geotiff.py 或 src/rsc/infer.py。
// ============================================================================

// ---------- 1. 参数 ----------
var AOI_ASSET = null;                       // 可选：'users/xxx/study_area'（面状 Asset）
var AOI_RECT = [116.30, 39.85, 116.45, 39.95];  // 没 Asset 时用矩形 [xmin,ymin,xmax,ymax]
var START = '2024-04-01';
var END = '2024-10-31';
var CLOUD_PCT = 15;
var TILE_M = 5120;                          // 每块 5.12 km，10m 分辨率 → 512×512 像素
var OUT_FOLDER = 'rs_landcover_s2';

var aoi = AOI_ASSET
  ? ee.FeatureCollection(AOI_ASSET).geometry()
  : ee.Geometry.Rectangle(AOI_RECT);

// ---------- 2. 去云函数（Sentinel-2 L2A 的 QA60 + SCL 双保险）----------
function maskS2(img) {
  var scl = img.select('SCL');
  // 3=云影 8=云中概率 9=高概率云 10=薄云 11=雪
  var bad = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10)).or(scl.eq(11));
  var qa = img.select('QA60');
  var cloud = qa.bitwiseAnd(1 << 10).neq(0).or(qa.bitwiseAnd(1 << 11).neq(0));
  return img.updateMask(bad.or(cloud).not()).copyProperties(img, ['system:time_start']);
}

// ---------- 3. 时相合成：中值合成，去掉残余云 ----------
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate(START, END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_PCT))
  .map(maskS2);

print('影像数量', s2.size());

// B2蓝 B3绿 B4红 B8近红外 B11短波红外（10m/20m；B11 重采样到 10m）
var composite = s2.median()
  .select(['B2', 'B3', 'B4', 'B8', 'B11'])
  .resample('bilinear')
  .clip(aoi);

// 常用指数，便于做"规则先验"或特征工程
var ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = composite.normalizedDifference(['B3', 'B8']).rename('NDWI');
var ndbi = composite.normalizedDifference(['B11', 'B8']).rename('NDBI');

Map.centerObject(aoi, 11);
Map.addLayer(composite, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'S2 真彩');
Map.addLayer(ndvi, {min: -0.2, max: 0.8, palette: ['brown', 'yellow', 'green']}, 'NDVI');

// ---------- 4. 按网格切块导出 ----------
// 用 coveringGrid 生成 512×512 像元的块（projection 必须是 10m）
var proj = ee.Projection('EPSG:32650').atScale(10);   // ← 改成你所在 UTM 带
var grid = composite.geometry().coveringGrid(proj, TILE_M);

var tiles = ee.FeatureCollection(grid).map(function (f) {
  return f.set('tile_id', f.get('system:index'));
});

// 批量导出（每块一个 GeoTIFF；超过 500 块请分批，Drive 会限流）
var n = tiles.size().getInfo();
print('将导出块数', n);

var list = tiles.toList(Math.min(n, 300));
for (var i = 0; i < Math.min(n, 300); i++) {
  var f = ee.Feature(list.get(i));
  var geom = f.geometry();
  var id = f.get('tile_id').getInfo();
  Export.image.toDrive({
    image: composite.toFloat(),          // 用 toFloat 保留反射率精度，别用 uint8
    description: 's2_' + id,
    folder: OUT_FOLDER,
    fileNamePrefix: 's2_' + id,
    region: geom,
    scale: 10,
    crs: proj.crs(),
    maxPixels: 1e13
  });
}

// ---------- 5. 训练样本点（可选）：分层随机采样 ----------
// 有现成土地利用矢量时，用 sampleRegions 直接生成带标签的样本点，
// 再在本地用 csv 训练随机森林/最小距离，作为深度学习之外的对照。
// var samples = composite.addBands(ndvi).sampleRegions({
//   collection: ee.FeatureCollection('users/xxx/landuse_points'),
//   properties: ['class_id'], scale: 10, geometries: true
// });
// Export.table.toDrive({collection: samples, description: 's2_samples', fileFormat: 'CSV'});

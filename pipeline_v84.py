"""
===============================================================================
КОНВЕЙЕР v8.4 — БАЙЕСОВСКИЙ KDE-КЛАССИФИКАТОР ПОТЕНЦИАЛА ВРИ
Cloud Edition (geopandas + rasterio, без QGIS)

Изменения v8.4 относительно v8.3:
  • SVG 02_alpha: размер 910×1032 → 600×681, отступы пересчитаны.
  • SVG 04_cv разделён на два файла: 04a_cv_accuracy (точность Top-1/3/5)
    и 04b_cv_ranks (распределение рангов истинного класса), каждый 600×681.
  • SVG 03_kde_*: повышена читаемость — добавлена линия общего тренда
    (KDE по всем данным независимо от ВРИ, толстый чёрный пунктир);
    индивидуальные кривые тоньше и полупрозрачнее; легенда вынесена под
    график; добавлена светлая сетка. Размер не изменён.

Изменения v8.3 относительно v8.2:
  • Индексы P и E перед построением матрицы «Потенциал × Эффект»
    нормализуются по принципу min-max в пределах одного участка:
    лучший ВРИ → 1.0, худший → 0.0. Это устраняет «слипание» всех
    классов в нижнем-левом углу из-за нормировки softmax по 13 классам
    (при равномерном распределении среднее значение ≈ 1/13 ≈ 0.077).
  • В results.json и boundary_results['ranking'] дополнительно
    сохраняются сырые softmax-значения: P_raw, E_raw, S_raw.
  • Поле 'S' теперь означает нормализованный индекс пригодности [0, 1],
    что делает пороги 0.4 / 0.6 в SVG достижимыми и осмысленными.

Запуск:  python pipeline_v84.py /path/to/data/
  Папка должна содержать: egrn.geojson, poi_point.geojson, poi_polygon.geojson,
  roads.geojson, zouit.geojson, dem.tif, boundary_1.geojson, boundary_2.geojson,
  boundary_3.geojson  (необязательные слои пропускаются)

Результаты: /path/to/data/results_YYYY-MM-DD_HH-MM-SS/
===============================================================================
"""
import os, sys, json, time, math, warnings, re, glob
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    from shapely.geometry import Point, MultiPolygon, Polygon
    from shapely.strtree import STRtree
    from shapely.ops import unary_union
    import shapely
except ImportError:
    sys.exit("pip install geopandas shapely")

try:
    from scipy.stats import gaussian_kde
    from scipy.spatial import cKDTree
except ImportError:
    sys.exit("pip install scipy")

try:
    import rasterio
    from rasterio.warp import transform as rio_transform
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠ rasterio не найден — slope будет = 0")

try:
    from pyproj import CRS
except ImportError:
    sys.exit("pip install pyproj")

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False; print("⚠ matplotlib не найден")

warnings.filterwarnings('ignore')
print("=" * 70)
print("  КОНВЕЙЕР v8.4 — БАЙЕСОВСКИЙ KDE-КЛАССИФИКАТОР ВРИ (Cloud)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
#   КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════
COL_VRI_GROUP       = 'vri_group'
COL_LAND_CATEGORY   = 'opt_land_record_category_type'
COL_ZOUIT_TYPE      = 'opt_type_zone'
COL_HIGHWAY         = 'highway'

POI_RADIUS_M        = 2500
ROAD_LEVELS_FILTER  = {'motorway','trunk','primary','secondary','tertiary'}
SPATIAL_BUFFER_M    = 800
MIN_CLASS_SIZE      = 20
N_VRI_CLASSES       = 13

RUN_STAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

VRI_NAMES = {
    1:'Сельскохозяйственное использование', 2:'Жилая застройка',
    3:'Общественное использование', 4:'Предпринимательство',
    5:'Отдых (рекреация)', 6:'Производственная деятельность',
    7:'Транспорт', 8:'Обеспечение обороны и безопасности',
    9:'Охрана и изучение природы', 10:'Использование лесов',
    11:'Водные объекты', 12:'Территории общего пользования',
    13:'Земельные участки общего назначения',
}
VRI_COLORS_HEX = {
    1:'#8B6914', 2:'#D4A017', 3:'#2E5CB8', 4:'#C0392B', 5:'#27AE60',
    6:'#7F8C8D', 7:'#2C3E50', 8:'#922B21', 9:'#1E8449', 10:'#196F3D',
    11:'#2980B9', 12:'#8E44AD', 13:'#B7950B',
}

# ═══════════════════════════════════════════════════════════════════════
#   ЗАГРУЗКА
# ═══════════════════════════════════════════════════════════════════════
if len(sys.argv) < 2:
    DATA_DIR = '.'
else:
    DATA_DIR = sys.argv[1]
DATA_DIR = os.path.abspath(DATA_DIR)
print(f"\n▶ Загрузка из {DATA_DIR}...")
t0 = time.time()

def load_gdf(name, required=True):
    """Ищет name.geojson, name.gpkg, name.shp в DATA_DIR."""
    for ext in ['.geojson', '.gpkg', '.shp', '.json']:
        p = os.path.join(DATA_DIR, name + ext)
        if os.path.isfile(p):
            gdf = gpd.read_file(p)
            print(f"  ✓ {name}: {len(gdf)} фич ({ext})")
            return gdf
    if required:
        raise FileNotFoundError(f"Файл '{name}.*' не найден в {DATA_DIR}")
    print(f"  – {name}: не найден")
    return None

gdf_egrn = load_gdf('egrn')
gdf_poi_pt = load_gdf('poi_point', False)
gdf_poi_pg = load_gdf('poi_polygon', False)
gdf_roads = load_gdf('roads', False)
gdf_zouit = load_gdf('zouit', False)

# Boundary — отдельные файлы
boundary_gdfs = []
for bn in ['boundary_1', 'boundary_2', 'boundary_3', 'boundary']:
    g = load_gdf(bn, False)
    if g is not None and len(g) > 0:
        boundary_gdfs.append((g, bn))
print(f"  Boundary: {len(boundary_gdfs)} слоёв")

# DEM
dem_path = None
for ext in ['.tif', '.tiff', '.vrt']:
    p = os.path.join(DATA_DIR, 'dem' + ext)
    if os.path.isfile(p): dem_path = p; break
if dem_path: print(f"  ✓ DEM: {dem_path}")
else: print("  – DEM: не найден")

# Каталог результатов
OUT_DIR = os.path.join(DATA_DIR, f'results_{RUN_STAMP}')
os.makedirs(OUT_DIR, exist_ok=True)
SVG_DIR = os.path.join(OUT_DIR, 'svg')
os.makedirs(SVG_DIR, exist_ok=True)
print(f"  ✓ Выход → {OUT_DIR}")
print(f"  За {time.time()-t0:.1f}с")

# ═══════════════════════════════════════════════════════════════════════
#   CRS + ДАННЫЕ
# ═══════════════════════════════════════════════════════════════════════
print("\n▶ CRS + данные...")
t0 = time.time()

src_crs = gdf_egrn.crs
if src_crs is None:
    raise ValueError("ЕГРН без CRS")
if src_crs.is_geographic:
    bounds = gdf_egrn.total_bounds  # minx, miny, maxx, maxy
    cx_g = (bounds[0] + bounds[2]) / 2
    cy_g = (bounds[1] + bounds[3]) / 2
    zone = int((cx_g + 180) / 6) + 1
    epsg = (32600 if cy_g >= 0 else 32700) + zone
    metric_epsg = epsg
    print(f"  → EPSG:{epsg}")
else:
    metric_epsg = src_crs.to_epsg()
    print(f"  CRS метрическая: EPSG:{metric_epsg}")

# Перепроецируем ЕГРН
gdf_m = gdf_egrn.to_crs(epsg=metric_epsg)

# Парсинг VRI
def parse_vri_group(raw):
    if raw is None: return []
    s = str(raw).strip()
    if not s or s.lower() in ('n/a', 'null', 'none', ''): return []
    codes = []
    for part in s.split(';'):
        part = part.strip()
        try:
            v = int(float(part))
            if 1 <= v <= N_VRI_CLASSES: codes.append(v)
        except (ValueError, TypeError): continue
    return codes

has_land_cat = COL_LAND_CATEGORY in gdf_m.columns
if COL_VRI_GROUP not in gdf_m.columns:
    raise ValueError(f"'{COL_VRI_GROUP}' не найдена. Поля: {list(gdf_m.columns)}")

egrn_vri_all = []; egrn_vri_primary = []
egrn_centroids_m = []; egrn_land_cat = []
egrn_geom_m = []; egrn_idx = []

for i, row in gdf_m.iterrows():
    vri_list = parse_vri_group(row[COL_VRI_GROUP])
    if not vri_list: continue
    geom = row.geometry
    if geom is None or geom.is_empty: continue
    c = geom.centroid
    if c is None or c.is_empty: continue
    egrn_vri_all.append(vri_list)
    egrn_vri_primary.append(vri_list[0])
    egrn_centroids_m.append((c.x, c.y))
    egrn_geom_m.append(geom)
    egrn_idx.append(i)
    lc = row[COL_LAND_CATEGORY] if has_land_cat else None
    egrn_land_cat.append(str(lc) if lc is not None and str(lc).lower() not in ('none','nan','') else 'unknown')

egrn_vri = np.array(egrn_vri_primary, dtype=np.int32)
egrn_centroids_m = np.array(egrn_centroids_m, dtype=np.float64)
N = len(egrn_vri)
if N == 0: raise ValueError("Нет участков с ВРИ!")

_vri_masks_cache = {}
def vri_mask(v):
    if v not in _vri_masks_cache:
        _vri_masks_cache[v] = np.array([v in codes for codes in egrn_vri_all])
    return _vri_masks_cache[v]

vri_counts = Counter()
for vl in egrn_vri_all:
    for v in vl: vri_counts[v] += 1
n_multi = sum(1 for vl in egrn_vri_all if len(vl) > 1)
print(f"  Участков: {N}  (мульти-ВРИ: {n_multi})")
for v in sorted(vri_counts): print(f"    ВРИ {v:>2}: {vri_counts[v]}")
print(f"  За {time.time()-t0:.1f}с")

# ═══════════════════════════════════════════════════════════════════════
#   ПРИЗНАКИ
# ═══════════════════════════════════════════════════════════════════════
print("\n▶ Извлечение признаков...")
t_total = time.time()
feat_slope = np.zeros(N, dtype=np.float64)
feat_land_cat = egrn_land_cat
feat_poi_dens = np.zeros(N, dtype=np.float64)
feat_road_dist = np.full(N, 10000.0, dtype=np.float64)

# f1: SLOPE
print("  f1: slope..."); t1 = time.time()
slope_arr = None; dem_transform = None; dem_crs = None
dem_rows = 0; dem_cols = 0
if dem_path and HAS_RASTERIO:
    with rasterio.open(dem_path) as ds:
        dem_arr_raw = ds.read(1).astype(np.float64)
        nodata = ds.nodata
        if nodata is not None:
            dem_arr_raw[dem_arr_raw == nodata] = np.nan
        dem_transform = ds.transform
        dem_crs = ds.crs
        dem_rows, dem_cols = dem_arr_raw.shape
        res_x = abs(dem_transform.a)
        res_y = abs(dem_transform.e)
        if ds.crs.is_geographic:
            clat = (ds.bounds.bottom + ds.bounds.top) / 2
            rx = res_x * 111320 * math.cos(math.radians(clat))
            ry = res_y * 110540
        else:
            rx = res_x; ry = res_y
        dy, dx = np.gradient(dem_arr_raw, ry, rx)
        slope_arr = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        slope_arr[np.isnan(dem_arr_raw)] = np.nan

    # Сэмплирование slope по центроидам
    from pyproj import Transformer
    tr_to_dem = Transformer.from_crs(f"EPSG:{metric_epsg}", dem_crs, always_xy=True)
    for i in range(N):
        px, py = tr_to_dem.transform(egrn_centroids_m[i][0], egrn_centroids_m[i][1])
        ci_d = int((px - dem_transform.c) / dem_transform.a)
        ri_d = int((py - dem_transform.f) / dem_transform.e)
        vs = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r2, c2 = ri_d + dr, ci_d + dc
                if 0 <= r2 < dem_rows and 0 <= c2 < dem_cols:
                    sv = slope_arr[r2, c2]
                    if not np.isnan(sv): vs.append(sv)
        feat_slope[i] = np.mean(vs) if vs else 0.0
    print(f"    min={feat_slope.min():.2f}° mean={feat_slope.mean():.2f}° max={feat_slope.max():.2f}°")
else:
    print("    ⚠ DEM не загружен")
print(f"    За {time.time()-t1:.1f}с")

# f2: категория
print(f"  f2: категория земель — {len(set(feat_land_cat))} уникальных")

# f3: POI density
print(f"  f3: POI density..."); t1 = time.time()
poi_centroids_m = []; poi_tree = None
area_km2 = math.pi * (POI_RADIUS_M / 1000)**2
for gdf_p in [gdf_poi_pt, gdf_poi_pg]:
    if gdf_p is None: continue
    gp = gdf_p.to_crs(epsg=metric_epsg)
    for geom in gp.geometry:
        if geom is None or geom.is_empty: continue
        c = geom.centroid
        poi_centroids_m.append((c.x, c.y))
print(f"    POI: {len(poi_centroids_m)}")
if poi_centroids_m:
    poi_arr = np.array(poi_centroids_m, dtype=np.float64)
    poi_tree = cKDTree(poi_arr)
    counts = poi_tree.query_ball_point(egrn_centroids_m, POI_RADIUS_M)
    feat_poi_dens = np.array([len(c) / area_km2 for c in counts])
    print(f"    min={feat_poi_dens.min():.1f} mean={feat_poi_dens.mean():.1f} max={feat_poi_dens.max():.1f}")
print(f"    За {time.time()-t1:.1f}с")

# f4: road distance
print("  f4: road dist..."); t1 = time.time()
road_tree = None; road_points = []
if gdf_roads is not None:
    gr = gdf_roads.to_crs(epsg=metric_epsg)
    has_hw = COL_HIGHWAY in gr.columns
    for _, row in gr.iterrows():
        if has_hw:
            hw = row[COL_HIGHWAY]
            if hw is None or str(hw) not in ROAD_LEVELS_FILTER: continue
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        ln = geom.length
        if ln <= 0: continue
        step = max(50.0, ln / 50.0); d = 0.0
        while d <= ln:
            pt = geom.interpolate(d)
            road_points.append((pt.x, pt.y))
            d += step
    print(f"    Точек: {len(road_points)}")
    if road_points:
        road_tree = cKDTree(np.array(road_points, dtype=np.float64))
        dists, _ = road_tree.query(egrn_centroids_m)
        feat_road_dist = dists.astype(np.float64)
        print(f"    min={feat_road_dist.min():.0f}м mean={feat_road_dist.mean():.0f}м")
print(f"    За {time.time()-t1:.1f}с")

# f5: ЗОУИТ
print("  f5: ЗОУИТ..."); t1 = time.time()
ZOUIT_NORMS = [
    ('Z01',[r'водоохран']),('Z02',[r'прибрежн']),
    ('Z03',[r'затопл',r'подтопл']),('Z04',[r'санитарно.*защит']),
    ('Z05',[r'санитарн.*охран']),('Z06',[r'электр.*энерг',r'электросет']),
    ('Z07',[r'трубопровод',r'газопровод',r'нефтепровод']),
    ('Z08',[r'придорожн']),('Z09',[r'линий.*связи',r'сооружений.*связи']),
    ('Z10',[r'культурн.*наслед']),('Z99',[r'.*']),
]
def norm_zouit(text):
    if not text: return 'Z99'
    t = str(text).lower().strip()
    for code, pats in ZOUIT_NORMS:
        for p in pats:
            if re.search(p, t): return code
    return 'Z99'

zouit_by_parcel = defaultdict(set); all_zouit_codes = set()
if gdf_zouit is not None and COL_ZOUIT_TYPE in gdf_zouit.columns:
    gz = gdf_zouit.to_crs(epsg=metric_epsg)
    # STRtree по участкам ЕГРН
    tree_egrn = STRtree(egrn_geom_m)
    zc = 0
    for _, zrow in gz.iterrows():
        zt = norm_zouit(zrow.get(COL_ZOUIT_TYPE)); all_zouit_codes.add(zt)
        zg = zrow.geometry
        if zg is None or zg.is_empty: continue
        hits = tree_egrn.query(zg, predicate='intersects')
        for idx in hits:
            zouit_by_parcel[idx].add(zt)
        zc += 1
        if zc % 500 == 0: print(f"      зон: {zc}")
    nwz = sum(1 for v in zouit_by_parcel.values() if v)
    print(f"    В ЗОУИТ: {nwz} ({nwz/N*100:.1f}%)")
else:
    print("    ⚠ ЗОУИТ не загружен")

zouit_codes_sorted = sorted(all_zouit_codes - {'Z99'})
n_zouit_types = len(zouit_codes_sorted)
zouit_code_idx = {c: i for i, c in enumerate(zouit_codes_sorted)}
feat_zouit_binary = np.zeros((N, max(n_zouit_types, 1)), dtype=np.int8)
for idx, codes in zouit_by_parcel.items():
    for c in codes:
        if c in zouit_code_idx: feat_zouit_binary[idx, zouit_code_idx[c]] = 1
print(f"    Флагов: {n_zouit_types}")
print(f"    За {time.time()-t1:.1f}с")
print(f"\n  ✓ Признаки за {(time.time()-t_total)/60:.1f} мин")

# ═══════════════════════════════════════════════════════════════════════
#   ЛИКЕЛИХУДЫ
# ═══════════════════════════════════════════════════════════════════════
print("\n▶ Обучение Naive Bayes...")
t0 = time.time()
active_classes = sorted([int(v) for v, c in vri_counts.items() if c >= MIN_CLASS_SIZE])
if not active_classes: raise ValueError("Нет классов!")
total_active = sum(vri_counts[v] for v in active_classes)
prior = {v: vri_counts[v] / total_active for v in active_classes}
log_prior = {v: np.log(p) for v, p in prior.items()}

# Для оценки территорий — равномерный приор (каждый ВРИ на равных)
# Обоснование: оцениваем пригодность по пространственным характеристикам,
# а не по частоте существующей застройки
UNIFORM_PRIOR_FOR_ASSESSMENT = True
_unif = 1.0 / len(active_classes)
log_prior_assess = {v: np.log(_unif) for v in active_classes}
print(f"  Приор для оценки: {'uniform' if UNIFORM_PRIOR_FOR_ASSESSMENT else 'data'}")

CONTINUOUS_FEATURES = ['slope', 'poi_density', 'road_dist']
continuous_data = {'slope': feat_slope, 'poi_density': feat_poi_dens, 'road_dist': feat_road_dist}
kde_models = {}
for fn, data in continuous_data.items():
    for v in active_classes:
        vals = data[vri_mask(v)]; vals = vals[np.isfinite(vals)]
        if len(vals) < MIN_CLASS_SIZE: continue
        if np.std(vals) < 1e-10: vals = vals + np.random.normal(0, 1e-6, len(vals))
        try: kde_models[(fn, v)] = gaussian_kde(vals, bw_method='silverman')
        except: pass
print(f"  KDE: {len(kde_models)}")

cat_unique = sorted(set(feat_land_cat)); n_cat_values = len(cat_unique)
cat_counts = {}
for v in active_classes:
    m = vri_mask(v)
    cat_counts[v] = Counter([feat_land_cat[i] for i in range(N) if m[i]])

def log_lik_cat(cat, vri):
    if vri not in cat_counts: return np.log(1.0 / max(n_cat_values, 1))
    cn = cat_counts[vri]; t = sum(cn.values())
    return np.log((cn.get(cat, 0) + 1) / (t + n_cat_values))

zouit_pos = {}
for v in active_classes:
    m = vri_mask(v); nv = int(m.sum())
    for j in range(n_zouit_types):
        zouit_pos[(j, v)] = (int(feat_zouit_binary[m, j].sum()), nv)

def log_lik_zouit(zvec, vri):
    ll = 0.0
    for j in range(n_zouit_types):
        pos, tot = zouit_pos.get((j, vri), (0, 1))
        p = ((pos + 1) / (tot + 2)) if zvec[j] == 1 else ((tot - pos + 1) / (tot + 2))
        ll += np.log(max(p, 1e-15))
    return ll
print(f"  За {time.time()-t0:.1f}с")

# ═══════════════════════════════════════════════════════════════════════
#   КАЛИБРОВКА α
# ═══════════════════════════════════════════════════════════════════════
print("\n▶ Калибровка α...")
t0 = time.time()

def compute_lp(idx, alpha):
    res = {}
    for v in active_classes:
        ll = 0.0
        for fn in CONTINUOUS_FEATURES:
            k = (fn, v)
            if k in kde_models:
                d = kde_models[k].evaluate(np.array([continuous_data[fn][idx]]))[0]
                ll += np.log(max(d, 1e-15))
        ll += log_lik_cat(feat_land_cat[idx], v)
        ll += log_lik_zouit(feat_zouit_binary[idx], v)
        res[v] = log_prior[v] + alpha * ll
    return res

def ll_single(idx, alpha):
    tv_set = set(egrn_vri_all[idx]) & set(active_classes)
    if not tv_set: return 0.0
    lp = compute_lp(idx, alpha)
    va = np.array(list(lp.values())); mx = np.max(va)
    ln = mx + np.log(np.sum(np.exp(va - mx)))
    best_lp = max(lp[tv] for tv in tv_set if tv in lp)
    return -(best_lp - ln)

np.random.seed(42)
csz = min(5000, max(200, N // 10))
cidx = np.random.choice(N, csz, replace=False)
def tot_ll(alpha):
    return sum(ll_single(i, alpha) for i in cidx) / len(cidx)

best_alpha = 1.0; best_loss = float('inf'); alpha_losses = []
for a in np.arange(0.05, 1.55, 0.05):
    l = tot_ll(a); alpha_losses.append((float(a), float(l)))
    if l < best_loss: best_loss = l; best_alpha = a
for a in np.arange(max(0.01, best_alpha - 0.1), best_alpha + 0.11, 0.01):
    l = tot_ll(a)
    if l < best_loss: best_loss = l; best_alpha = a
ALPHA = round(float(best_alpha), 3)
baseline_loss = np.log(len(active_classes))
print(f"  α={ALPHA}, loss={best_loss:.4f} (base={baseline_loss:.4f})")
print(f"  За {time.time()-t0:.1f}с")

# ═══════════════════════════════════════════════════════════════════════
#   SPATIAL CV
# ═══════════════════════════════════════════════════════════════════════
print(f"\n▶ Spatial CV (buf={SPATIAL_BUFFER_M}м)...")
t0 = time.time()
cv_size = min(3000, N)
cv_idx = np.random.choice(N, cv_size, replace=False)
all_tree = cKDTree(egrn_centroids_m)
top_hits = {1: 0, 3: 0, 5: 0}; cv_lls = []; cv_tested = 0; cv_ranks = []

for ci, ti in enumerate(cv_idx):
    tv_set = set(egrn_vri_all[ti]) & set(active_classes)
    if not tv_set: continue
    nb = all_tree.query_ball_point(egrn_centroids_m[ti], SPATIAL_BUFFER_M)
    exc = set(nb)
    nec = Counter()
    for idx_ in exc:
        for v_ in egrn_vri_all[idx_]: nec[v_] += 1
    at = total_active - sum(nec[v_] for v_ in nec if v_ in active_classes)
    if at <= 0: continue
    lps = {}
    for v in active_classes:
        ac = vri_counts[v] - nec.get(v, 0)
        if ac <= 0: continue
        ap = np.log(ac / at); ll = 0.0
        for fn in CONTINUOUS_FEATURES:
            k = (fn, v)
            if k in kde_models:
                d = kde_models[k].evaluate(np.array([continuous_data[fn][ti]]))[0]
                ll += np.log(max(d, 1e-15))
        ll += log_lik_cat(feat_land_cat[ti], v)
        ll += log_lik_zouit(feat_zouit_binary[ti], v)
        lps[v] = ap + ALPHA * ll
    if not lps: continue
    va = np.array(list(lps.values())); ka = list(lps.keys())
    mx = np.max(va); ln = mx + np.log(np.sum(np.exp(va - mx)))
    probs = {k: float(np.exp(lp - ln)) for k, lp in zip(ka, va)}
    ranked = sorted(probs.keys(), key=lambda c: -probs[c])
    ranks_true = [ranked.index(tv) + 1 for tv in tv_set if tv in ranked]
    rk = min(ranks_true) if ranks_true else len(ranked) + 1
    cv_ranks.append(rk)
    p_true = max(probs.get(tv, 1e-15) for tv in tv_set)
    cv_lls.append(-np.log(max(p_true, 1e-15)))
    for k in top_hits:
        if tv_set & set(ranked[:k]): top_hits[k] += 1
    cv_tested += 1
    if (ci + 1) % 500 == 0:
        print(f"    {ci+1}/{cv_size} t1={top_hits[1]/max(cv_tested,1):.1%}")

cv_results = {}
print(f"\n  ── CV ({cv_tested} уч.) ──")
for k in sorted(top_hits):
    a = top_hits[k] / max(cv_tested, 1); cv_results[k] = a; print(f"  Top-{k}: {a:.1%}")
mean_ll = float(np.mean(cv_lls)) if cv_lls else 0.0
cv_results['log_loss'] = mean_ll
print(f"  Log-loss: {mean_ll:.4f} (base={baseline_loss:.4f})")
if cv_ranks: print(f"  Ср. ранг: {np.mean(cv_ranks):.2f}")
print(f"  За {time.time()-t0:.1f}с")

# ═══════════════════════════════════════════════════════════════════════
#   BOUNDARY
# ═══════════════════════════════════════════════════════════════════════
print("\n▶ Оценка территорий...")
t0 = time.time()

def _softmax(log_dict):
    va = np.array(list(log_dict.values()))
    ka = list(log_dict.keys())
    mx = np.max(va); ln = mx + np.log(np.sum(np.exp(va - mx)))
    return {k: float(np.exp(lp - ln)) for k, lp in zip(ka, va)}

def assess_pt_pe(sl, lc, pd, rd, zv):
    _lp = log_prior_assess if UNIFORM_PRIOR_FOR_ASSESSMENT else log_prior
    lp_P = {}; lp_E = {}; lp_full = {}
    for v in active_classes:
        ll_p = 0.0
        for fn, val in [('slope', sl)]:
            k = (fn, v)
            if k in kde_models:
                d = kde_models[k].evaluate(np.array([val]))[0]
                ll_p += np.log(max(d, 1e-15))
        ll_p += log_lik_cat(lc, v)
        ll_p += log_lik_zouit(zv, v)
        lp_P[v] = _lp[v] + ALPHA * ll_p
        ll_e = 0.0
        for fn, val in [('poi_density', pd), ('road_dist', rd)]:
            k = (fn, v)
            if k in kde_models:
                d = kde_models[k].evaluate(np.array([val]))[0]
                ll_e += np.log(max(d, 1e-15))
        lp_E[v] = _lp[v] + ALPHA * ll_e
        lp_full[v] = _lp[v] + ALPHA * (ll_p + ll_e)
    prob_P = _softmax(lp_P); prob_E = _softmax(lp_E); prob_full = _softmax(lp_full)

    # Min-max нормализация в пределах оцениваемого участка:
    # лучший ВРИ → 1.0, худший → 0.0. Приводит индексы к единой шкале [0, 1]
    # и устраняет «слипание» в углу матрицы, обусловленное нормировкой softmax
    # по 13 классам (среднее значение ≈ 1/13 ≈ 0.077).
    def _minmax(d):
        vs = np.array(list(d.values()), dtype=float)
        lo, hi = float(vs.min()), float(vs.max())
        rng = hi - lo
        if rng < 1e-12:
            return {k: 0.5 for k in d}
        return {k: float((v - lo) / rng) for k, v in d.items()}

    P_norm = _minmax(prob_P)
    E_norm = _minmax(prob_E)

    res = []
    for v in active_classes:
        pi_raw = prob_P[v]; ei_raw = prob_E[v]
        si_raw = math.sqrt(pi_raw * ei_raw) if pi_raw > 0 and ei_raw > 0 else 0.0
        pi = P_norm[v]; ei = E_norm[v]
        si = math.sqrt(pi * ei) if pi > 0 and ei > 0 else 0.0
        res.append({'vri': v,
                    'P': round(pi, 6), 'E': round(ei, 6), 'S': round(si, 6),
                    'P_raw': round(pi_raw, 6), 'E_raw': round(ei_raw, 6),
                    'S_raw': round(si_raw, 6),
                    'prob': round(prob_full[v], 6),
                    'log_post': round(lp_full[v], 4)})
    res.sort(key=lambda x: -x['S'])
    return res

boundary_results = []; _bid = 0
# Подготовка ЗОУИТ для boundary
gz_m = None
if gdf_zouit is not None and COL_ZOUIT_TYPE in gdf_zouit.columns:
    gz_m = gdf_zouit.to_crs(epsg=metric_epsg)

for gdf_b, bn in boundary_gdfs:
    gb = gdf_b.to_crs(epsg=metric_epsg)
    for _, brow in gb.iterrows():
        g = brow.geometry
        if g is None or g.is_empty: continue
        _bid += 1
        c = g.centroid; ah = g.area / 10000
        print(f"\n  ── Участок {_bid} «{bn}» ({ah:.2f} га) ──")
        # f1 slope
        bs = 0.0
        if slope_arr is not None:
            from pyproj import Transformer
            tr_to_dem2 = Transformer.from_crs(f"EPSG:{metric_epsg}", dem_crs, always_xy=True)
            px, py = tr_to_dem2.transform(c.x, c.y)
            ci_ = int((px - dem_transform.c) / dem_transform.a)
            ri_ = int((py - dem_transform.f) / dem_transform.e)
            if 0 <= ri_ < dem_rows and 0 <= ci_ < dem_cols:
                sv = slope_arr[ri_, ci_]; bs = 0.0 if np.isnan(sv) else float(sv)
        # f2 category
        dd = np.sqrt((egrn_centroids_m[:, 0] - c.x)**2 + (egrn_centroids_m[:, 1] - c.y)**2)
        n20 = np.argsort(dd)[:20]
        blc = Counter([feat_land_cat[i] for i in n20]).most_common(1)[0][0]
        # f3 poi
        bpd = 0.0
        if poi_tree: bpd = len(poi_tree.query_ball_point([c.x, c.y], POI_RADIUS_M)) / area_km2
        # f4 road
        brd = 10000.0
        if road_tree:
            dr_, _ = road_tree.query(np.array([[c.x, c.y]])); brd = float(dr_[0])
        # f5 zouit
        bz = np.zeros(max(n_zouit_types, 1), dtype=np.int8)
        if gz_m is not None and n_zouit_types > 0:
            for _, zrow in gz_m.iterrows():
                zg = zrow.geometry
                if zg is None or zg.is_empty: continue
                if g.intersects(zg):
                    zc_ = norm_zouit(zrow.get(COL_ZOUIT_TYPE))
                    if zc_ in zouit_code_idx: bz[zouit_code_idx[zc_]] = 1

        rk = assess_pt_pe(bs, blc, bpd, brd, bz)
        print(f"    slope={bs:.2f}° cat='{blc}' poi={bpd:.1f} road={brd:.0f}м")
        print(f"    {'ВРИ':>6}  {'Название':<35}  {'S':>6}  {'P':>6}  {'E':>6}")
        for r in rk[:5]:
            print(f"    {r['vri']:>6}  {VRI_NAMES.get(r['vri'],''):<35}  "
                  f"{r['S']:.4f}  {r['P']:.4f}  {r['E']:.4f}")
        boundary_results.append({'id': _bid, 'layer': bn, 'area_ha': round(ah, 2),
            'centroid_m': (c.x, c.y),
            'chars': {'slope': bs, 'land_category': blc, 'poi_density': bpd,
                      'road_distance': brd,
                      'zouit_flags': {zouit_codes_sorted[j]: int(bz[j]) for j in range(n_zouit_types)}},
            'ranking': rk})
print(f"\n  За {time.time()-t0:.1f}с")

# ═══════════════════════════════════════════════════════════════════════
#   ЭКСПОРТ GPKG + JSON
# ═══════════════════════════════════════════════════════════════════════
print("\n▶ Экспорт...")
t0 = time.time()
gpkg_path = os.path.join(OUT_DIR, 'egrn_features.gpkg')

# Собираем GeoDataFrame для экспорта
export_data = []
for i in range(N):
    row = {
        'vri': int(egrn_vri[i]),
        'vri_all': ';'.join(str(v) for v in egrn_vri_all[i]),
        'f1_slope': float(feat_slope[i]),
        'f2_land_cat': feat_land_cat[i],
        'f3_poi_density': float(feat_poi_dens[i]),
        'f4_road_dist': float(feat_road_dist[i]),
    }
    for j, zc in enumerate(zouit_codes_sorted):
        row[f'f5_{zc}'] = int(feat_zouit_binary[i, j])
    row['geometry'] = egrn_geom_m[i]
    export_data.append(row)

gdf_out = gpd.GeoDataFrame(export_data, crs=f"EPSG:{metric_epsg}")
gdf_out.to_file(gpkg_path, driver='GPKG')
print(f"  ✓ {gpkg_path}")

# JSON
rj = {'meta': {'version': 'v8.4-cloud', 'timestamp': datetime.now().isoformat(), 'n': N,
    'n_multi_vri': n_multi, 'metric_epsg': metric_epsg,
    'classes': len(active_classes), 'alpha': ALPHA,
    'prior_mode': 'uniform' if UNIFORM_PRIOR_FOR_ASSESSMENT else 'data',
    'features': ['slope', 'land_cat', 'poi_density', 'road_dist', 'zouit_flags'],
    'pe_normalization': 'minmax_within_site',
    'pe_normalization_note': ('P и E в ranking[] нормализованы min-max в пределах участка '
                              '(лучший ВРИ=1.0, худший=0.0); сырые softmax-значения '
                              'сохранены в P_raw, E_raw, S_raw'),
    'zouit_codes': zouit_codes_sorted},
    'prior': {str(v): round(p, 6) for v, p in prior.items()},
    'cv': {'method': f'spatial_loo_{SPATIAL_BUFFER_M}m', 'n': cv_tested,
        'top1': round(cv_results.get(1, 0), 4), 'top3': round(cv_results.get(3, 0), 4),
        'top5': round(cv_results.get(5, 0), 4), 'log_loss': round(mean_ll, 4),
        'baseline': round(baseline_loss, 4)},
    'boundaries': boundary_results, 'alpha_cal': alpha_losses}
jp = os.path.join(OUT_DIR, 'results.json')
with open(jp, 'w', encoding='utf-8') as f:
    json.dump(rj, f, ensure_ascii=False, indent=2, default=str)
print(f"  ✓ {jp}")
print(f"  За {time.time()-t0:.1f}с")

# ═══════════════════════════════════════════════════════════════════════
#   SVG (копия из v8.1, без изменений)
# ═══════════════════════════════════════════════════════════════════════
if HAS_MPL:
    print("\n▶ SVG...")
    t0 = time.time()
    _BG = '#F2F2F2'
    _MAIN = '#D4AA42'
    _C = {'tx': '#2C2C2A', 'mu': '#777770', 'grid': '#C8C8C0',
          'red': '#C05040', 'blue': '#5080B0', 'green': '#508050'}
    _F = 12
    _FDPI = 96
    _SZ_01_02 = (910/_FDPI, 1032/_FDPI)
    _SZ_03    = (600/_FDPI, 681/_FDPI)
    _SZ_04    = (1840/_FDPI, 681/_FDPI)
    _SZ_05_06 = (600/_FDPI, 681/_FDPI)
    _SZ_07    = (1065/_FDPI, 1032/_FDPI)

    import matplotlib.font_manager as fm
    _GOLOS = None
    for fp in fm.findSystemFonts():
        if 'golos' in fp.lower():
            _GOLOS = fm.FontProperties(fname=fp).get_name(); break
    plt.rcParams.update({'axes.unicode_minus': False, 'svg.fonttype': 'none',
        'font.size': _F, 'font.family': _GOLOS or 'sans-serif',
        'axes.facecolor': _BG, 'figure.facecolor': _BG, 'savefig.facecolor': _BG})

    _VRI_SHORT = {1:'С/х',2:'Жилая',3:'Общественн.',4:'Бизнес',5:'Рекреация',
        6:'Производств.',7:'Транспорт',8:'Оборона',9:'Природа',10:'Леса',
        11:'Водные',12:'Общ. польз.',13:'Общ. назнач.'}
    _KDE_LABELS = {'slope':'Уклон (°)','poi_density':'Плотность POI (шт/км²)',
        'road_dist':'Расстояние до дороги (м)'}
    _KDE_COLORS = {1:'#B08830',2:'#D4AA42',3:'#5080B0',4:'#C05040',5:'#508050',
        6:'#888078',7:'#405060',8:'#905040',9:'#407040',10:'#306030',
        11:'#4878A0',12:'#806898',13:'#A09030'}

    def _sa(ax):
        for s in ax.spines.values(): s.set_color(_C['grid']); s.set_linewidth(0.5)
        ax.tick_params(colors=_C['tx'], labelsize=_F); ax.set_facecolor(_BG)
    def _sv(fig, nm):
        fig.savefig(os.path.join(SVG_DIR, f'{nm}.svg'), format='svg', facecolor=_BG)
        plt.close(fig); print(f"    ✓ {nm}.svg")

    # 01 prior
    fig,ax=plt.subplots(figsize=_SZ_01_02);fig.patch.set_facecolor(_BG);_sa(ax)
    vs=sorted(active_classes);cn=[vri_counts[v] for v in vs]
    lb=[f"{v}. {VRI_NAMES.get(v,'')[:25]}" for v in vs]
    ax.barh(range(len(vs)),cn,color=_MAIN,height=0.6)
    for i,c in enumerate(cn): ax.text(c+max(cn)*0.01,i,f"{c}",va='center',fontsize=_F,color=_C['tx'])
    ax.set_yticks(range(len(vs)));ax.set_yticklabels(lb,fontsize=_F);ax.invert_yaxis()
    ax.set_xlabel('Количество участков',fontsize=_F)
    fig.subplots_adjust(left=0.38,right=0.95,top=0.97,bottom=0.06);_sv(fig,'01_prior')

    # 02 alpha (600×681)
    fig,ax=plt.subplots(figsize=_SZ_05_06);fig.patch.set_facecolor(_BG);_sa(ax)
    ax.plot([a for a,_ in alpha_losses],[l for _,l in alpha_losses],color=_MAIN,lw=2)
    ax.axhline(baseline_loss,color=_C['red'],ls='--',lw=1,label=f'Базовый loss = {baseline_loss:.3f}')
    ax.axvline(ALPHA,color=_C['green'],ls='--',lw=1,label=f'Оптимальный α = {ALPHA}')
    ax.set_xlabel('Коэффициент α',fontsize=_F);ax.set_ylabel('Средний log-loss',fontsize=_F)
    ax.legend(fontsize=_F-2,facecolor=_BG,edgecolor=_C['grid'])
    fig.subplots_adjust(left=0.16,right=0.96,top=0.97,bottom=0.10);_sv(fig,'02_alpha')

    # 03 KDE — с линией общего тренда, улучшенной читаемостью
    for fn in CONTINUOUS_FEATURES:
        fig,ax=plt.subplots(figsize=_SZ_03);fig.patch.set_facecolor(_BG);_sa(ax)
        fv=continuous_data[fn];fv2=fv[np.isfinite(fv)]
        if len(fv2)==0: plt.close(fig); continue
        xr=np.linspace(np.percentile(fv2,1),np.percentile(fv2,99),300)

        # Линии классов: чуть толще, чуть прозрачнее, чтобы тренд читался поверх
        cls_max=[]
        for v in active_classes:
            k=(fn,v)
            if k not in kde_models: continue
            yv=kde_models[k].evaluate(xr)
            cls_max.append(float(np.max(yv)))
            ax.plot(xr,yv,color=_KDE_COLORS.get(v,'#888'),
                    lw=1.6,alpha=0.75,label=f'{v}. {_VRI_SHORT.get(v,"")}')

        # Линия общего тренда: KDE по всем данным, без разделения на ВРИ
        try:
            _trend_kde=gaussian_kde(fv2,bw_method='silverman')
            yt=_trend_kde.evaluate(xr)
            cls_max.append(float(np.max(yt)))
            ax.plot(xr,yt,color=_C['tx'],lw=2.6,ls='--',alpha=0.95,
                    label='Общий тренд (все ВРИ)',zorder=20)
        except Exception:
            pass

        # Адаптивный верх оси Y: 99-й перцентиль максимумов с небольшим запасом
        if cls_max:
            yhi=float(np.percentile(cls_max,99))*1.15
            if yhi>0: ax.set_ylim(0,yhi)

        ax.set_xlabel(_KDE_LABELS.get(fn,fn),fontsize=_F)
        ax.set_ylabel('P(f|ВРИ)',fontsize=_F)
        # Легенда вынесена под график — не закрывает кривые
        ax.legend(fontsize=_F-5,ncol=4,loc='upper center',
                  bbox_to_anchor=(0.5,-0.14),
                  facecolor=_BG,edgecolor=_C['grid'],frameon=True)
        fig.subplots_adjust(left=0.16,right=0.97,top=0.97,bottom=0.30);_sv(fig,f'03_kde_{fn}')

    # 04 CV — разделено на два самостоятельных графика 600×681
    if cv_tested>0:
        # 04a — точность Top-1/3/5
        fig,a1=plt.subplots(figsize=_SZ_05_06);fig.patch.set_facecolor(_BG);_sa(a1)
        tk=[1,3,5];ac=[cv_results.get(k,0)*100 for k in tk]
        a1.bar(range(3),ac,color=[_C['red'],_MAIN,_C['green']],width=0.5)
        for i,a in enumerate(ac): a1.text(i,a+2,f"{a:.1f}%",ha='center',fontweight='bold',fontsize=_F)
        a1.set_xticks(range(3));a1.set_xticklabels([f"Top-{k}" for k in tk],fontsize=_F)
        a1.set_ylim(0,105);a1.set_ylabel('Точность (%)',fontsize=_F)
        fig.subplots_adjust(left=0.16,right=0.96,top=0.97,bottom=0.10);_sv(fig,'04a_cv_accuracy')

        # 04b — распределение рангов истинного класса
        if cv_ranks:
            fig,a2=plt.subplots(figsize=_SZ_05_06);fig.patch.set_facecolor(_BG);_sa(a2)
            mr=min(max(cv_ranks),13)
            a2.hist(cv_ranks,bins=list(range(1,mr+2)),color=_MAIN,alpha=0.8,align='left',rwidth=0.8)
            a2.set_xlabel('Ранг истинного класса',fontsize=_F);a2.set_ylabel('Количество',fontsize=_F)
            fig.subplots_adjust(left=0.16,right=0.96,top=0.97,bottom=0.10);_sv(fig,'04b_cv_ranks')

    # 05 ranking
    for br in boundary_results:
        rk=br['ranking'];fig,ax=plt.subplots(figsize=_SZ_05_06)
        fig.patch.set_facecolor(_BG);_sa(ax)
        ss=[r['S'] for r in rk];lb=[f"{r['vri']}. {_VRI_SHORT.get(r['vri'],'')}" for r in rk]
        cl=[_C['green'] if r['S']>=0.6 else _MAIN if r['S']>=0.4 else _C['red'] for r in rk]
        ax.barh(range(len(rk)),ss,color=cl,height=0.55)
        for i,r in enumerate(rk):
            if r['S']>0.01: ax.text(r['S']+0.005,i,f"{r['S']:.3f}",va='center',fontsize=_F-4,color=_C['tx'])
        ax.axvline(0.6,color=_C['green'],ls='--',lw=0.7,alpha=0.5)
        ax.axvline(0.4,color=_MAIN,ls='--',lw=0.7,alpha=0.5)
        ax.set_yticks(range(len(rk)));ax.set_yticklabels(lb,fontsize=_F-4);ax.invert_yaxis()
        ax.set_xlabel('S = √(P × E)',fontsize=_F);ax.set_xlim(0,max(ss)*1.18 if ss else 1)
        fig.subplots_adjust(left=0.32,right=0.95,top=0.97,bottom=0.10);_sv(fig,f"05_ranking_T{br['id']}")

    # 06 PE matrix
    for br in boundary_results:
        rk=br['ranking'];fig,ax=plt.subplots(figsize=_SZ_05_06)
        fig.patch.set_facecolor(_BG);_sa(ax)
        ax.axhline(y=0.5,color=_C['grid'],ls='--',lw=0.8);ax.axvline(x=0.5,color=_C['grid'],ls='--',lw=0.8)
        ax.fill_between([0.5,1.05],0.5,1.05,color=_C['green'],alpha=0.05)
        _qf=_F-5
        ax.text(0.75,0.04,'Высокий E\nНизкий P',ha='center',fontsize=_qf,color=_C['mu'],alpha=0.6)
        ax.text(0.25,0.96,'Высокий P\nНизкий E',ha='center',va='top',fontsize=_qf,color=_C['mu'],alpha=0.6)
        ax.text(0.75,0.96,'Высокий\nP+E',ha='center',va='top',fontsize=_qf,color=_C['green'],alpha=0.6)
        ax.text(0.25,0.04,'Низкий\nP+E',ha='center',fontsize=_qf,color=_C['mu'],alpha=0.6)
        for r in rk:
            v=r['vri'];pi_=r['P'];ei_=r['E'];si_=r['S']
            sz=max(si_*500,20)
            c=_C['green'] if si_>=0.6 else _MAIN if si_>=0.4 else _C['red']
            ax.scatter(ei_,pi_,s=sz,c=c,alpha=0.75,edgecolors=_C['grid'],lw=0.8,zorder=5)
            ax.annotate(f"{v}. {_VRI_SHORT.get(v,'')}",(ei_,pi_),textcoords='offset points',
                xytext=(5,-11),ha='left',fontsize=_F-5,color=_C['tx'],
                bbox=dict(boxstyle='round,pad=0.1',fc=_BG,alpha=0.85,ec='none'))
        ax.set_xlim(-0.03,1.03);ax.set_ylim(-0.03,1.03)
        ax.set_xlabel('Индекс эффекта (E)',fontsize=_F);ax.set_ylabel('Индекс потенциала (P)',fontsize=_F)
        fig.subplots_adjust(left=0.16,right=0.97,top=0.97,bottom=0.10);_sv(fig,f"06_pe_matrix_T{br['id']}")

    # 07 radar
    if boundary_results:
        fig,ax=plt.subplots(figsize=_SZ_07,subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(_BG);ax.set_facecolor(_BG)
        ang=np.linspace(0,2*np.pi,len(active_classes),endpoint=False).tolist()
        rc=[_C['red'],_MAIN,_C['blue']]
        for bi,br in enumerate(boundary_results[:3]):
            sm={r['vri']:r['S'] for r in br['ranking']}
            va=[sm.get(v,0) for v in active_classes];vc=va+va[:1];ac=ang+ang[:1]
            ax.plot(ac,vc,color=rc[bi%3],lw=2,label=f"Участок {br['id']} ({br['area_ha']} га)")
            ax.fill(ac,vc,color=rc[bi%3],alpha=0.07)
        ax.set_xticks(ang)
        ax.set_xticklabels([f"{v}. {_VRI_SHORT.get(v,'')}" for v in active_classes],fontsize=_F-2)
        ax.set_ylim(0,1);ax.set_yticks([0.25,0.5,0.75])
        ax.set_yticklabels(['0.25','0.50','0.75'],fontsize=_F-2,color=_C['mu'])
        ax.grid(True,linestyle='-',linewidth=0.5,alpha=0.3,color=_C['grid'])
        ax.legend(loc='upper right',bbox_to_anchor=(1.22,1.06),fontsize=_F-1,facecolor=_BG,edgecolor=_C['grid'])
        fig.subplots_adjust(left=0.08,right=0.85,top=0.92,bottom=0.05);_sv(fig,'07_radar_S')
    print(f"  За {time.time()-t0:.1f}с")

# ═══════════════════════════════════════════════════════════════════════
#   ИТОГ
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  КОНВЕЙЕР v8.4 ЗАВЕРШЁН")
print("=" * 70)
print(f"  N={N} (мульти-ВРИ: {n_multi}), классов={len(active_classes)}, α={ALPHA}")
print(f"  Loss={best_loss:.4f} (base={baseline_loss:.4f})")
print(f"  Top-1/3/5: {cv_results.get(1,0):.1%}/{cv_results.get(3,0):.1%}/{cv_results.get(5,0):.1%}")
print(f"  Выход: {OUT_DIR}")
ns = len([f for f in os.listdir(SVG_DIR) if f.endswith('.svg')]) if os.path.isdir(SVG_DIR) else 0
print(f"  SVG: {ns} файлов")
print("=" * 70)

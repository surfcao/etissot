"""Area weights, weighted statistics and geographic block resampling."""
import numpy as np
LATS = np.arange(-85, 85.01, 2.5)
LONS = np.arange(-177.5, 177.51, 5.0)
GLAT, GLON = np.meshgrid(LATS, LONS, indexing='ij')
W = np.cos(np.radians(GLAT))

def wquantile(x, q, w=None):
    """Midpoint-interpolated quantile over finite, positive-weight observations."""
    x = np.asarray(x).ravel()
    w = W.ravel() if w is None else np.asarray(w, dtype=float).ravel()
    q = np.asarray(q, dtype=float)
    if x.size != w.size:
        raise ValueError('Values and weights must have the same size.')
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Weights must be finite and nonnegative.')
    if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('Quantiles must lie in [0, 1].')
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        raise ValueError('No finite observations with positive weight.')
    x, w = (x[ok], w[ok])
    o = np.argsort(x)
    x, w = (x[o], w[o])
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return np.interp(q, c, x)

def wmedian(x, w=None):
    return wquantile(x, 0.5, w)

def wcorr(a, b, w=None, mask=None):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    w = (W.ravel() if w is None else np.asarray(w).ravel()).copy()
    if not (a.size == b.size == w.size):
        raise ValueError('Values and weights must have the same size.')
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('Weights must be finite and nonnegative.')
    if mask is not None:
        mask = np.asarray(mask, dtype=float).ravel()
        if mask.size != a.size or not np.isfinite(mask).all() or np.any(mask < 0):
            raise ValueError('Mask must match the values and be finite and nonnegative.')
        w = w * mask
    ok = np.isfinite(a) & np.isfinite(b) & (w > 0)
    if not ok.any():
        raise ValueError('No finite paired observations with positive weight.')
    a, b, w = (a[ok], b[ok], w[ok])
    ma, mb = (np.average(a, weights=w), np.average(b, weights=w))
    denominator = np.sqrt(np.sum(w * (a - ma) ** 2) * np.sum(w * (b - mb) ** 2))
    return np.sum(w * (a - ma) * (b - mb)) / denominator if denominator > 0 else float('nan')

def block_ids(bl_lat=15.0, bl_lon=30.0):
    if not np.isfinite([bl_lat, bl_lon]).all() or bl_lat <= 0 or bl_lon <= 0:
        raise ValueError('Block widths must be finite and positive.')
    longitude_blocks = int(np.ceil(360 / bl_lon))
    return (np.floor((GLAT + 90) / bl_lat) * longitude_blocks + np.floor((GLON + 180) / bl_lon)).astype(int).ravel()

def block_bootstrap_corr(a, b, mask=None, n_boot=1000, seed=0, bl_lat=15.0, bl_lon=30.0):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    w = W.ravel().copy()
    if not isinstance(n_boot, (int, np.integer)) or isinstance(n_boot, (bool, np.bool_)) or n_boot < 1:
        raise ValueError('At least one bootstrap replicate is required.')
    point = wcorr(a, b, w, mask)
    if not np.isfinite(point):
        raise ValueError('Bootstrap correlation requires nonconstant paired observations.')
    if mask is not None:
        w = w * np.asarray(mask).ravel()
    ids = block_ids(bl_lat, bl_lon)
    ub = np.unique(ids[w > 0])
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(n_boot):
        pick = rng.choice(ub, size=len(ub), replace=True)
        cnt = np.zeros(ids.max() + 1)
        np.add.at(cnt, pick, 1)
        wb = w * cnt[ids]
        if wb.sum() == 0:
            continue
        try:
            correlation = wcorr(a, b, wb)
        except ValueError:
            continue
        if np.isfinite(correlation):
            est.append(correlation)
    est = np.array(est)
    if not est.size:
        raise ValueError('No bootstrap draw produced a defined correlation.')
    return (point, np.percentile(est, 2.5), np.percentile(est, 97.5))

def fmt_ci(t):
    return f'{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]'

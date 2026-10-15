"""Follow-up analyses for the 2026-10-13 review (requires torch and the pinned model stack).

Subcommands (default: all):
  replay      re-evaluate cached fixed-grid profiles at a few sites (loader consistency)
  relu        activation-region crossings at the Jacobian sites (GeoCLIP, SINR)
  common      all 13 checkpoints at a common 0.3 km probe radius on the fixed grid
  land        Table 2 statistics restricted to land sites
  bearings    16/32/64-bearing sensitivity at 200 sites per checkpoint

Outputs: outputs/summaries/review_followups.json (merged across subcommands),
outputs/profiles/common_radius_0p3.npz, outputs/tables/review_followups.tex.
Usage: python scripts/review_followups.py [--models MODELS_DIR] [subcommand ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get('ETISSOT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src'))
from etissot.geometry.fitting import fit_many, fit_ellipse  # noqa: E402
from etissot.geometry.probing import probe_points, R_EARTH  # noqa: E402
from etissot.geometry.references import equal_earth_metric  # noqa: E402
from etissot.statistics.spatial import GLAT, GLON, wcorr, wquantile  # noqa: E402
from etissot.models import loaders  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
B32 = np.arange(32) * 2 * np.pi / 32
NAMES = ['GeoCLIP', 'SatCLIP vit16l10', 'SatCLIP r18l10', 'SatCLIP r50l10', 'SatCLIP vit16l40', 'SatCLIP r18l40',
         'SatCLIP r50l40', 'SINR cap1000', 'SINR cap10', 'SINR distilled', 'CSP iNat2018', 'CSP fMoW', 'Climplicit']
SHORT = {'GeoCLIP': 'GeoCLIP', 'SatCLIP vit16l10': 'SatCLIP ViT-L10', 'SatCLIP r18l10': 'SatCLIP R18-L10',
         'SatCLIP r50l10': 'SatCLIP R50-L10', 'SatCLIP vit16l40': 'SatCLIP ViT-L40', 'SatCLIP r18l40': 'SatCLIP R18-L40',
         'SatCLIP r50l40': 'SatCLIP R50-L40', 'SINR cap1000': 'SINR cap-1000', 'SINR cap10': 'SINR cap-10',
         'SINR distilled': 'SINR distilled', 'CSP iNat2018': 'CSP-iNat', 'CSP fMoW': 'CSP-fMoW', 'Climplicit': 'Climplicit'}
RADIUS = {n: 1.0 for n in NAMES}
RADIUS.update({'SINR cap1000': 0.3, 'SINR cap10': 0.3, 'SINR distilled': 0.3, 'CSP iNat2018': 0.1, 'CSP fMoW': 0.1, 'Climplicit': 0.1})
JITTER = {'SatCLIP vit16l40': 'SatCLIP vit16l40', 'SINR cap1000': 'SINR cap1000 (raw)', 'SatCLIP r18l40': 'SatCLIP r18l40',
          'SatCLIP r50l40': 'SatCLIP r50l40', 'SINR distilled': 'SINR distilled-env (raw)', 'CSP iNat2018': 'CSP iNat2018 grid (norm)',
          'CSP fMoW': 'CSP fMoW grid (norm)', 'Climplicit': 'Climplicit 4-month (raw)'}
CHART = ('GeoCLIP', 'CSP iNat2018', 'CSP fMoW', 'Climplicit')
OUT = ROOT / 'outputs/summaries/review_followups.json'


def load_results():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_results(key, value):
    res = load_results()
    res[key] = value
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))


def rings(lat_deg, lon_deg, r, n=32):
    """(N*(n+1),) lat, lon in radians: centre then n bearings for each site."""
    la, lo = [], []
    for a, b in zip(np.radians(lat_deg), np.radians(lon_deg)):
        x, y, _ = probe_points(a, b, r, n)
        la.append(x)
        lo.append(y)
    return np.concatenate(la), np.concatenate(lo)


def evaluate(fn, la, lo, batch=4096):
    out = []
    for s in range(0, len(la), batch):
        with torch.no_grad():
            y = fn(torch.tensor(la[s:s + batch], dtype=torch.float64), torch.tensor(lo[s:s + batch], dtype=torch.float64))
        out.append(y.double().numpy())
    return np.concatenate(out)


def profiles(fn, lat_deg, lon_deg, r, n=32):
    la, lo = rings(lat_deg, lon_deg, r, n)
    y = evaluate(fn, la, lo).reshape(len(lat_deg), n + 1, -1)
    return np.linalg.norm(y[:, 1:] - y[:, :1], axis=-1) / r


def domain_mask(name, lat, lon, r):
    if name not in CHART:
        return np.ones(len(lat), bool)
    clearance = R_EARTH * np.arcsin(np.clip(abs(np.cos(np.radians(lat)) * np.sin(np.radians(180 - abs(lon)))), 0, 1))
    return ~((abs(lon) > 90) & (clearance <= r))


def load_models(models):
    return loaders.load_all(models, ROOT / 'data/reference/summaries/checkpoint_manifest.json')


# ------------------------------------------------------------------ replay
def replay(enc):
    g = np.load(ROOT / 'data/profiles/global.npz')
    idx = np.array([400, 1500, 2484, 3600])
    out = {}
    for name in NAMES:
        p = profiles(enc[name], GLAT.ravel()[idx], GLON.ravel()[idx], RADIUS[name])
        c = g['prof__' + name][idx]
        out[name] = float(np.max(abs(p / c - 1)))
        print('replay', name, out[name], flush=True)
    save_results('replay_max_relative_profile_difference', out)


# ------------------------------------------------------------------ relu
def relu_models(models):
    geo_path = Path(models) / 'encoders/geoclip/location_encoder_weights.pth'
    if not geo_path.exists():
        geo_path = loaders.package_root('geoclip') / 'model/weights/location_encoder_weights.pth'
    geo = loaders.load_geoclip_checkpoint(geo_path)
    sm = loaders.filemodule('audit_sinr_relu', ROOT / 'src/etissot/models/vendor/sinr/models.py')
    sinr = {}
    for name, tail in [('SINR cap1000', 'hard_cap_num_per_class_1000'), ('SINR cap10', 'hard_cap_num_per_class_10'),
                       ('SINR distilled', 'distilled_from_env')]:
        c = torch.load(Path(models) / f'encoders/sinr/pretrained_models/model_an_full_input_enc_sin_cos_{tail}.pt',
                       map_location='cpu', weights_only=False)
        p = c['params']
        m = sm.ResidualFCNet(p['input_dim'], p['num_classes'], p['num_filts'], p['depth']).double().eval()
        m.load_state_dict(c['state_dict'])
        for mod in m.modules():
            if isinstance(mod, torch.nn.ReLU):
                mod.inplace = False
        sinr[name] = m
    return geo, sinr


def geo_input(la, lo):
    return torch.stack([torch.rad2deg(la), (torch.rad2deg(lo) + 180) % 360 - 180], 1)


def sinr_input(la, lo):
    x = torch.stack([((torch.rad2deg(lo) + 180) % 360 - 180) / 180, torch.rad2deg(la) / 90], 1)
    return torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x)], 1)


def crossing(model, make_input, la, lo, n, groups=None):
    """Per site: does any probe point's ReLU pattern differ from the centre's? Optionally per module group."""
    pats = []
    mods = [m for m in model.modules() if isinstance(m, torch.nn.ReLU)]
    hooks = [m.register_forward_pre_hook(lambda mod, inp: pats.append((inp[0] > 0).reshape(inp[0].shape[0], -1).clone())) for m in mods]
    res = []
    per_group = []
    flips = []
    for s in range(0, len(la), 33 * 120):
        pats.clear()
        with torch.no_grad():
            model(make_input(torch.tensor(la[s:s + 33 * 120], dtype=torch.float64), torch.tensor(lo[s:s + 33 * 120], dtype=torch.float64)))
        P = [p.reshape(-1, n + 1, p.shape[1]) for p in pats]
        unit = [(p[:, 1:] != p[:, :1]).any(dim=1) for p in P]
        diff = [u.any(dim=1) for u in unit]
        flips.append(torch.cat(unit, 1).sum(1).numpy())
        res.append(torch.stack(diff, 1).any(1).numpy())
        if groups is not None:
            per_group.append(np.stack([torch.stack([diff[i] for i in g], 1).any(1).numpy() for g in groups], 1))
    for h in hooks:
        h.remove()
    return np.concatenate(res), (np.concatenate(per_group) if groups is not None else None), np.concatenate(flips)


def sinr_jacobian_A(model, lat_deg, lon_deg):
    la = torch.tensor(np.radians(lat_deg), dtype=torch.float64)
    lo = torch.tensor(np.radians(lon_deg), dtype=torch.float64)
    f = lambda a, b: model(sinr_input(a, b), return_feats=True)
    _, dla = torch.func.jvp(f, (la, lo), (torch.ones_like(la), torch.zeros_like(lo)))
    _, dlo = torch.func.jvp(f, (la, lo), (torch.zeros_like(la), torch.ones_like(lo)))
    J = torch.stack([dla / R_EARTH, dlo / (R_EARTH * torch.cos(la))[:, None]], -1).detach().numpy()
    w = np.linalg.eigvalsh(np.einsum('ndi,ndj->nij', J, J))
    return np.sqrt(w[:, 1] / w[:, 0])


def relu(models, only_sinr=False):
    geo, sinr = relu_models(models)
    jac = np.load(ROOT / 'data/geometry/jacobians.npz')
    g = np.load(ROOT / 'data/profiles/global.npz')
    sel = np.arange(0, GLAT.size, 2)
    sel = sel[abs(GLAT.ravel()[sel]) <= 80]
    lat, lon = GLAT.ravel()[sel], GLON.ravel()[sel]
    w = np.cos(np.radians(lat))
    out = {}

    def q(x, m, p):
        return float(wquantile(x[m], p, w[m])) if m.any() else None

    def rank(x):
        r = np.empty(len(x))
        r[np.argsort(x, kind='stable')] = np.arange(len(x))
        return r

    def summary(err, cross, aj, flips):
        high = aj >= wquantile(aj, 0.9, w)
        top = err >= wquantile(err, 0.9, w)
        t1, t2 = np.quantile(flips, [1 / 3, 2 / 3])
        terc = [flips <= t1, (flips > t1) & (flips <= t2), flips > t2]
        d = dict(n=int(len(err)), crossing_fraction=float(np.average(cross, weights=w)),
                 median_error_crossing=q(err, cross, 0.5), median_error_within=q(err, ~cross, 0.5),
                 p95_error_crossing=q(err, cross, 0.95), p95_error_within=q(err, ~cross, 0.95),
                 crossing_fraction_top_error_decile=float(np.average(cross[top], weights=w[top])),
                 crossing_fraction_high_A_decile=float(np.average(cross[high], weights=w[high])),
                 median_flipped_units=float(np.median(flips)),
                 spearman_flips_error=float(np.corrcoef(rank(flips), rank(err))[0, 1]),
                 median_error_by_flip_tercile=[q(err, m, 0.5) for m in terc],
                 p95_error_by_flip_tercile=[q(err, m, 0.95) for m in terc],
                 median_flips_top_error_decile=float(np.median(flips[top])), median_flips_other=float(np.median(flips[~top])),
                 crossing_fraction_error_above_0p1pct=float(cross[err > 1e-3].mean()) if (err > 1e-3).any() else None,
                 crossing_fraction_error_above_1pct=float(cross[err > 1e-2].mean()) if (err > 1e-2).any() else None,
                 n_error_above_1pct=int((err > 1e-2).sum()))
        return d

    # GeoCLIP at 1 km (cached profiles) and 0.3 km (cached multiradius profiles)
    aj = jac['AJ__GeoCLIP']
    small = np.load(ROOT / 'data/profiles/geoclip_multiradius.npz')
    caps = [[i for i, m in enumerate([m for m in geo.modules() if isinstance(m, torch.nn.ReLU)]) if i // 3 == k] for k in range(3)]
    for r, prof in ([] if only_sinr else [(1.0, g['prof__GeoCLIP'][sel]), (0.3, small['prof_0.3'][sel])]):
        err = abs(fit_many(prof, B32)['A_fit'] / aj - 1)
        la, lo = rings(lat, lon, r)
        cross, per, flips = crossing(geo, geo_input, la, lo, 32, groups=caps)
        d = summary(err, cross, aj, flips)
        d['crossing_fraction_by_branch_sigma_1_16_256'] = [float(np.average(per[:, k], weights=w)) for k in range(3)]
        out[f'GeoCLIP {r:g} km'] = d
        print('GeoCLIP', r, json.dumps(d), flush=True)
    # SINR at 0.3 km; Jacobians recomputed (cap1000 also checked against the cache)
    for name, m in sinr.items():
        if name == 'SINR cap1000':
            aj = jac['AJ__SINR cap1000']
            aj_new = sinr_jacobian_A(m, lat, lon)
            check = float(np.max(abs(aj_new / aj - 1)))
            prof = g['prof__' + name][sel]
        else:
            aj = sinr_jacobian_A(m, lat, lon)
            check = None
            prof = g['prof__' + name][sel]
        err = abs(fit_many(prof, B32)['A_fit'] / aj - 1)
        la, lo = rings(lat, lon, 0.3)
        cross, _, flips = crossing(m, sinr_input, la, lo, 32)
        d = summary(err, cross, aj, flips)
        d['jacobian_cache_max_relative_difference'] = check
        out[name] = d
        print(name, json.dumps(d), flush=True)
    if only_sinr:
        prev = load_results().get('relu_crossing', {})
        prev.update(out)
        out = prev
    save_results('relu_crossing', out)


# ------------------------------------------------------------------ common radius
def common(enc, r=0.3):
    lat, lon = GLAT.ravel(), GLON.ravel()
    store = {}
    out = {}
    for name in NAMES:
        p = profiles(enc[name], lat, lon, r)
        store['prof__' + name] = p
        f = fit_many(p, B32)
        dom = domain_mask(name, lat, lon, r)
        m = (abs(lat) <= 85) & dom
        w = np.cos(np.radians(lat))
        out[name] = dict(median=float(wquantile(f['A_fit'][m], 0.5, w[m])), p95=float(wquantile(f['A_fit'][m], 0.95, w[m])),
                         fallback=int((f['fallback'] & dom).sum()))
        print('common', name, out[name], flush=True)
    dest = ROOT / 'outputs/profiles'
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'common_radius_0p3.npz', **store)
    save_results('common_radius_0p3_fixed_grid', out)


# ------------------------------------------------------------------ land
def land_lookup(lat, lon):
    c = np.load(ROOT / 'data/reference/maps/coastlines.npz')
    i = np.clip(np.rint((lat - c['lat'][0]) / (c['lat'][1] - c['lat'][0])).astype(int), 0, len(c['lat']) - 1)
    j = np.clip(np.rint((lon - c['lon'][0]) / (c['lon'][1] - c['lon'][0])).astype(int), 0, len(c['lon']) - 1)
    return c['land'][i, j] > 0.5


def stats(name, prof, inp, lat, lon, keep):
    fit = fit_many(prof, B32)
    if name == 'GeoCLIP':
        ain, thin = equal_earth_metric(lat, lon)
    elif name.startswith('SatCLIP'):
        fi = fit_many(inp, B32)
        ain, thin = fi['A_fit'], fi['theta_fit']
    else:
        sec = 1 / np.cos(np.radians(lat))
        if name.startswith('CSP'):
            ain, thin = sec, np.full(len(lat), np.pi / 2)
        else:
            ain, thin = np.maximum(2 / sec, sec / 2), np.where(abs(lat) < 60, 0, np.pi / 2)
    dom = domain_mask(name, lat, lon, RADIUS[name]) & keep
    w = np.cos(np.radians(lat))
    mask = (abs(lat) <= 60) & dom
    m = (abs(lat) <= 85) & dom
    c = np.cos(2 * (fit['theta_fit'] - thin))
    valid = (ain > 1 + 1e-6) & (fit['A_fit'] > 1 + 1e-6)
    bm = mask & valid
    return dict(median=float(wquantile(fit['A_fit'][m], 0.5, w[m])), corr60=float(wcorr(np.log(fit['A_fit']), np.log(ain), w * mask)),
                bearing=float(np.average(c[bm], weights=w[bm])), n60=int(mask.sum()), n85=int(m.sum()))


def land():
    g = np.load(ROOT / 'data/profiles/global.npz')
    j = dict(np.load(ROOT / 'data/profiles/randomized_primary.npz'))
    j.update(dict(np.load(ROOT / 'data/profiles/randomized_satclip.npz')))
    inputs = np.load(ROOT / 'data/profiles/input_references.npz')
    out = {}
    for name in NAMES:
        if name in JITTER:
            P, I, LA, LO = [], [], [], []
            for seed in [0, 1]:
                key = f'{JITTER[name]}__{seed}'
                la, lo = np.degrees(j['lat__' + key]), np.degrees(j['lon__' + key])
                P.append(j['prof__' + key])
                I.append(j['profin__' + key] if name.startswith('SatCLIP') else np.ones((len(la), 32)))
                LA.append(la)
                LO.append(lo)
            prof, inp, lat, lon = map(np.concatenate, (P, I, LA, LO))
        else:
            lat, lon = GLAT.ravel(), GLON.ravel()
            prof = g['prof__' + name]
            iname = ('SatCLIP vit16l10' if 'l10' in name else 'SatCLIP vit16l40') if name.startswith('SatCLIP') else name
            inp = inputs['in__' + iname] if name.startswith('SatCLIP') else np.ones((len(lat), 32))
        isl = land_lookup(lat, lon)
        out[name] = dict(all=stats(name, prof, inp, lat, lon, np.ones(len(lat), bool)), land=stats(name, prof, inp, lat, lon, isl),
                         land_area_fraction_85=float(np.average(isl[abs(lat) <= 85], weights=np.cos(np.radians(lat[abs(lat) <= 85])))))
        print('land', name, json.dumps(out[name]), flush=True)
    save_results('land_only', out)


# ------------------------------------------------------------------ bearings
def bearings(enc, n_sites=200):
    rng = np.random.default_rng(20261016)
    z = rng.uniform(-np.sin(np.radians(60)), np.sin(np.radians(60)), n_sites)
    lat = np.degrees(np.arcsin(z))
    lon = rng.uniform(-170, 170, n_sites)
    out = {}
    B64 = np.arange(64) * 2 * np.pi / 64
    for name in NAMES:
        p64 = profiles(enc[name], lat, lon, RADIUS[name], n=64)
        f64 = fit_many(p64, B64)
        f32 = fit_many(p64[:, ::2], B64[::2])
        f16 = fit_many(p64[:, ::4], B64[::4])
        ok = (f32['A_fit'] > 1.05) & ~f32['fallback'] & ~f16['fallback'] & ~f64['fallback']

        def dbear(a, b):
            d = abs(np.degrees(a - b)) % 180
            return np.minimum(d, 180 - d)
        out[name] = dict(
            median_rel_A_16_vs_32=float(np.median(abs(f16['A_fit'] / f32['A_fit'] - 1))),
            p95_rel_A_16_vs_32=float(np.quantile(abs(f16['A_fit'] / f32['A_fit'] - 1), 0.95)),
            median_rel_A_64_vs_32=float(np.median(abs(f64['A_fit'] / f32['A_fit'] - 1))),
            p95_rel_A_64_vs_32=float(np.quantile(abs(f64['A_fit'] / f32['A_fit'] - 1), 0.95)),
            p95_bearing_deg_16_vs_32=float(np.quantile(dbear(f16['theta_fit'], f32['theta_fit'])[ok], 0.95)) if ok.any() else None,
            p95_bearing_deg_64_vs_32=float(np.quantile(dbear(f64['theta_fit'], f32['theta_fit'])[ok], 0.95)) if ok.any() else None,
            fallbacks_16_32_64=[int(f16['fallback'].sum()), int(f32['fallback'].sum()), int(f64['fallback'].sum())])
        print('bearings', name, json.dumps(out[name]), flush=True)
    save_results('bearing_count', out)


# ------------------------------------------------------------------ LaTeX
def num(v, fmt='{:.2f}'):
    t = fmt.format(v)
    return '$-$' + t[1:] if t.startswith('-') else t


def pct(v):
    if v is None:
        return '---'
    v = 100 * v
    if v == 0:
        return '0'
    if v < 0.01:
        e = int(np.floor(np.log10(v)))
        return f'${v / 10 ** e:.1f}\\times10^{{{e}}}$'
    return f'{v:.2f}' if v < 10 else f'{v:.1f}'


def tables():
    res = load_results()
    files = {}
    L = []
    if 'relu_crossing' in res:
        R = res['relu_crossing']
        L += [r'\begin{table}[htbp]\centering\small',
              r'\caption{Activation-region crossings at the 2,340 Jacobian sites. A probe disc crosses when any probe point has a different ReLU activation pattern from the center. Errors are area-weighted $|A_r/A_J-1|$ in percent.}',
              r'\label{tab:relu}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
              r'&&\multicolumn{2}{c}{Median error (\%)}&\multicolumn{2}{c}{95th percentile (\%)}\\',
              r'Checkpoint, radius&Crossing discs&Within&Crossing&Within&Crossing\\\midrule']
        for key, lab in [('SINR cap1000', 'SINR cap-1000, 0.3 km'), ('SINR cap10', 'SINR cap-10, 0.3 km'), ('SINR distilled', 'SINR distilled, 0.3 km')]:
            d = R[key]
            L.append(f"{lab}&{100 * d['crossing_fraction']:.1f}\\%&{pct(d['median_error_within'])}&{pct(d['median_error_crossing'])}&{pct(d['p95_error_within'])}&{pct(d['p95_error_crossing'])}\\\\")
        L += [r'\midrule', r'&Median switching units&\multicolumn{2}{c}{Lowest / highest switching tercile}&\multicolumn{2}{c}{Lowest / highest switching tercile}\\']
        for key, lab in [('GeoCLIP 1 km', 'GeoCLIP, 1 km'), ('GeoCLIP 0.3 km', 'GeoCLIP, 0.3 km')]:
            d = R[key]
            m, q = d['median_error_by_flip_tercile'], d['p95_error_by_flip_tercile']
            L.append(f"{lab}&{d['median_flipped_units']:.0f}&\\multicolumn{{2}}{{c}}{{{pct(m[0])} / {pct(m[2])}}}&\\multicolumn{{2}}{{c}}{{{pct(q[0])} / {pct(q[2])}}}\\\\")
        g1, g3 = R['GeoCLIP 1 km'], R['GeoCLIP 0.3 km']
        L += [r'\bottomrule\end{tabular}', r'\end{adjustbox}',
              r'\par\smallskip\parbox{.97\textwidth}{\footnotesize SINR Jacobians are double-precision automatic derivatives; for cap-1000 they reproduce the cached values to $'
              + f"{R['SINR cap1000']['jacobian_cache_max_relative_difference']:.0e}".replace('e-', r'\times10^{-').replace('e+', r'\times10^{') + r'}$. For all three SINR checkpoints, every site with error above 0.1\% is a crossing disc. Every GeoCLIP disc crosses at least one of its 9,216 units at both radii; the rows therefore compare terciles of the number of units whose pattern changes within the disc (Spearman correlation with error '
              + f"{g1['spearman_flips_error']:.2f} at 1 km, {g3['spearman_flips_error']:.2f} at 0.3 km). Crossing fractions by branch ($\\sigma=1,16,256$) are "
              + ', '.join(f'{100 * v:.0f}' for v in g1['crossing_fraction_by_branch_sigma_1_16_256']) + r'\% at 1 km and '
              + ', '.join(f'{100 * v:.0f}' for v in g3['crossing_fraction_by_branch_sigma_1_16_256']) + r'\% at 0.3 km.}',
              r'\end{table}']
        files['review_relu'], L = L, []
    if 'land_only' in res:
        R = res['land_only']
        L += [r'\begin{table}[htbp]\centering\small',
              r'\caption{Table~\ref{tab:enc} statistics over all sites and over land sites only. Land is defined by the 0.5$^\circ$ land mask used for the map coastlines; land covers '
              + f"{100 * R['GeoCLIP']['land_area_fraction_85']:.0f}" + r'\% of the area within $85^\circ$. Same samples, weights and exclusions as Table~\ref{tab:enc}.}',
              r'\label{tab:land}', r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',
              r'&\multicolumn{2}{c}{Median $A$}&\multicolumn{2}{c}{$r_A$}&\multicolumn{2}{c}{$C_\theta$}\\',
              r'Checkpoint&All&Land&All&Land&All&Land\\\midrule']
        for n in NAMES:
            a, b = R[n]['all'], R[n]['land']
            L.append(f"{SHORT[n]}&{num(a['median'])}&{num(b['median'])}&{num(a['corr60'])}&{num(b['corr60'])}&{num(a['bearing'])}&{num(b['bearing'])}\\\\")
        L += [r'\bottomrule\end{tabular}', r'\end{table}']
        files['review_land'], L = L, []
    if 'bearing_count' in res:
        R = res['bearing_count']
        L += [r'\begin{table}[htbp]\centering\small',
              r'\caption{Sensitivity to the number of bearings at 200 sites per checkpoint within $60^\circ$ (seed 20261016), each checkpoint at its primary radius. Profiles with 64 bearings are fitted in full and subsampled to 32 and 16. Entries are 95th percentiles over sites of the relative change in anisotropy (percent) and of the bearing change (degrees, sites with $A>1.05$), relative to 32 bearings.}',
              r'\label{tab:bearingcount}', r'\begin{tabular}{@{}lrrrr@{}}\toprule',
              r'&\multicolumn{2}{c}{$A$ change (\%)}&\multicolumn{2}{c}{Bearing change ($^\circ$)}\\',
              r'Checkpoint&16 bearings&64 bearings&16 bearings&64 bearings\\\midrule']
        for n in NAMES:
            d = R[n]
            sm = lambda v: '$<$0.01' if v < 0.005 else f'{v:.2f}'
            L.append(f"{SHORT[n]}&{sm(100 * d['p95_rel_A_16_vs_32'])}&{sm(100 * d['p95_rel_A_64_vs_32'])}&{sm(d['p95_bearing_deg_16_vs_32'])}&{sm(d['p95_bearing_deg_64_vs_32'])}\\\\")
        L += [r'\bottomrule\end{tabular}', r'\end{table}']
        files['review_bearings'], L = L, []
    dest = ROOT / 'outputs/tables'
    dest.mkdir(parents=True, exist_ok=True)
    for name, lines in files.items():
        (dest / f'{name}.tex').write_text('\n'.join(lines) + '\n')
        print('wrote', dest / f'{name}.tex')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=str(ROOT / 'models'))
    ap.add_argument('tasks', nargs='*', default=['replay', 'relu', 'common', 'land', 'bearings', 'tables'])
    a = ap.parse_args()
    enc = None
    for t in a.tasks:
        if t in ('replay', 'common', 'bearings') and enc is None:
            enc, _ = load_models(a.models)
        if t == 'replay':
            replay(enc)
        elif t == 'relu':
            relu(a.models)
        elif t == 'relu_sinr':
            relu(a.models, only_sinr=True)
        elif t == 'common':
            common(enc)
        elif t == 'land':
            land()
        elif t == 'bearings':
            bearings(enc)
        elif t == 'tables':
            tables()


if __name__ == '__main__':
    main()

"""Geographic input-reference, sampling and Jacobian-tail diagnostics."""
from etissot.geometry.references import shapes, equal_earth_metric
from pathlib import Path
import json
import numpy as np
from etissot.geometry.fitting import fit_many
from etissot.statistics.spatial import GLAT, GLON, wcorr, wquantile
B = np.arange(32) * 2 * np.pi / 32
NAMES = ['GeoCLIP', 'SatCLIP vit16l10', 'SatCLIP r18l10', 'SatCLIP r50l10', 'SatCLIP vit16l40', 'SatCLIP r18l40', 'SatCLIP r50l40', 'SINR cap1000', 'SINR cap10', 'SINR distilled', 'CSP iNat2018', 'CSP fMoW', 'Climplicit']
JITTER = {'SatCLIP vit16l40': 'SatCLIP vit16l40', 'SINR cap1000': 'SINR cap1000 (raw)', 'SatCLIP r18l40': 'SatCLIP r18l40', 'SatCLIP r50l40': 'SatCLIP r50l40', 'SINR distilled': 'SINR distilled-env (raw)', 'CSP iNat2018': 'CSP iNat2018 grid (norm)', 'CSP fMoW': 'CSP fMoW grid (norm)', 'Climplicit': 'Climplicit 4-month (raw)'}

def boot(x, y, c, lat, lon, block, n=1000, domain=None):
    mask = (abs(lat) <= 60) & (np.ones(len(lat), bool) if domain is None else domain)
    w = np.cos(np.radians(lat)) * mask
    ids = np.floor((lat + 90) / block).astype(int) * 100 + np.floor((lon + 180) / (2 * block)).astype(int)
    unique = np.unique(ids[mask])
    rng = np.random.default_rng(16)
    rs = []
    cs = []
    for _ in range(n):
        draw = rng.choice(unique, len(unique))
        counts = np.bincount(draw, minlength=ids.max() + 1)
        weights = w * counts[ids]
        rs.append(wcorr(x, y, weights))
        valid = np.isfinite(c)
        cs.append(np.average(c[valid], weights=weights[valid]))
    return (np.percentile(rs, [2.5, 97.5]).tolist(), np.percentile(cs, [2.5, 97.5]).tolist())

def run(workspace):
    """Run this analysis with explicit paths; return outputs in workspace.output."""
    ROOT = workspace.data
    P = workspace.output
    d = np.load(ROOT / 'profiles/global.npz')
    j = dict(np.load(ROOT / 'profiles/randomized_primary.npz'))
    j.update(dict(np.load(ROOT / 'profiles/randomized_satclip.npz')))
    inputs = np.load(ROOT / 'profiles/input_references.npz')
    jac = np.load(ROOT / 'geometry/jacobians.npz')
    rows = {}
    fixed = {}
    phase = {}
    tails = {}

    def summarise(name, profile, inp, lat, lon, primary):
        fit = fit_many(profile, B)
        fi = fit_many(inp, B) if inp.ndim == 2 else None
        if name == 'GeoCLIP':
            ain, thin = equal_earth_metric(lat, lon)
        elif name.startswith('SatCLIP'):
            ain = fi['A_fit']
            thin = fi['theta_fit']
        else:
            sec = 1 / np.cos(np.radians(lat))
            if name.startswith('CSP'):
                ain = sec
                thin = np.full(len(lat), np.pi / 2)
            else:
                ain = np.maximum(2 / sec, sec / 2)
                thin = np.where(abs(lat) < 60, 0, np.pi / 2)
        w = np.cos(np.radians(lat))
        domain = np.ones(len(lat), bool)
        if name == 'GeoCLIP' or name.startswith('CSP') or name == 'Climplicit':
            radius = 1.0 if name == 'GeoCLIP' else 0.1
            clearance = 6371 * np.arcsin(np.clip(abs(np.cos(np.radians(lat)) * np.sin(np.radians(180 - abs(lon)))), 0, 1))
            domain = ~((abs(lon) > 90) & (clearance <= radius))
        mask = (abs(lat) <= 60) & domain
        c = np.cos(2 * (fit['theta_fit'] - thin))
        x = np.log(fit['A_fit'])
        y = np.log(ain)
        valid_direction = (ain > 1 + 1e-06) & (fit['A_fit'] > 1 + 1e-06)
        c = np.where(valid_direction, c, np.nan)
        bearing_mask = mask & valid_direction
        block = 30 if name == 'GeoCLIP' or 'l10' in name or name == 'SINR cap10' else 15
        ci, bc = boot(x, y, c, lat, lon, block, domain=domain)
        m = (abs(lat) <= 85) & domain
        result = {'median': float(wquantile(fit['A_fit'][m], 0.5, w[m])), 'p95': float(wquantile(fit['A_fit'][m], 0.95, w[m])), 'corr60': float(wcorr(x, y, w * mask)), 'corr85': float(wcorr(x, y, w * m)), 'bearing': float(np.average(c[bearing_mask], weights=w[bearing_mask])), 'corr_ci': ci, 'bearing_ci': bc, 'n': int(mask.sum()), 'sample': primary, 'fallback': int((fit['fallback'] & domain).sum()), 'seam_excluded': int((~domain).sum()), 'bearing_n': int(bearing_mask.sum()), 'bearing_area_retained': float(w[bearing_mask].sum() / w[mask].sum()), 'bearing_sensitivity': {str(threshold): {'mean': float(np.average(c[mask & (ain > threshold) & (fit['A_fit'] > threshold)], weights=w[mask & (ain > threshold) & (fit['A_fit'] > threshold)])), 'area_retained': float(w[mask & (ain > threshold) & (fit['A_fit'] > threshold)].sum() / w[mask].sum())} for threshold in [1.01, 1.05]}, 'corr60_no_fallback': float(wcorr(x, y, w * mask * ~fit['fallback'])), 'residual_minor_p95': float(wquantile(fit['residual_minor'][m], 0.95, w[m]))}
        return (result, fit)
    for name in NAMES:
        lat, lon = (GLAT.ravel(), GLON.ravel())
        input_name = ('SatCLIP vit16l10' if 'l10' in name else 'SatCLIP vit16l40') if name.startswith('SatCLIP') else 'SINR cap1000' if name.startswith('SINR') else name
        ref = inputs['in__' + input_name]
        result, fit = summarise(name, d['prof__' + name], ref, lat, lon, 'fixed')
        fixed[name] = result
        if name in JITTER:
            tag = JITTER[name]
            profiles = []
            lats = []
            lons = []
            ins = []
            phase[name] = []
            for seed in [0, 1]:
                key = f'{tag}__{seed}'
                la = np.degrees(j['lat__' + key])
                lo = np.degrees(j['lon__' + key])
                p = j['prof__' + key]
                inp = j['profin__' + key] if name.startswith('SatCLIP') else np.ones(len(la))
                r, _ = summarise(name, p, inp, la, lo, f'jitter {seed}')
                phase[name].append(r)
                profiles.append(p)
                lats.append(la)
                lons.append(lo)
                ins.append(inp)
            result, _ = summarise(name, np.concatenate(profiles), np.concatenate(ins), np.concatenate(lats), np.concatenate(lons), 'two randomized grids')
        rows[name] = result
        print(name, result, flush=True)
        if 'AJ__' + name in jac:
            sel = np.arange(0, GLAT.size, 2)
            sel = sel[abs(GLAT.ravel()[sel]) <= 80]
            aj = jac['AJ__' + name]
            ar = fit['A_fit'][sel]
            w = np.cos(np.radians(GLAT.ravel()[sel]))
            error = abs(ar / aj - 1)
            high = aj >= wquantile(aj, 0.9, w)
            tails[name] = {'n': len(sel), 'median_error': float(wquantile(error, 0.5, w)), 'p95_error': float(wquantile(error, 0.95, w)), 'high_A_p95_error': float(wquantile(error[high], 0.95, w[high])), 'fallback_n': int(fit['fallback'][sel].sum()), 'fallback_max_error': float(error[fit['fallback'][sel]].max()) if fit['fallback'][sel].any() else None}
            mask60 = abs(GLAT.ravel()[sel]) <= 60
            tails[name]['within60_p95_error'] = float(wquantile(error[mask60], 0.95, w[mask60]))
    fg = fit_many(d['prof__GeoCLIP'], B)
    fi = fit_many(inputs['in__GeoCLIP'], B)
    ea, et = equal_earth_metric(GLAT.ravel(), GLON.ravel())
    g = shapes(fg['A_fit'], fg['theta_fit'])
    h = shapes(ea, et)
    lam, v = np.linalg.eigh(h)
    invs = v * (lam ** (-0.5))[:, None, :] @ v.transpose(0, 2, 1)
    eig = np.linalg.eigvalsh(invs @ g @ invs)
    relative = np.sqrt(eig[:, 1] / eig[:, 0])
    mask = abs(GLAT.ravel()) <= 60
    w = np.cos(np.radians(GLAT.ravel()[mask]))
    shape = {'median_relative_anisotropy': float(wquantile(relative[mask], 0.5, w)), 'p95_relative_anisotropy': float(wquantile(relative[mask], 0.95, w))}
    x = np.log(fg['A_fit']).reshape(GLAT.shape)
    y = np.log(ea).reshape(GLAT.shape)
    xr = (x - x.mean(axis=1, keepdims=True)).ravel()
    yr = (y - y.mean(axis=1, keepdims=True)).ravel()
    shape['dezonal_corr'] = float(wcorr(xr, yr, np.cos(np.radians(GLAT.ravel())) * mask))
    radius_check = {}
    small = np.load(ROOT / 'profiles/geoclip_multiradius.npz')
    sel = np.arange(0, GLAT.size, 2)
    sel = sel[abs(GLAT.ravel()[sel]) <= 80]
    aj = jac['AJ__GeoCLIP']
    mask60 = abs(GLAT.ravel()[sel]) <= 60
    ww = np.cos(np.radians(GLAT.ravel()[sel]))
    for r in [0.3, 0.5, 1.0]:
        fit = fit_many(small[f'prof_{r}'][sel], B)
        err = abs(fit['A_fit'] / aj - 1)
        radius_check[str(r)] = {'median_error60': float(wquantile(err[mask60], 0.5, ww[mask60])), 'p95_error60': float(wquantile(err[mask60], 0.95, ww[mask60])), 'fallback60': int(fit['fallback'][mask60].sum())}
    report = {'primary': rows, 'fixed_grid': fixed, 'randomized_phases': phase, 'jacobian_tails': tails, 'geoclip_shape': shape, 'geoclip_radius': radius_check}
    (P / 'summaries/geometry.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Tails', tails, 'GeoCLIP shape', shape)

"""Validate covariance, gradients, analytic geometry and stored fit decisions."""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from etissot.prediction.gaussian_process import GaussianProcess as GP, T
from etissot.geometry.fitting import fit_ellipse, fit_many, refine_radius
from etissot.geometry.references import equal_earth_metric
from pyproj import Proj
from pathlib import Path
import json

def run(workspace):
    """Run this analysis using explicit input and output directories."""
    if not __debug__:
        raise RuntimeError('Numerical validation requires Python without optimization (-O).')
    ROOT = workspace.data
    P = workspace.output
    rng = np.random.default_rng(16)
    torch.set_num_threads(2)
    x = rng.normal(size=(18, 2)) * 100
    y = rng.normal(size=18)
    a = rng.uniform(1, 4, 18)
    b = rng.uniform(0, np.pi, 18)
    e = rng.normal(size=(18, 7))
    max_gradient_error = 0.0
    max_kernel_error = 0.0
    for kind in ['iso', 'embedding', 'SatCLIP-L40', 'spatial+embedding', 'tensor+embedding']:
        gp = GP(x, y, a, b, e if 'embedding' in kind else None, kind)
        p = [np.log(150), 0.2, -2.0]
        if gp.tensor:
            p += [0.7]
        if gp.is_combo:
            p += [np.log(3), -0.4]
        p = np.array(p)
        value, grad = gp.objective(p)
        numeric = np.array([(gp.objective(p + np.eye(len(p))[i] * 1e-05)[0] - gp.objective(p - np.eye(len(p))[i] * 1e-05)[0]) / 2e-05 for i in range(len(p))])
        err = np.max(abs(grad - numeric))
        max_gradient_error = max(max_gradient_error, float(err))
        assert err < 1e-06, (kind, grad, numeric)
        k = gp.kernel(T(p), gp.dx, gp.dy, gp.a, gp.b, gp.a, gp.b, gp.ed).numpy()
        expected = np.empty_like(k)
        for i in range(18):
            for j in range(18):
                if kind == 'embedding':
                    d = np.linalg.norm(e[i] - e[j])
                    fac = 1.0
                elif gp.tensor:

                    def sigma(aa, bb):
                        u = np.array([np.sin(bb), np.cos(bb)])
                        v = np.array([np.cos(bb), -np.sin(bb)])
                        return aa ** (-p[3]) * np.outer(u, u) + aa ** p[3] * np.outer(v, v)
                    s = (sigma(a[i], b[i]) + sigma(a[j], b[j])) / 2
                    d = np.sqrt((x[i] - x[j]) @ np.linalg.solve(s, x[i] - x[j]))
                    fac = np.linalg.det(s) ** (-0.5)
                else:
                    d = np.linalg.norm(x[i] - x[j])
                    fac = 1.0
                z = np.sqrt(3) * d / np.exp(p[0])
                expected[i, j] = fac * np.exp(p[1]) * (1 + z) * np.exp(-z)
                if gp.is_combo:
                    z = np.sqrt(3) * np.linalg.norm(e[i] - e[j]) / np.exp(p[-2])
                    expected[i, j] += np.exp(p[-1]) * (1 + z) * np.exp(-z)
        err = np.max(abs(k - expected))
        max_kernel_error = max(max_kernel_error, float(err))
        assert err < 1e-12
        assert np.linalg.eigvalsh(k).min() > -1e-10
        if gp.tensor and (not gp.is_combo):
            p[3] = 0
            iso = GP(x, y, a, b, None, 'iso')
            assert abs(gp.objective(p)[0] - iso.objective(p[:3])[0]) < 1e-10
    print('Covariance max absolute error:', max_kernel_error)
    print('Autograd vs central difference max absolute error:', max_gradient_error)
    print('Positive definiteness and beta=0 nesting checks passed.')
    bear = np.arange(32) * 2 * np.pi / 32
    profiles = []
    for _ in range(500):
        aa = rng.uniform(1.01, 50)
        theta = rng.uniform(0, np.pi)
        scale = 10 ** rng.uniform(-3, 3)
        profile = scale * np.sqrt(aa ** 2 * np.cos(bear - theta) ** 2 + np.sin(bear - theta) ** 2)
        f = fit_ellipse(profile, bear)
        assert not f['fallback']
        assert abs(f['A_fit'] / aa - 1) < 1e-10
        profiles.append(profile)
    fm = fit_many(profiles, bear)
    for i, profile in enumerate(profiles):
        f = fit_ellipse(profile, bear)
        for name in ['lambda1', 'lambda2', 'residual_minor']:
            assert fm[name][i] == f[name]
    proj = Proj(proj='eqearth', R=1)
    max_metric_error = 0.0
    for _ in range(100):
        lat = rng.uniform(-80, 80)
        lon = rng.uniform(-170, 170)
        eps = 0.0001
        n = (np.array(proj(lon, lat + eps)) - np.array(proj(lon, lat - eps))) / (2 * np.radians(eps))
        e = (np.array(proj(lon + eps, lat)) - np.array(proj(lon - eps, lat))) / (2 * np.radians(eps) * np.cos(np.radians(lat)))
        sv = np.linalg.svd(np.column_stack([n, e]), compute_uv=False)
        exact = equal_earth_metric(np.array([lat]), np.array([lon]))[0][0]
        err = abs(sv[0] / sv[1] / exact - 1)
        max_metric_error = max(max_metric_error, err)
        assert err < 1e-08
    print('500 exact ellipses and fit_many diagnostics passed.')
    print('100 analytic Equal Earth comparisons; maximum relative error:', max_metric_error)
    records = {}
    for path in workspace.fit_files():
        records.update(json.loads(path.read_text()))
    assert len(records) == 360
    combined = 0
    for key, fit in records.items():
        assert len(fit['starts']) == 3
        assert abs(fit['nll'] - min((x['nll'] for x in fit['starts']))) < 1e-09
        if key.endswith('|tensor+embedding'):
            base = records[key.rsplit('|', 1)[0] + '|spatial+embedding']
            assert np.allclose(fit['starts'][1]['start'], np.insert(base['p'], 3, 0), rtol=0, atol=1e-12)
            assert abs(fit['starts'][1]['initial_nll'] - base['nll']) < 1e-07
            assert fit['nll'] <= base['nll'] + 1e-07
            combined += 1
    assert combined == 36
    report = json.loads((P / 'summaries/released_audit.json').read_text())
    assert len(report['metadata']) == 13 and len(report['radius_cases']) == 50
    for case in report['radius_cases']:
        ps = case['profiles']
        rr = refine_radius(lambda radius: np.array(next((p['profile'] for p in ps if p['radius_km'] == radius))), case['start_radius_km'], bear)
        assert rr['converged'] == case['converged'] and rr['radius'] == case['radius_km']
        assert len(rr['trail']) == len(case['trail'])
        if case['seam_clearance_km'] is not None:
            assert case['start_radius_km'] < case['seam_clearance_km']
    assert sum((c['converged'] for c in report['radius_cases'])) == 34
    geom = json.loads((P / 'summaries/geometry.json').read_text())
    assert sum((x['seam_excluded'] for x in geom['primary'].values())) == 3
    for n, x in geom['primary'].items():
        assert x['bearing_n'] <= x['n']
        assert x['bearing_sensitivity']['1.05']['area_retained'] <= x['bearing_sensitivity']['1.01']['area_retained'] <= x['bearing_area_retained'] + 1e-12
    print('360 selections, 36 combined nesting candidates, 50 radius trails, 13 bearing masks and 3 seam exclusions verified.')

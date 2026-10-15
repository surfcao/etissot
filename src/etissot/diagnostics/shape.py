"""Matched finite-lag and Jacobian shape comparison."""
from pathlib import Path
import sys, json
import numpy as np
from etissot.geometry.fitting import fit_many
from etissot.statistics.spatial import GLAT, GLON, wquantile, wcorr
from etissot.geometry.references import shapes, equal_earth_metric

def run(workspace):
    """Run this analysis using explicit input and output directories."""
    ROOT = workspace.data
    P = workspace.output
    B = np.arange(32) * 2 * np.pi / 32
    D = np.load(ROOT / 'profiles/global.npz')
    J = np.load(ROOT / 'geometry/jacobians.npz')
    sel = np.arange(0, GLAT.size, 2)
    sel = sel[abs(GLAT.ravel()[sel]) <= 80]
    mask = abs(GLAT.ravel()[sel]) <= 60
    lat = GLAT.ravel()[sel][mask]
    lon = GLON.ravel()[sel][mask]
    w = np.cos(np.radians(lat))
    f = fit_many(D['prof__GeoCLIP'][sel][mask], B)
    a, t = equal_earth_metric(lat, lon)
    g = shapes(f['A_fit'], f['theta_fit'])
    h = shapes(a, t)
    j = shapes(J['AJ__GeoCLIP'][mask], J['THJ__GeoCLIP'][mask])

    def distance(g, h):
        l, v = np.linalg.eigh(h)
        u = v * l[:, None, :] ** (-0.5) @ v.transpose(0, 2, 1)
        eig = np.linalg.eigvalsh(u @ g @ u)
        return np.sqrt(eig[:, 1] / eig[:, 0])

    def summary(x):
        return {'median': float(wquantile(x, 0.5, w)), 'p95': float(wquantile(x, 0.95, w))}
    report = {'geoclip_shape_same_sites': {'n': len(w), 'finite_vs_reference': summary(distance(g, h)), 'jacobian_vs_reference': summary(distance(j, h)), 'finite_vs_jacobian': summary(distance(g, j)), 'isotropic_vs_reference': summary(distance(np.broadcast_to(np.eye(2), h.shape), h)), 'jacobian_input_corr': float(wcorr(np.log(J['AJ__GeoCLIP'][mask]), np.log(a), w)), 'finite_input_corr': float(wcorr(np.log(f['A_fit']), np.log(a), w))}}
    (P / 'summaries/shape.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

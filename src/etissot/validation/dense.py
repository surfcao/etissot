"""Validate saved dense profiles and unchanged prediction baselines."""
from pathlib import Path
import json, numpy as np
from etissot.geometry.fitting import fit_many

def compare_dense_fit(anisotropy, bearing, fallback, fit):
    # LAPACK differences are far below reported precision; axes are modulo pi.
    np.testing.assert_allclose(np.ravel(anisotropy), fit['A_fit'], rtol=1e-10, atol=1e-12)
    delta = np.asarray(bearing).ravel() - fit['theta_fit']
    axis_error = 0.5 * np.arctan2(np.sin(2 * delta), np.cos(2 * delta))
    np.testing.assert_allclose(axis_error, 0, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(np.ravel(fallback), fit['fallback'])

def run(workspace):
    """Run this analysis using explicit input and output directories."""
    if not __debug__:
        raise RuntimeError('Numerical validation requires Python without optimization (-O).')
    ROOT = workspace.data
    P = workspace.output
    D = np.load(ROOT / 'profiles/dense_regional.npz')
    B = np.arange(32) * 2 * np.pi / 32
    count = 0
    for key in D.files:
        if not key.startswith('prof__'):
            continue
        tag = key[6:]
        p = D[key]
        assert p.shape[1] == 32 and np.isfinite(p).all() and (p >= 0).all()
        f = fit_many(p, B)
        a = D['A__' + tag]
        theta = D['B__' + tag]
        assert a.shape == (len(D['lat__' + tag]), len(D['lon__' + tag])) and (a >= 1).all()
        compare_dense_fit(a, theta, D['fallback__' + tag], f)
        count += len(p)
    assert count == 135165
    old = json.loads((P / 'baselines/kriging_before_display_update.json').read_text())
    new = json.loads((P / 'baselines/kriging_before_downstream_update.json').read_text())
    assert old == new
    stats = json.loads((P / 'baselines/radius_audit.json').read_text())
    assert stats['total_converged'] == 34 and stats['total_tested'] == 50
    geo = next((r for r in stats['radius_by_checkpoint'] if r['encoder'] == 'GeoCLIP'))
    assert abs(geo['max_accepted_tensor_error_pct'] - 0.7799526109224139) < 1e-12
    print('135,165 dense profiles, all fitted maps and fallback flags verified. Primary GP results unchanged. Tensor-error reporting verified.')

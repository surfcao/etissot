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
    # Compare the audit summary regenerated in this run with the archived baseline;
    # the GP results themselves are compared in validation.downstream.
    regenerated = json.loads((P / 'summaries/audit_summary.json').read_text())
    baseline = json.loads((workspace.data / 'reference/baselines/radius_audit.json').read_text())  # data folder, not outputs
    assert regenerated['total_converged'] == baseline['total_converged'] == 34
    assert regenerated['total_tested'] == baseline['total_tested'] == 50
    by_name = {r['encoder']: r for r in baseline['radius_by_checkpoint']}
    for row in regenerated['radius_by_checkpoint']:
        old = by_name[row['encoder']]
        assert (row['tested'], row['converged']) == (old['tested'], old['converged']), row['encoder']
        for key in ('max_accepted_A_error_pct', 'max_accepted_tensor_error_pct', 'max_accepted_shape_discrepancy'):
            assert (row[key] is None) == (old[key] is None), (row['encoder'], key)
            if row[key] is not None:
                assert abs(row[key] - old[key]) <= 1e-9 * max(1.0, abs(old[key])), (row['encoder'], key)
    print('135,165 dense profiles, all fitted maps and fallback flags verified. Regenerated refinement audit matches its baseline.')

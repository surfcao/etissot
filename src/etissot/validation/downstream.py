"""Validate independent downstream points and bootstrap intervals."""
from pathlib import Path
import json
import re
import numpy as np

def run(workspace):
    """Run this analysis using explicit input and output directories."""
    if not __debug__:
        raise RuntimeError('Numerical validation requires Python without optimization (-O).')
    ROOT = workspace.data
    P = workspace.output
    d = json.loads((P / 'summaries/downstream.json').read_text())
    old = json.loads((P / 'baselines/downstream_points.json').read_text())
    assert len(d['rows']) == len(old) == 12
    for r, o in zip(d['rows'], old):
        assert (r['model'], r['space'], r['metric']) == (o['model'], o['space'], o['metric'])
        np.testing.assert_allclose([r['raw'], r['partial']], [o['current']['raw'], o['current']['partial']], atol=1e-14, rtol=0)
        reps = np.asarray(r['replicates'])
        assert reps.shape == (500, 2) and np.isfinite(reps).all()
        np.testing.assert_array_equal(np.percentile(reps, [2.5, 97.5], axis=0), np.array([r['raw_ci'], r['partial_ci']]).T)
    assert 0.152 < d['max_partial'] < 0.153
    assert 0.613 < d['global_feature_head_logA_correlation']['cap1000'] < 0.615
    assert json.loads((P / 'summaries/kriging.json').read_text()) == json.loads((P / 'baselines/kriging_before_downstream_update.json').read_text())
    text = (P / 'tables/encoder_table.tex').read_text()
    assert re.search(r'unnormali[sz]ed(?: numerical)? harmonic-input profiles at 1 km', text)
    print('All 12 downstream points match independent recomputation; 6,000 paired bootstrap replicates and intervals verified. Primary GP results unchanged.')

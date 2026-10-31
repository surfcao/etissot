"""Grid-scale smoothness of the fixed-grid anisotropy fields (supplementary Section S3).

For each checkpoint, the Pearson correlation of log anisotropy between adjacent nodes of the
fixed 2.5 x 5 degree grid is computed within 60 degrees latitude, separately for neighbours
along longitude (same row, excluding the pair across the antimeridian) and along latitude
(same column). A field that varies smoothly at grid scale gives a high lag-one correlation;
a field that varies within a grid cell gives a low one.

The checkpoints summarized on two pooled randomized grids (diagnostics.geographic.JITTER)
are compared with those summarized on the fixed grid, and the two groups are required to
separate on both correlations. Writes outputs/summaries/grid_smoothness.json.
"""
import json

import numpy as np

from etissot.diagnostics.geographic import JITTER, NAMES
from etissot.geometry.fitting import fit_many
from etissot.statistics.spatial import GLAT

BEARINGS = np.arange(32) * 2 * np.pi / 32


def lag_one(log_a, rows):
    """Lag-one correlations of a (lat, lon) field along longitude and latitude within the given rows."""
    field = log_a[rows]
    along_lon = np.corrcoef(field[:, :-1].ravel(), field[:, 1:].ravel())[0, 1]
    index = np.flatnonzero(rows)
    adjacent = index[:-1][np.diff(index) == 1]
    along_lat = np.corrcoef(log_a[adjacent].ravel(), log_a[adjacent + 1].ravel())[0, 1]
    return float(along_lon), float(along_lat)


def summarize(global_profiles):
    rows = np.abs(GLAT[:, 0]) <= 60
    out = {}
    for name in NAMES:
        log_a = np.log(fit_many(global_profiles['prof__' + name], BEARINGS)['A_fit'].reshape(GLAT.shape))
        lon, lat = lag_one(log_a, rows)
        out[name] = dict(randomized=name in JITTER, lag_one_longitude=lon, lag_one_latitude=lat)

    def span(randomized, key):
        values = [v[key] for v in out.values() if v['randomized'] == randomized]
        return [min(values), max(values)]

    groups = {g: {k: span(g == 'randomized', k) for k in ('lag_one_longitude', 'lag_one_latitude')}
              for g in ('randomized', 'fixed')}
    separated = all(groups['randomized'][k][1] < groups['fixed'][k][0] for k in groups['fixed'])
    if not separated:
        raise AssertionError('randomized-grid and fixed-grid checkpoints do not separate on lag-one correlation')
    return dict(latitude_limit=60, checkpoints=out, groups=groups, separated=separated)


def run(workspace):
    with np.load(workspace.data / 'profiles/global.npz') as g:
        result = summarize(g)
    (workspace.output / 'summaries').mkdir(parents=True, exist_ok=True)
    (workspace.output / 'summaries/grid_smoothness.json').write_text(json.dumps(result, indent=1) + '\n')

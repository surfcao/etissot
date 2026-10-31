"""Opposite-direction asymmetry of the fixed-grid profiles (supplementary Section S1.1).

For each checkpoint, fits the 32-bearing fixed-grid profiles at the primary radius and
summarizes the relative first-harmonic amplitude sqrt(a1^2 + b1^2)/a0 and its share of
the residual: on the equispaced grid the first harmonic contributes (a1^2 + b1^2)/2 to
the mean squared residual, so its share is (asym^2/2)/nu^2 at each site. Medians are
area-weighted (cos latitude) over all fixed-grid centers (|latitude| <= 85 degrees); no
fixed-grid probe disc crosses the antimeridian.
"""
import json

import numpy as np

from etissot.geometry.fitting import fit_many
from etissot.statistics.spatial import GLAT, wmedian, wquantile

BEARINGS = np.arange(32) * 2 * np.pi / 32


def summarize(profiles):
    weights = np.cos(np.radians(GLAT.ravel()))
    out = {}
    for key in profiles.files:
        if not key.startswith('prof__') or key.startswith('prof__ref'):
            continue
        name = key[len('prof__'):]
        fit = fit_many(profiles[key], BEARINGS)
        asym, nu = fit['asym'], fit['ne']
        share = np.where(nu > 0, 0.5 * asym ** 2 / np.maximum(nu, 1e-300) ** 2, np.nan)
        ok = np.isfinite(asym) & np.isfinite(share)
        out[name] = dict(median_asymmetry=float(wmedian(asym[ok], weights[ok])),
                         p95_asymmetry=float(wquantile(asym[ok], 0.95, weights[ok])),
                         median_residual=float(wmedian(nu[ok], weights[ok])),
                         median_first_harmonic_share=float(wmedian(share[ok], weights[ok])),
                         sites=int(ok.sum()))
    medians = {k: v['median_asymmetry'] for k, v in out.items()}
    return dict(checkpoints=out,
                min_median_asymmetry=min(medians.values()), min_checkpoint=min(medians, key=medians.get),
                max_median_asymmetry=max(medians.values()), max_checkpoint=max(medians, key=medians.get),
                min_median_first_harmonic_share=min(v['median_first_harmonic_share'] for v in out.values()))


def run(workspace):
    with np.load(workspace.data / 'profiles/global.npz') as profiles:
        result = summarize(profiles)
    (workspace.output / 'summaries').mkdir(parents=True, exist_ok=True)
    (workspace.output / 'summaries/asymmetry.json').write_text(json.dumps(result, indent=1) + '\n')

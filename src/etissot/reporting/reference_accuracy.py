"""Agreement of fitted and analytic input-reference anisotropy (main Section 4.1, supplementary Figure S1).

Fits the 32-bearing profiles of the input references on the fixed 2.5 x 5 degree grid
(|latitude| <= 85 degrees) and compares the fitted anisotropy with its analytic value at
every site:

  grid      equally scaled longitude-latitude sinusoids, sec(lat)            global.npz 'prof__ref grid'
  cart3d    three-dimensional Cartesian coordinates, A = 1                     global.npz 'prof__ref cart3d'
  wrap      SINR 2:1 wrap, max(2 cos lat, sec(lat)/2)                          global.npz 'prof__ref SINR-input'
  equal     Equal Earth coordinates, Tissot axis ratio of the projection       input_references.npz 'in__GeoCLIP'
  sh_l10    SatCLIP harmonics as implemented (m = 0 scaled by pi), L = 10      input_references.npz 'in__SatCLIP vit16l10'
  sh_l40    the same for L = 40; the implementation degrades numerically       input_references.npz 'in__SatCLIP vit16l40'

The relative difference |A_fit/A_analytic - 1| is summarized per reference within 60 degrees
and through 85 degrees. Aggregates exclude sh_l40, whose departure reflects the implemented
basis rather than the estimator. Writes outputs/summaries/reference_accuracy.json.
"""
import json

import numpy as np

from etissot.geometry.fitting import fit_many
from etissot.geometry.references import equal_earth_metric
from etissot.statistics.spatial import GLAT, GLON

BEARINGS = np.arange(32) * 2 * np.pi / 32


def _legendre(l, x):
    c = np.zeros(l + 1)
    c[l] = 1
    return np.polynomial.legendre.legval(x, c)


def scaled_harmonic_anisotropy(degree, lat_deg, c=np.pi):
    """Analytic A for real harmonics of degree < L with m = 0 components scaled by c (S1.5)."""
    theta = np.radians(np.asarray(lat_deg, dtype=float)) + np.pi / 2
    x, sn = np.cos(theta), np.sin(theta)
    z = np.zeros_like(x)
    s_iso = sum((2 * l + 1) * l * (l + 1) / (8 * np.pi) for l in range(degree))
    for l in range(1, degree):
        dp = -l * (x * _legendre(l, x) - _legendre(l - 1, x)) / sn
        z += (2 * l + 1) / (4 * np.pi) * dp ** 2
    q = 1 + (c ** 2 - 1) * z / s_iso
    return np.sqrt(np.maximum(q, 1 / q))


def references(global_profiles, input_profiles):
    lat, lon = GLAT.ravel(), GLON.ravel()
    phi = np.radians(lat)
    return {
        'grid': (global_profiles['prof__ref grid'], 1 / np.cos(phi), float(global_profiles['r__ref grid'])),
        'cart3d': (global_profiles['prof__ref cart3d'], np.ones_like(phi), float(global_profiles['r__ref cart3d'])),
        'wrap': (global_profiles['prof__ref SINR-input'], np.maximum(2 * np.cos(phi), 1 / (2 * np.cos(phi))),
                 float(global_profiles['r__ref SINR-input'])),
        'equal': (input_profiles['in__GeoCLIP'], equal_earth_metric(lat, lon)[0], 1.0),
        'sh_l10': (input_profiles['in__SatCLIP vit16l10'], scaled_harmonic_anisotropy(10, lat), 1.0),
        'sh_l40': (input_profiles['in__SatCLIP vit16l40'], scaled_harmonic_anisotropy(40, lat), 1.0),
    }


def summarize(global_profiles, input_profiles):
    lat = GLAT.ravel()
    within60 = np.abs(lat) <= 60
    out = {}
    for name, (profiles, analytic, radius) in references(global_profiles, input_profiles).items():
        fit = fit_many(profiles, BEARINGS)
        rel = np.abs(fit['A_fit'] / analytic - 1)
        worst = int(np.argmax(rel))
        out[name] = dict(radius_km=radius, sites=int(rel.size),
                         max_relative_error=float(rel.max()), latitude_of_max=float(lat[worst]),
                         analytic_at_max=float(analytic[worst]),
                         max_relative_error_60=float(rel[within60].max()),
                         median_relative_error=float(np.median(rel)),
                         fallbacks=int(fit['fallback'].sum()))
    kept = [k for k in out if k != 'sh_l40']
    worst = max(kept, key=lambda k: out[k]['max_relative_error'])
    return dict(references=out,
                max_relative_error_60_excluding_l40=max(out[k]['max_relative_error_60'] for k in kept),
                max_relative_error_85_excluding_l40=out[worst]['max_relative_error'],
                reference_of_max_85=worst, latitude_of_max_85=out[worst]['latitude_of_max'],
                analytic_at_max_85=out[worst]['analytic_at_max'],
                l40_max_relative_error=out['sh_l40']['max_relative_error'])


def run(workspace):
    with np.load(workspace.data / 'profiles/global.npz') as g, np.load(workspace.data / 'profiles/input_references.npz') as i:
        result = summarize(g, i)
    # Analytic values quoted in the text: the latitude band where the CSP sec(phi)
    # reference stays below 1.05, and Equal Earth anisotropy at 60N on the central
    # meridian and at the antimeridian.
    result['csp_reference_below_1p05_deg'] = float(np.degrees(np.arccos(1 / 1.05)))
    a, _ = equal_earth_metric(np.array([60.0, 60.0]), np.array([0.0, 180.0]))
    result['equal_earth_60N'] = dict(central_meridian=float(a[0]), antimeridian=float(a[1]))
    (workspace.output / 'summaries').mkdir(parents=True, exist_ok=True)
    (workspace.output / 'summaries/reference_accuracy.json').write_text(json.dumps(result, indent=1) + '\n')

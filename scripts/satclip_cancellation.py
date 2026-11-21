"""Numerical cancellation in the SatCLIP spherical-harmonic input (supplementary Section S6).

The released SatCLIP code evaluates each spherical harmonic from an expanded
polynomial in cos(theta), with SatCLIP's colatitude theta = latitude + 90 degrees
and the zonal (m = 0) functions multiplied by pi. Near the poles, the terms of the
high-degree polynomials are many orders of magnitude larger than their sum, so the
small relative errors in the stored coefficients and in double-precision arithmetic
become large absolute errors in the basis functions. This script reports:

  1. the zonal harmonic of degree 39 at 77.5 N: the largest polynomial term, the
     value computed by the code, the value of the stored polynomial in exact
     rational arithmetic and the SciPy reference, which separate the error due
     to coefficient rounding from the error due to floating-point arithmetic;
  2. the largest absolute error over all orders m of the basis functions of
     degrees 30-39 against SciPy references (with the same pi scaling of the
     zonal functions) between 70 and 85 degrees latitude;
  3. the fitted anisotropy of the L = 40 input at 77.5 N and a 1 km probe radius
     (32 bearings), computed from the code and from SciPy references, with and
     without the pi scaling, against the analytic value of the scaled input
     (etissot.diagnostics.harmonics.zonal_anisotropy) and 1 for the unscaled input.

SciPy evaluates the harmonics by recurrence, which avoids the large intermediate
terms. No checkpoint is needed. Writes outputs/measurements/review/satclip_cancellation.json.
Usage: python scripts/satclip_cancellation.py   (requires torch and SciPy 1.15 or later)
"""
import inspect
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from scipy.special import sph_harm_y

ROOT = Path(os.environ.get('ETISSOT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'src'))
from etissot.models.loaders import filemodule  # noqa: E402
from etissot.geometry.probing import probe_points  # noqa: E402
from etissot.geometry.fitting import fit_ellipse  # noqa: E402
from etissot.diagnostics.harmonics import zonal_anisotropy  # noqa: E402

YLM = filemodule('satclip_ylm', ROOT / 'src/etissot/models/vendor/satclip/satclip/positional_encoding/spherical_harmonics_ylm.py')
DEGREES = 40


def released(m, degree, lat_deg, lon_deg):
    """Basis function as computed by the released code (float64)."""
    theta = torch.tensor(np.radians(np.asarray(lat_deg, dtype=float) + 90.0), dtype=torch.float64)
    phi = torch.tensor(np.radians(np.asarray(lon_deg, dtype=float) + 180.0), dtype=torch.float64)
    value = YLM.SH(m, degree, phi, theta)
    return (value * torch.ones_like(theta)).numpy() if torch.is_tensor(value) else value * np.ones(theta.shape)


def reference(m, degree, lat_deg, lon_deg, zonal_multiplier=np.pi):
    """The same real basis function from SciPy, with the code's zonal scaling."""
    theta = np.radians(np.asarray(lat_deg, dtype=float) + 90.0)
    phi = np.radians(np.asarray(lon_deg, dtype=float) + 180.0)
    y = sph_harm_y(degree, abs(m), theta, phi)
    if m == 0:
        return zonal_multiplier * y.real
    return np.sqrt(2) * (-1) ** m * (y.real if m > 0 else y.imag)


def zonal_39(lat_deg=77.5):
    source = inspect.getsource(YLM.Yl39_m0).split('return', 1)[1]
    terms = [(s, c, k or '1') for s, c, k in
             re.findall(r'([+-]?)([0-9.]+)\*cos\(theta\)(?:\*\*(\d+))?', source.replace(' ', ''))]
    x = np.cos(np.radians(lat_deg + 90.0))
    exact_x = Fraction(x)
    largest = max(float(c) * abs(x) ** int(k) for _, c, k in terms)
    exact = sum((-1 if s == '-' else 1) * Fraction(c) * exact_x ** int(k) for s, c, k in terms)
    computed = float(released(0, 39, lat_deg, 0.0))
    stored = float(exact)
    true = float(reference(0, 39, lat_deg, 0.0))
    return dict(latitude=lat_deg, number_of_terms=len(terms), largest_term=largest,
                computed_by_code=computed, stored_polynomial_exact=stored, scipy_reference=true,
                coefficient_rounding_error=stored - true, arithmetic_error=computed - stored,
                note='values include the pi scaling of the zonal functions, as computed by the code')


def basis_errors(latitudes=np.arange(70.0, 85.0001, 2.5), longitude=0.0):
    out, by_latitude = {}, {}
    for degree in range(30, DEGREES):
        worst, rows = 0.0, np.zeros(len(latitudes))
        for m in range(-degree, degree + 1):
            actual = released(m, degree, latitudes, np.full_like(latitudes, longitude))
            expected = reference(m, degree, latitudes, np.full_like(latitudes, longitude))
            if m != 0:  # the code's sign convention for nonzonal functions differs from SciPy's
                expected = abs(expected) * np.sign(actual + 1e-300)
            worst = max(worst, float(np.max(abs(actual - expected))))
            rows = np.maximum(rows, abs(actual - expected))
        out[str(degree)] = worst
        by_latitude[str(degree)] = rows.tolist()
    return dict(latitudes=latitudes.tolist(), longitude=longitude, maximum_absolute_error_by_degree=out,
                maximum_absolute_error_by_degree_and_latitude=by_latitude)


def features(lats, lons, source, zonal_multiplier):
    columns = []
    for degree in range(DEGREES):
        for m in range(-degree, degree + 1):
            if source == 'code':
                y = released(m, degree, lats, lons)
                if m == 0:
                    y = y * zonal_multiplier / np.pi
            else:
                y = reference(m, degree, lats, lons, zonal_multiplier)
            columns.append(y)
    return np.stack(columns, axis=-1)


def input_anisotropy(lat_deg=77.5, lon_deg=0.0, radius_km=1.0):
    la, lo, bearings = probe_points(np.radians(lat_deg), np.radians(lon_deg), radius_km)
    la, lo = np.degrees(la), np.degrees(lo)
    rows = {}
    for label, multiplier, analytic in [('pi_scaled', np.pi, float(zonal_anisotropy(np.array([lat_deg]), DEGREES)[0])),
                                        ('unscaled', 1.0, 1.0)]:
        row = dict(analytic=analytic)
        for source in ('code', 'scipy'):
            f = features(la, lo, source, multiplier)
            profile = np.linalg.norm(f[1:] - f[0], axis=1) / radius_km
            a = fit_ellipse(profile, bearings)['A_fit']
            row[source] = dict(A_fit=a, departure_percent=100 * (a / analytic - 1))
        rows[label] = row
    return dict(latitude=lat_deg, longitude=lon_deg, radius_km=radius_km, n_directions=len(bearings), **rows)


def main():
    report = dict(zonal_degree_39=zonal_39(), basis_errors=basis_errors(), input_anisotropy=input_anisotropy())
    out = ROOT / 'outputs/measurements/review'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'satclip_cancellation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

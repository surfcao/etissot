"""Radius screening: directional stretch as a function of probe radius.

For each released checkpoint, evaluates the encoder on 32-bearing geodesic rings at
log-spaced radii around four fixed-grid sites and records the directional stretch
s_r(theta) = ||f(p_r(theta)) - f(p)|| / r. In the regime where a local linear
approximation holds, s_r does not depend on r; a decrease at large r indicates
that feature differences level off, and an increase or scatter at small r
indicates floating-point rounding. The primary radius of each family should lie
in the flat part of the curve. At the primary radius the profile is also compared
with the cached global profile at the same site, which checks that this
environment reproduces the study measurements.

Requires the `study` and `models` extras and the checkpoints (etissot models download).
Usage: python scripts/radius_screening.py --models ./models --data ./data --output ./outputs [--only NAME ...]

Writes outputs/measurements/review/radius_screening.json, the collected input that
the pipeline reads from data/measurements/review/. With --only, existing entries
for other checkpoints are kept; `last_run` and each entry's `collected_utc` show
which were collected when.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from etissot.geometry.probing import destination_point
from etissot.geometry.fitting import fit_ellipse
from etissot.statistics.spatial import LATS, LONS
from etissot.models.loaders import load_all

SITES = [(0.0, 22.5), (30.0, 77.5), (45.0, -97.5), (-30.0, -57.5)]  # fixed-grid nodes on land, away from the antimeridian
RADII = np.unique(np.round(np.concatenate([np.geomspace(1e-3, 30.0, 31), [0.1, 0.3, 1.0]]), 10))
BEARINGS = np.linspace(0, 2 * np.pi, 32, endpoint=False)


def profiles(fn, lat, lon):
    la0, lo0 = np.radians(lat), np.radians(lon)
    las, los = [la0], [lo0]
    for r in RADII:
        a, b = destination_point(la0, lo0, BEARINGS, r)
        las.extend(a); los.extend(b)
    with torch.no_grad():
        out = fn(torch.tensor(las, dtype=torch.float64), torch.tensor(los, dtype=torch.float64))
    emb = out.detach().double().numpy()
    centre, ring = emb[0], emb[1:].reshape(len(RADII), len(BEARINGS), -1)
    return np.linalg.norm(ring - centre, axis=2) / RADII[:, None]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--models', type=Path, default=Path('models'))
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('outputs'))
    p.add_argument('--only', nargs='*')
    a = p.parse_args()
    # Verify every checkpoint against the study manifest before screening.
    enc, meta = load_all(a.models, a.data / 'reference/summaries/checkpoint_manifest.json')
    cached = np.load(a.data / 'profiles/global.npz')
    # Collected measurements, not the pipeline's derived summary in outputs/summaries.
    dest = a.output / 'measurements/review/radius_screening.json'
    report = json.loads(dest.read_text()) if dest.exists() else {}
    if report and (report.get('radii_km') != RADII.tolist() or [tuple(s) for s in report.get('sites', [])] != SITES):
        raise SystemExit(f'{dest} was collected on a different radius or site grid; use another --output.')
    report['sites'] = SITES; report['radii_km'] = RADII.tolist(); report['bearings'] = 32
    report.setdefault('checkpoints', {})
    report['last_run'] = []
    for name, fn in enc.items():
        if a.only and name not in a.only:
            continue
        t = time.time(); primary = meta[name]['radius']; rows = []
        for lat, lon in SITES:
            s = profiles(fn, lat, lon)
            fits = [fit_ellipse(row, BEARINGS) for row in s]
            i = int(np.argmin(abs(RADII - primary)))
            gi, gj = int(np.argmin(abs(LATS - lat))), int(np.argmin(abs(LONS - lon)))
            ref = cached[f'prof__{name}'].reshape(len(LATS), len(LONS), -1)[gi, gj]
            rows.append(dict(lat=lat, lon=lon, rms_stretch=np.sqrt((s ** 2).mean(1)).tolist(),
                             min_stretch=s.min(1).tolist(), max_stretch=s.max(1).tolist(),
                             anisotropy=[f['A_fit'] for f in fits], fallback=[bool(f['fallback']) for f in fits],
                             cached_profile_max_relative_difference=float(np.max(abs(s[i] / ref - 1)))))
        report['checkpoints'][name] = dict(primary_radius_km=primary, measured_vector=meta[name]['normalization'], sites=rows,
                                           collected_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        report['last_run'].append(name)
        print(f'{name}: {time.time() - t:.1f} s; max difference from cached profile at primary radius '
              f'{max(r["cached_profile_max_relative_difference"] for r in rows):.2e}', flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()

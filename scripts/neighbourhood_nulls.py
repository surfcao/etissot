"""Null and extrapolation controls for the neighbourhood comparison (review, 2026-10-13).

Reuses the cached measurements in data/measurements/neighbourhoods.npz; no encoder
is re-evaluated. For every valid (non-fallback) query--model pair it ranks the 400
candidates by several predictors and compares the selected sets with the actual
embedding neighbours (E/X overlap):

  tensor     -- the fitted 0.3 km tensor (primary predictor of the manuscript)
  isotropic  -- geographic distance (equivalent to an isotropic tensor)
  swapped    -- the fitted tensor with its principal axis rotated by 90 degrees
  random     -- the fitted anisotropy with a uniformly random principal bearing
                (200 draws per comparison, seed 20261013)
  other      -- the full fitted tensor of each other valid query of the same model

Writes outputs/summaries/neighbourhood_nulls.json and outputs/tables/neighbourhood_nulls.tex.
Usage: python scripts/neighbourhood_nulls.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from etissot.geometry.fitting import fit_ellipse  # noqa: E402

BEARINGS = np.arange(32) * 2 * np.pi / 32
FIT_INDEX = 1                     # 0.3 km fit radius (primary)
PRIMARY_RADIUS, PRIMARY_K = 1, 40  # 1 km candidate disc, k = 40
N_RANDOM = 200
# Land/ocean status of the 20 query centres from the GLOBE-based 30-arcsec land
# mask in the `global-land-mask` package (True = land); order follows `queries`.
LAND = [False, False, True, False, True, True, True, False, True, True, True, False,
        False, True, True, False, False, True, True, False]


def tensor(lam1, lam2, theta):
    u = np.array([np.cos(theta), np.sin(theta)])
    return lam2 * np.eye(2) + (lam1 - lam2) * np.outer(u, u)


def topk(x, k):
    return np.argsort(x, kind='stable')[:k]


def overlap(a, b):
    return len(np.intersect1d(a, b)) / len(a)


def predicted(delta, G):
    return np.einsum('ni,ij,nj->n', delta, G, delta)


def main():
    d = np.load(ROOT / 'data/measurements/neighbourhoods.npz')
    meta = json.loads(str(d['metadata']))
    nq = len(d['queries'])
    rng = np.random.default_rng(20261013)
    fits = {}
    for mi in range(len(meta)):
        fits[mi] = [fit_ellipse(d[f'profiles_{mi}'][FIT_INDEX, q], BEARINGS) for q in range(nq)]

    rows = []
    for mi, m in enumerate(meta):
        valid = [q for q in range(nq) if not fits[mi][q]['fallback']]
        for q in valid:
            f = fits[mi][q]
            for ri, radius in enumerate(d['radii']):
                delta = d['displacements'][ri, q]
                actual = d[f'distances_{mi}'][ri, q]
                geo = np.linalg.norm(delta, axis=1)
                for k in (10, 20, 40, 80):
                    e = topk(actual, k)
                    rec = dict(model=m['name'], query=q, land=LAND[q], radius=float(radius), k=k,
                               d_k_km=float(np.sort(geo[e])[-1]))
                    rec['tensor'] = overlap(e, topk(predicted(delta, tensor(f['lambda1'], f['lambda2'], f['theta_fit'])), k))
                    rec['isotropic'] = overlap(e, topk(geo, k))
                    rec['swapped'] = overlap(e, topk(predicted(delta, tensor(f['lambda1'], f['lambda2'], f['theta_fit'] + np.pi / 2)), k))
                    if ri == PRIMARY_RADIUS and k == PRIMARY_K:
                        rnd = [overlap(e, topk(predicted(delta, tensor(f['lambda1'], f['lambda2'], th)), k))
                               for th in rng.uniform(0, np.pi, N_RANDOM)]
                        rec['random_mean'] = float(np.mean(rnd))
                        rec['random_exact_rate'] = float(np.mean(np.isclose(rnd, 1.0)))
                        oth = [overlap(e, topk(predicted(delta, tensor(fits[mi][o]['lambda1'], fits[mi][o]['lambda2'], fits[mi][o]['theta_fit'])), k))
                               for o in valid if o != q]
                        rec['other_mean'] = float(np.mean(oth))
                        rec['other_exact_rate'] = float(np.mean(np.isclose(oth, 1.0)))
                    rows.append(rec)

    prim = [r for r in rows if r['radius'] == d['radii'][PRIMARY_RADIUS] and r['k'] == PRIMARY_K]
    n = len(prim)

    def exact(key, sub):
        return int(sum(np.isclose(r[key], 1.0) for r in sub))

    def med(key, sub):
        return float(np.median([r[key] for r in sub]))

    summary = dict(comparisons=n)
    for key in ('tensor', 'isotropic', 'swapped'):
        summary[key] = dict(exact=exact(key, prim), median=med(key, prim), mean=float(np.mean([r[key] for r in prim])))
    summary['random'] = dict(expected_exact=float(sum(r['random_exact_rate'] for r in prim)),
                             median=med('random_mean', prim), mean=float(np.mean([r['random_mean'] for r in prim])))
    summary['other'] = dict(expected_exact=float(sum(r['other_exact_rate'] for r in prim)),
                            median=med('other_mean', prim), mean=float(np.mean([r['other_mean'] for r in prim])))

    # Query-cluster bootstrap of the tensor exact-match fraction (queries resampled with replacement).
    by_q = {q: [r for r in prim if r['query'] == q] for q in range(nq)}
    boot_rng = np.random.default_rng(20261014)
    stats = []
    for _ in range(5000):
        qs = boot_rng.integers(0, nq, nq)
        sub = [r for q in qs for r in by_q[q]]
        stats.append(np.mean([np.isclose(r['tensor'], 1.0) for r in sub]))
    summary['tensor_exact_fraction'] = float(summary['tensor']['exact'] / n)
    summary['tensor_exact_fraction_query_bootstrap_95'] = [float(np.quantile(stats, .025)), float(np.quantile(stats, .975))]

    for label, flag in (('land', True), ('ocean', False)):
        sub = [r for r in prim if r['land'] == flag]
        summary[label] = dict(comparisons=len(sub), queries=int(sum(LAND) if flag else nq - sum(LAND)),
                              tensor_exact=exact('tensor', sub), isotropic_exact=exact('isotropic', sub),
                              swapped_exact=exact('swapped', sub), median_EG=med('isotropic', sub))

    extrap = []
    for ri, radius in enumerate(d['radii']):
        for k in (10, 20, 40, 80):
            sub = [r for r in rows if r['radius'] == radius and r['k'] == k]
            extrap.append(dict(radius=float(radius), k=k, comparisons=len(sub),
                               median_dk_km=float(np.median([r['d_k_km'] for r in sub])),
                               tensor_exact=exact('tensor', sub), tensor_median=med('tensor', sub),
                               isotropic_exact=exact('isotropic', sub), isotropic_median=med('isotropic', sub),
                               swapped_exact=exact('swapped', sub), swapped_median=med('swapped', sub)))
    summary['extrapolation'] = extrap

    per_query = []
    for q in range(nq):
        sub = by_q[q]
        per_query.append(dict(query=q, lat=float(np.degrees(d['queries'][q, 0])), lon=float(np.degrees(d['queries'][q, 1])),
                              land=LAND[q], comparisons=len(sub), tensor_exact=exact('tensor', sub),
                              isotropic_exact=exact('isotropic', sub), median_EG=med('isotropic', sub)))
    summary['per_query'] = per_query

    out = ROOT / 'outputs/summaries/neighbourhood_nulls.json'
    out.write_text(json.dumps(summary, indent=1))

    # LaTeX tables
    s = summary
    t = [r'\begin{table}[htbp]\centering\small',
         r'\caption{Null and ablation predictors for the primary neighborhood comparison ($k=40$ of 400 candidates within 1\,km; tensors from separate 0.3\,km probes; ' + f'{n}' + r' valid query--model comparisons). Each predictor ranks the same candidates; entries compare its 40 selections with the 40 actual embedding neighbors.}',
         r'\label{tab:neighbournulls}',
         r'\begin{tabular}{@{}lrrr@{}}\toprule',
         r'Predictor&Identical sets&Median overlap&Mean overlap\\\midrule',
         f"Fitted tensor&{s['tensor']['exact']}&{s['tensor']['median']:.3f}&{s['tensor']['mean']:.3f}\\\\",
         f"Isotropic (geographic distance)&{s['isotropic']['exact']}&{s['isotropic']['median']:.3f}&{s['isotropic']['mean']:.3f}\\\\",
         f"Principal axis rotated $90^\\circ$&{s['swapped']['exact']}&{s['swapped']['median']:.3f}&{s['swapped']['mean']:.3f}\\\\",
         f"Random principal bearing$^a$&{s['random']['expected_exact']:.1f}&{s['random']['median']:.3f}&{s['random']['mean']:.3f}\\\\",
         f"Tensor of another query$^b$&{s['other']['expected_exact']:.1f}&{s['other']['median']:.3f}&{s['other']['mean']:.3f}\\\\",
         r'\bottomrule\end{tabular}',
         r'\par\smallskip\parbox{.97\textwidth}{\footnotesize $^a$Fitted anisotropy with a uniformly random bearing; entries average 200 draws per comparison, and the identical-set count is the expected number. $^b$Full fitted tensor of each other valid query of the same checkpoint, averaged over those queries. Query-cluster bootstrap 95\% interval for the fitted-tensor identical fraction: ' + f"{s['tensor_exact_fraction_query_bootstrap_95'][0]:.3f}--{s['tensor_exact_fraction_query_bootstrap_95'][1]:.3f}" + r'.}',
         r'\end{table}', '',
         r'\begin{table}[htbp]\centering\small',
         r'\caption{Neighborhood agreement beyond the fitting scale. Candidate radius sets the disc from which 400 candidates are drawn; $\tilde d_k$ is the median, over comparisons, of the largest geographic distance from the query among the $k$ embedding neighbors. Tensors are fitted from 0.3\,km probes throughout. Entries give identical-set counts and median overlap with the actual embedding neighbors.}',
         r'\label{tab:neighbourextrap}',
         r'\begin{tabular}{@{}rrrrrrr@{}}\toprule',
         r'Disc radius (km)&$k$&$\tilde d_k$ (km)&\multicolumn{2}{c}{Fitted tensor}&\multicolumn{2}{c}{Isotropic}\\',
         r'&&&Identical&Median&Identical&Median\\\midrule']
    for e in extrap:
        if e['k'] == 40 or e['radius'] == 1.0:
            t.append(f"{e['radius']:g}&{e['k']}&{e['median_dk_km']:.2f}&{e['tensor_exact']}/{e['comparisons']}&{e['tensor_median']:.3f}&{e['isotropic_exact']}/{e['comparisons']}&{e['isotropic_median']:.3f}\\\\")
    t += [r'\bottomrule\end{tabular}', r'\end{table}']
    (ROOT / 'outputs/tables/neighbourhood_nulls.tex').write_text('\n'.join(t) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_query', 'extrapolation')}, indent=1))
    for e in extrap:
        print(e)
    for p in per_query:
        print(p)


if __name__ == '__main__':
    main()

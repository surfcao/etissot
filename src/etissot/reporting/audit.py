"""Generate radius, boundary, checkpoint and shape audit tables."""
from pathlib import Path
import json
import numpy as np
from collections import Counter

WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']


def word(n):
    """Spell small counts as in the manuscript prose."""
    return WORDS[n] if 0 <= n < len(WORDS) else f'{n:,}'


TEENS = ['eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']


def checkpoint_count(n):
    """Spell checkpoint counts up to twenty, as in the manuscript prose."""
    return TEENS[n - 11] if 11 <= n <= 20 else word(n)


def join(names):
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def seam_exclusion_sentences(excluded, randomized, fixed_grid_clearance_km, max_radius_km):
    """Prose for antimeridian exclusions, computed from the per-checkpoint counts.

    ``excluded`` maps display names to the number of excluded probe discs and
    ``randomized`` is the set of display names summarized on the randomized grids.
    """
    if not fixed_grid_clearance_km > max_radius_km:
        raise ValueError('A fixed-grid center lies within a probe radius of the antimeridian; update the boundary prose.')
    if any(excluded[n] for n in excluded if n not in randomized):
        raise ValueError('A fixed-grid checkpoint excludes seam discs; update the boundary prose.')
    hit = [n for n in excluded if excluded[n]]
    if not hit:
        tail = 'so no disc is excluded'
    elif len(set(excluded[n] for n in hit)) == 1:
        k = excluded[hit[0]]
        tail = f"so exclusions arise only on the randomized grids, where they remove {word(k)} disc{'s' if k != 1 else ''} each for {join(hit)}"
    else:
        tail = 'so exclusions arise only on the randomized grids, where they remove ' + join(
            [f"{word(excluded[n])} disc{'s' if excluded[n] != 1 else ''} for {n}" for n in hit])
    return 'Fixed-grid centers lie beyond every probe radius from the antimeridian, ' + tail + '.'


def run(workspace):
    """Run this analysis using explicit input and output directories."""
    ROOT = workspace.data
    P = workspace.output
    A = json.loads((P / 'summaries/released_audit.json').read_text())
    G = json.loads((P / 'summaries/geometry.json').read_text())
    S = json.loads((P / 'summaries/shape.json').read_text())['geoclip_shape_same_sites']
    short = {'GeoCLIP': 'GeoCLIP', 'SatCLIP vit16l10': 'SatCLIP ViT-L10', 'SatCLIP r18l10': 'SatCLIP R18-L10', 'SatCLIP r50l10': 'SatCLIP R50-L10', 'SatCLIP vit16l40': 'SatCLIP ViT-L40', 'SatCLIP r18l40': 'SatCLIP R18-L40', 'SatCLIP r50l40': 'SatCLIP R50-L40', 'SINR cap1000': 'SINR cap-1000', 'SINR cap10': 'SINR cap-10', 'SINR distilled': 'SINR distilled', 'CSP iNat2018': 'CSP-iNat', 'CSP fMoW': 'CSP-fMoW', 'Climplicit': 'Climplicit'}

    def write(name, lines):
        (P / name).write_text('\n'.join(lines) + '\n')

    def table(caption, label, columns, head, rows, foot=''):
        return ['\\begin{table}[htbp]\\centering\\small', '\\caption{' + caption + '}', '\\label{' + label + '}', '\\begin{tabular}{@{}' + columns + '@{}}\\toprule', head + '\\\\\\midrule', *[r + '\\\\' for r in rows], '\\bottomrule\\end{tabular}', '\\par\\smallskip\\parbox{.97\\textwidth}{\\footnotesize ' + foot + '}' if foot else '', '\\end{table}']
    def neg(value):
        text = f'{value:.2f}'
        return '$-$' + text[1:] if text.startswith('-') else text

    def sci(value):
        text = f'{value:.3g}'
        if 'e' not in text:
            return text
        mantissa, exponent = text.split('e')
        return f'${mantissa}\\times10^{{{int(exponent)}}}$'

    rows = []
    for n, r in G['primary'].items():
        b = r['bearing_sensitivity']
        rows.append(f"{short[n]}&{neg(r['bearing'])}&{100 * r['bearing_area_retained']:.1f}&{neg(b['1.01']['mean'])}&{100 * b['1.01']['area_retained']:.1f}&{neg(b['1.05']['mean'])}&{100 * b['1.05']['area_retained']:.1f}")
    write('tables/bearing_table.tex', table('Bearing agreement and retained area under fixed isotropy thresholds. Both input and output anisotropy must exceed the threshold.', 'tab:bearing', 'lrrrrrr', '&\\multicolumn{2}{c}{$1+10^{-6}$}&\\multicolumn{2}{c}{1.01}&\\multicolumn{2}{c}{1.05}\\\\Checkpoint&$C_\\theta$&Area (\\%)&$C_\\theta$&Area (\\%)&$C_\\theta$&Area (\\%)', rows, 'Area is relative to the admissible within-60-degree primary sample. Thresholds specify different conditional summaries; they are not significance tests.'))
    from etissot.diagnostics.geographic import JITTER
    from etissot.statistics.spatial import GLAT, GLON
    excluded = {short[n]: int(r['seam_excluded']) for n, r in G['primary'].items()}
    radii = [float(m['radius']) for m in A['metadata'].values()]
    clearance = 6371 * np.arcsin(np.abs(np.cos(np.radians(GLAT)) * np.sin(np.radians(180 - np.abs(GLON)))))
    seam_sentence = seam_exclusion_sentences(excluded, {short[n] for n in JITTER}, float(clearance.min()), max(radii))
    csp = {excluded['CSP-iNat'], excluded['CSP-fMoW']}
    if len(csp) != 1:
        raise ValueError('CSP checkpoints exclude different numbers of seam discs; update the boundary prose.')
    k_csp, k_clim, k_geo = csp.pop(), excluded['Climplicit'], excluded['GeoCLIP']
    # Only chart-dependent encoders (outputs that jump at the antimeridian) are excluded;
    # the prose below states that SINR and SatCLIP rings are retained.
    chart_dependent = {'GeoCLIP', 'CSP-iNat', 'CSP-fMoW', 'Climplicit'}
    if any(excluded[n] for n in excluded if n not in chart_dependent):
        raise ValueError('A seam-continuous checkpoint excludes discs; update the boundary prose.')
    parts = []
    if k_csp:
        parts.append(f"{word(k_csp)} excluded ring{'s' if k_csp != 1 else ''} for each CSP checkpoint")
    if k_clim:
        parts.append(f"{word(k_clim)} for Climplicit" if parts else f"{word(k_clim)} excluded ring{'s' if k_clim != 1 else ''} for Climplicit")
    grid_sentence = ('The two randomized grids contain ' + (' and '.join(parts) if parts else 'no excluded rings') + '; '
                     + ('no fixed-grid GeoCLIP ring crosses the seam.' if k_geo == 0 else f"{word(k_geo)} fixed-grid GeoCLIP ring{'s' if k_geo != 1 else ''} cross the seam."))
    # The boundary prose below names families; confirm the data still support it.
    def family(name):
        return next(f for f in ('GeoCLIP', 'SatCLIP', 'SINR', 'CSP', 'Climplicit') if name.startswith(f))
    behaviour = {}
    for n, r in A['boundaries'].items():
        at = {e: [x['distance_normalized'] for x in r['antimeridian'] if x['epsilon_degrees'] == e] for e in (1e-2, 1e-6)}
        seam = ('jump' if min(at[1e-6]) >= 1e-3 else
                'continuous' if max(at[1e-6]) < 1e-5 and max(at[1e-6]) < 1e-2 * max(at[1e-2]) else 'unclear')
        pole = 'identical' if r['pole_normalized_distance'] < 1e-12 else 'differs' if r['pole_normalized_distance'] > 1e-3 else 'unclear'
        behaviour.setdefault(family(n), set()).add((seam, pole))
    stated = {'GeoCLIP': {('jump', 'differs')}, 'CSP': {('jump', 'differs')}, 'Climplicit': {('jump', 'differs')},
              'SINR': {('continuous', 'differs')}, 'SatCLIP': {('continuous', 'identical')}}
    if behaviour != stated:
        raise ValueError(f'Boundary results no longer match the boundary prose: {behaviour}')
    rows = []
    for n, r in A['boundaries'].items():
        v = [x['distance_normalized'] for x in r['antimeridian'] if x['epsilon_degrees'] == 1e-06]
        rows.append(f"{short[n]}&{sci(min(v))}--{sci(max(v))}&{sci(r['pole_normalized_distance'])}")
    lines = ['The boundary tests examine whether equivalent geographic coordinates yield consistent feature vectors. All outputs are L2-normalized for this comparison, providing a bounded scale without changing the raw-feature quantity used in the main SINR and Climplicit analysis. At latitudes $0,45,75^\\circ$, paired longitudes $180^\\circ-\\epsilon$ and $-180^\\circ+\\epsilon$ are evaluated for $\\epsilon=10^{-2},10^{-4},10^{-6}$ degrees. A separate pole test compares $(90^\\circ,0^\\circ)$ and $(90^\\circ,-180^\\circ)$, two coordinate descriptions of the same point. The full offset trajectories and checkpoint hashes are retained. ' + seam_sentence]
    lines += table('Coordinate-boundary checks on released outputs. Seam distances are ranges across the three test latitudes at the smallest longitude offset; pole distances compare two descriptions of one point.', 'tab:boundary', 'lrr', 'Checkpoint&Seam distance&Same-pole distance', rows, 'All distances here are between unit-normalized feature vectors. A vanishing seam distance does not by itself certify differentiability. Testing beyond the primary latitude band diagnoses coordinate equivalence, not in-domain prediction performance.')
    lines += ['GeoCLIP, CSP and Climplicit retain finite jumps as the longitude gap shrinks. The sampled SINR and SatCLIP seam distances decrease towards zero. SINR, GeoCLIP, CSP and Climplicit give different outputs for the two descriptions of the pole, whereas SatCLIP gives identical outputs.\n\nFor chart-dependent models, a probe ring is rejected from the smooth-field summary if its disc intersects the antimeridian. For a center in the adjacent longitude half-plane, the clearance is $R\\arcsin(|\\cos\\varphi\\sin(\\pi-|\\lambda|)|)$. Near-seam refinement halves the starting radius until it is smaller than this clearance. Polar-coordinate singularities are excluded; the reported primary centers lie within $85^\\circ$. For encoders whose outputs are continuous across the seam (SINR and SatCLIP), rings that cross it are retained. ' + grid_sentence]
    write('tables/boundaries.tex', lines)
    def fmt3(value):
        if value != 0 and abs(value) < 1e-3:
            mantissa, exponent = f'{value:.2e}'.split('e')
            return f'${mantissa}\\times10^{{{int(exponent)}}}$'
        return f'{value:#.3g}'.rstrip('.')

    rows = []
    summary = []
    for n in ['GeoCLIP', 'SatCLIP vit16l10', 'SatCLIP vit16l40', 'SINR cap1000', 'Climplicit', 'CSP iNat2018', 'CSP fMoW']:
        r = [x for x in A['radius_cases'] if x['encoder'] == n]
        good = [x for x in r if x['converged']]
        rr = [1000 * x['radius_km'] for x in good]
        errors = [100 * x['final_A_error'] for x in good if 'final_A_error' in x]
        range_text = (f'{min(rr):.3g}' if min(rr) == max(rr) else f'{min(rr):.3g}--{max(rr):.3g}') if rr else '---'
        err = fmt3(max(errors)) if errors else '---'
        tensor_errors = [100 * x['final_tensor_minor_relative_error'] for x in good if 'final_tensor_minor_relative_error' in x]
        shape_errors = [x['final_shape_discrepancy'] for x in good if 'final_shape_discrepancy' in x]
        te = fmt3(max(tensor_errors)) if tensor_errors else '---'
        sd = f'{max(shape_errors):.6f}' if shape_errors else '---'
        rows.append(f'{short[n]}&{len(good)}/{len(r)}&{range_text}&{err}&{te}&{sd}')
        summary.append({'encoder': n, 'tested': len(r), 'converged': len(good), 'accepted_radii_m': rr, 'max_accepted_A_error_pct': max(errors) if errors else None, 'max_accepted_tensor_error_pct': max(tensor_errors) if tensor_errors else None, 'max_accepted_shape_discrepancy': max(shape_errors) if shape_errors else None})
    cases = A['radius_cases']
    accepted = [x for x in cases if x['converged']]
    n_cases, n_accepted = len(cases), len(accepted)
    n_checkpoints = len({x['encoder'] for x in cases})
    max_A_error = max(100 * x['final_A_error'] for x in accepted if 'final_A_error' in x)
    max_tensor_error = max(100 * x['final_tensor_minor_relative_error'] for x in accepted if 'final_tensor_minor_relative_error' in x)
    max_shape = max(x['final_shape_discrepancy'] for x in accepted if 'final_shape_discrepancy' in x)
    geo_tensor_error = max(100 * x['final_tensor_minor_relative_error'] for x in accepted if x['encoder'] == 'GeoCLIP')
    tolerance_pct = 100 * A['protocol']['tolerance']
    failed = {}
    for x in cases:
        if not x['converged']:
            failed.setdefault(x['encoder'], []).append(x)
    totals = Counter(x['encoder'] for x in cases)
    # The failure narrative below is prose; refuse to write it if the results it
    # describes have changed, rather than letting it go stale.
    narrative = {n: (sorted(Counter(x['stratum'] for x in v).items()), totals[n]) for n, v in failed.items()}
    expected = {'GeoCLIP': ([('high_anisotropy', 2)], 8), 'SatCLIP vit16l40': ([('fallback', 2), ('ordinary', 1)], 8),
                'CSP iNat2018': ([('fallback', 2), ('high_anisotropy', 2), ('near_seam', 2), ('ordinary', 2)], 8),
                'CSP fMoW': ([('high_anisotropy', 1 + 1), ('near_seam', 1)], 6)}
    geo_failed_latitudes = sorted({x['latitude'] for x in failed.get('GeoCLIP', [])})
    geo_high = sum(x['stratum'] == 'high_anisotropy' for x in cases if x['encoder'] == 'GeoCLIP')
    if narrative != expected or geo_failed_latitudes != [-80.0] or geo_high != 2:
        raise ValueError(f'Refinement failures changed ({narrative}); update the failure narrative in reporting/audit.py.')
    lines = ["The refinement demonstration selects sites from cached fields before refinement: two nearest the median anisotropy within $60^\\circ$, two with the largest anisotropy through $80^\\circ$, and up to two initial fallback sites. Selection uses Jacobian anisotropy where available and finite-lag anisotropy otherwise. Two further sites, $(0^\\circ,179.999^\\circ)$ and $(45^\\circ,179.999^\\circ)$, examine refinement under narrow chart clearance. Selection categories can overlap.\n\nRefinement begins at the checkpoint's primary radius, reduced where needed to keep the disc inside the chart. Up to " + str(A['protocol']['max_halvings']) + " additional halvings are allowed under the stopping rule of Section~\\ref{supp:radius}. Each sampled profile, fitted tensor, radius, chart reduction and termination is recorded. Cases that do not meet the criterion are recorded as failures, and their final estimates are not used."]
    lines += table('Released-checkpoint refinement. Errors and shape discrepancies are maxima over accepted cases with automatic-derivative references.', 'tab:releasedradius', 'lrrrrr', 'Checkpoint&Pass/total&Radius (m)&$e_A$ (\\%)&$e_G$ (\\%)&$D$', rows, '$e_A=|A_r/A_J-1|$; $e_G=\\|\\widehat G_r-G_J\\|_2/\\lambda_{\\min}(G_J)$; $D$ is the shape discrepancy defined in Section~\\ref{supp:geoclip}. Values are rounded; $D=1.000000$ denotes agreement to the shown precision. The CSP rows use the released single-precision path, which converts coordinates through NumPy and therefore provides no automatic derivatives; a separate double-precision Jacobian comparison for CSP is reported in Table~\\ref{tab:cspprecision}. Other automatic derivatives follow the same evaluation path as the finite probes, including GeoCLIP\'s mixed precision; they are independent of the finite-radius fit but are not arbitrary-precision certificates.')
    lines += [f"The {n_cases} cases were selected to be difficult (high anisotropy, fallbacks and proximity to the antimeridian), so the convergence fraction is not a global success rate. Across these cases, {n_accepted} converge. GeoCLIP fails at the two selected high-anisotropy sites at $80^\\circ$S, and SatCLIP ViT-L40 fails at one ordinary-category site and two fallback sites. All eight CSP-iNat cases and three of six CSP-fMoW cases fail. The accepted results also illustrate the distinction between successive-change control and reference error: GeoCLIP's largest accepted tensor error is {geo_tensor_error:.3f}\\%, although successive estimates differed by at most {tolerance_pct:g}\\% when refinement stopped. Continued halving alone therefore cannot establish accuracy. Section~\\ref{{supp:csp}} shows that single-precision rounding prevents convergence at the CSP sites that were swept in double precision; the remaining failures are not diagnosed further."]
    write('tables/released_supplement.tex', lines)
    write('tables/released_main.tex', [f'The released-model refinement test examines {n_cases} selected checkpoint--site cases across {word(n_checkpoints)} checkpoints, including ordinary and high anisotropy, initial fallback and near-antimeridian locations. Starting radii are reduced where necessary to keep the probe disc within a chart. The criterion accepts {n_accepted} cases and rejects {n_cases - n_accepted}, with all eight CSP-iNat cases failing and additional failures in GeoCLIP, SatCLIP-L40 and CSP-fMoW. Among accepted cases with automatic derivatives, maximum anisotropy error is {max_A_error:.3f}\\%, maximum minor-eigenvalue-relative tensor error is {max_tensor_error:.3f}\\%, and maximum shape discrepancy is $D={max_shape:.6f}$. These reference comparisons show why the {tolerance_pct:g}\\% successive-change threshold must be interpreted as a stopping rule rather than an accuracy guarantee. Site selection, radii, errors and evaluation precision are documented in supplementary \\S S4.\n\nCoordinate-boundary tests provide a complementary assessment across all {checkpoint_count(len(A["boundaries"]))} checkpoints. GeoCLIP, CSP and Climplicit retain antimeridian jumps, while SINR is longitude-periodic but gives different outputs for coordinate-equivalent poles. SatCLIP passes the sampled equivalence checks. These properties determine how the geographic domain can be used for local measurement, although agreement on a finite set of boundary tests does not establish global smoothness.'])
    shape_labels = {'finite_vs_reference': 'Finite fit / Equal Earth', 'jacobian_vs_reference': 'Jacobian / Equal Earth', 'finite_vs_jacobian': 'Finite fit / Jacobian', 'isotropic_vs_reference': 'Isotropic / Equal Earth'}
    write('tables/shape_table.tex', table('Area-weighted shape discrepancy $D$ for GeoCLIP on the same 1,764 cached-Jacobian sites within $60^\\circ$. $D=1$ denotes identical shape; the isotropic row is the anisotropy of Equal Earth itself.', 'tab:shape', 'lrr', 'Comparison&Median $D$&95th percentile', [f"{label}&{S[k]['median']:.3f}&{S[k]['p95']:.3f}" for k, label in shape_labels.items()]))
    rows = []
    for n, m in A['metadata'].items():
        code = 'M32' if n == 'GeoCLIP' else 'F32' if n.startswith('CSP') else 'F64'
        normal = 'U' if n.startswith(('GeoCLIP', 'SatCLIP', 'CSP')) else 'R'
        rows.append(f"{short[n]}&{m['dim']}&{normal}&{m['radius']:g}&{code}")
    lines = table('Measured vectors, probe radii and evaluation precision. Exact checkpoint identifiers and full hashes are retained in the package metadata.', 'tab:checkpoints', 'lrlrl', 'Checkpoint&Dim.&Vector&Radius (km)&Precision', rows, 'U: unit-normalized output; R: raw features. F64: float64 evaluation; M32: float64 weights/input with released float32 accumulator/output; F32: float32 forward, then float64 output storage. SINR uses pre-head raw features. Climplicit concatenates raw 256-dimensional outputs for months 3, 6, 9 and 12 (1,024 dimensions).')
    lines += [r'Longitudes are wrapped to $[-180^\circ,180^\circ)$ before encoding; the seam and pole tests in Section~\ref{supp:sampling} assess coordinate equivalence.']
    write('tables/checkpoint_table.tex', lines)
    (P / 'summaries/audit_summary.json').write_text(json.dumps({'radius_by_checkpoint': summary, 'total_tested': len(A['radius_cases']), 'total_converged': sum((x['converged'] for x in A['radius_cases'])), 'boundary_exclusions': {n: r['seam_excluded'] for n, r in G['primary'].items()}}, indent=2) + '\n')
    (P / 'summaries/checkpoint_metadata.json').write_text(json.dumps(A['metadata'], indent=2) + '\n')

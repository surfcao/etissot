"""Generate radius, boundary, checkpoint and shape audit tables."""
from pathlib import Path
import json
import numpy as np

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
    rows = []
    for n, r in A['boundaries'].items():
        v = [x['distance_normalized'] for x in r['antimeridian'] if x['epsilon_degrees'] == 1e-06]
        rows.append(f"{short[n]}&{sci(min(v))}--{sci(max(v))}&{sci(r['pole_normalized_distance'])}")
    lines = ['The boundary tests examine whether equivalent geographic coordinates yield consistent feature vectors. All outputs are L2-normalized for this comparison, providing a bounded scale without changing the raw-feature quantity used in the main SINR and Climplicit analysis. At latitudes $0,45,75^\\circ$, paired longitudes $180^\\circ-\\epsilon$ and $-180^\\circ+\\epsilon$ are evaluated for $\\epsilon=10^{-2},10^{-4},10^{-6}$ degrees. A separate pole test compares $(90^\\circ,0^\\circ)$ and $(90^\\circ,-180^\\circ)$, two coordinate descriptions of the same point. The full offset trajectories and checkpoint hashes are retained.']
    lines += table('Coordinate-boundary checks on released outputs. Seam distances are ranges across the three test latitudes at the smallest longitude offset; pole distances compare two descriptions of one point.', 'tab:boundary', 'lrr', 'Checkpoint&Seam distance&Same-pole distance', rows, 'All distances here are between unit-normalized feature vectors. A vanishing seam distance does not by itself certify differentiability. Testing beyond the primary latitude band diagnoses coordinate equivalence, not in-domain prediction performance.')
    lines += ['GeoCLIP, CSP and Climplicit retain finite jumps as the longitude gap shrinks. The sampled SINR and SatCLIP seam distances decrease towards zero. SINR, GeoCLIP, CSP and Climplicit give different outputs for the two descriptions of the pole, whereas SatCLIP gives identical outputs.\n\nFor chart-dependent models, a probe ring is rejected from the smooth-field summary if its disc intersects the antimeridian. For a center in the adjacent longitude half-plane, the clearance is $R\\arcsin(|\\cos\\varphi\\sin(\\pi-|\\lambda|)|)$. Near-seam refinement halves the starting radius until it is smaller than this clearance. Polar-coordinate singularities are excluded; the reported primary centers lie within $85^\\circ$. For encoders whose outputs are continuous across the seam (SINR and SatCLIP), rings that cross it are retained. The two randomized grids contain one excluded ring for each CSP checkpoint and one for Climplicit; no fixed-grid GeoCLIP ring crosses the seam.']
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
    lines = ["The refinement demonstration selects sites from cached fields before refinement: two nearest the median anisotropy within $60^\\circ$, two with the largest anisotropy through $80^\\circ$, and up to two initial fallback sites. Selection uses Jacobian anisotropy where available and finite-lag anisotropy otherwise. Two further sites, $(0^\\circ,179.999^\\circ)$ and $(45^\\circ,179.999^\\circ)$, examine refinement under narrow chart clearance. Selection categories can overlap.\n\nRefinement begins at the checkpoint's primary radius, reduced where needed to keep the disc inside the chart. Up to 16 additional halvings are allowed under the stopping rule of Section~\\ref{supp:radius}. Each sampled profile, fitted tensor, radius, chart reduction and termination is recorded. Cases that do not meet the criterion are recorded as failures, and their final estimates are not used."]
    lines += table('Released-checkpoint refinement. Errors and shape discrepancies are maxima over accepted cases with automatic-derivative references.', 'tab:releasedradius', 'lrrrrr', 'Checkpoint&Pass/total&Radius (m)&$e_A$ (\\%)&$e_G$ (\\%)&$D$', rows, '$e_A=|A_r/A_J-1|$; $e_G=\\|\\widehat G_r-G_J\\|_2/\\lambda_{\\min}(G_J)$; $D$ is the shape discrepancy defined in Section~\\ref{supp:geoclip}. Values are rounded; $D=1.000000$ denotes agreement to the shown precision. The CSP rows use the released single-precision path, which converts coordinates through NumPy and therefore provides no automatic derivatives; a separate double-precision Jacobian comparison for CSP is reported in Table~\\ref{tab:cspprecision}. Other automatic derivatives follow the same evaluation path as the finite probes, including GeoCLIP\'s mixed precision; they are independent of the finite-radius fit but are not arbitrary-precision certificates.')
    lines += ["The 50 cases were selected to be difficult (high anisotropy, fallbacks and proximity to the antimeridian), so the convergence fraction is not a global success rate. Across these cases, 34 converge. GeoCLIP fails at the two selected high-anisotropy sites at $80^\\circ$S, and SatCLIP ViT-L40 fails at one ordinary-category site and two fallback sites. All eight CSP-iNat cases and three of six CSP-fMoW cases fail. The accepted results also illustrate the distinction between successive-change control and reference error: GeoCLIP's largest accepted tensor error is 0.780\\%, although successive estimates differed by at most 0.5\\% when refinement stopped. Continued halving alone therefore cannot establish accuracy. Section~\\ref{supp:csp} shows that single-precision rounding prevents convergence at the CSP sites that were swept in double precision; the remaining failures are not diagnosed further."]
    write('tables/released_supplement.tex', lines)
    write('tables/released_main.tex', ['The released-model refinement test examines 50 selected checkpoint--site cases across seven checkpoints, including ordinary and high anisotropy, initial fallback and near-antimeridian locations. Starting radii are reduced where necessary to keep the probe disc within a chart. The criterion accepts 34 cases and rejects 16, with all eight CSP-iNat cases failing and additional failures in GeoCLIP, SatCLIP-L40 and CSP-fMoW. Among accepted cases with automatic derivatives, maximum anisotropy error is 0.141\\%, maximum minor-eigenvalue-relative tensor error is 0.780\\%, and maximum shape discrepancy is $D=1.001411$. These reference comparisons show why the 0.5\\% successive-change threshold must be interpreted as a stopping rule rather than an accuracy guarantee. Site selection, radii, errors and evaluation precision are documented in supplementary \\S S4.\n\nCoordinate-boundary tests provide a complementary assessment across all thirteen checkpoints. GeoCLIP, CSP and Climplicit retain antimeridian jumps, while SINR is longitude-periodic but gives different outputs for coordinate-equivalent poles. SatCLIP passes the sampled equivalence checks. These properties determine how the geographic domain can be used for local measurement, although agreement on a finite set of boundary tests does not establish global smoothness.'])
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

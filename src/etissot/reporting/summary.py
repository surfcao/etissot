"""Generate encoder and prediction summaries and primary figures."""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from etissot.geometry.fitting import fit_many, fit_ellipse
from etissot.statistics.spatial import GLAT, GLON
from etissot.geometry.references import equal_earth_metric

def run(workspace):
    """Run this analysis using explicit input and output directories."""
    ROOT = workspace.data
    P = workspace.output
    matplotlib.use('Agg')
    G = json.loads((P / 'summaries/geometry.json').read_text())
    W = ['CONUS', 'Europe', 'Australia', 'TropSAm']
    T = ['ELEVATION', 'TREE_COVER', 'SOIL_MOISTURE']
    V = ['iso', 'SatCLIP-L40', 'SINR', 'Climplicit', 'GeoCLIP-1km', 'shuffled', 'oracle', 'embedding', 'spatial+embedding', 'tensor+embedding']
    labels = {'iso': 'Isotropic', 'SatCLIP-L40': 'SatCLIP-L40 shape', 'SINR': 'SINR shape', 'Climplicit': 'Climplicit shape', 'GeoCLIP-1km': 'GeoCLIP shape', 'shuffled': 'Shuffled tensor', 'oracle': 'Target-derived tensor', 'embedding': 'Embedding kernel', 'spatial+embedding': 'Spatial + embedding', 'tensor+embedding': 'Tensor + embedding'}
    short = {'GeoCLIP': 'GeoCLIP', 'SatCLIP vit16l10': 'SatCLIP ViT-L10', 'SatCLIP r18l10': 'SatCLIP R18-L10', 'SatCLIP r50l10': 'SatCLIP R50-L10', 'SatCLIP vit16l40': 'SatCLIP ViT-L40', 'SatCLIP r18l40': 'SatCLIP R18-L40', 'SatCLIP r50l40': 'SatCLIP R50-L40', 'SINR cap1000': 'SINR cap-1000', 'SINR cap10': 'SINR cap-10', 'SINR distilled': 'SINR distilled', 'CSP iNat2018': 'CSP-iNat', 'CSP fMoW': 'CSP-fMoW', 'Climplicit': 'Climplicit'}
    R = {}
    for w in W:
        d = json.loads(workspace.fit_path(w).read_text())
        assert len(d) == 90, (w, len(d))
        R.update(d)
    ratio = lambda w, t, s, v: R[f'{w}|{t}|{s}|{v}']['rmse'] / R[f'{w}|{t}|{s}|iso']['rmse']
    vals = {v: np.array([ratio(w, t, s, v) for w in W for t in T for s in range(3)]).reshape(4, 3, 3) for v in V}

    def write(name, lines):
        (P / name).write_text('\n'.join(lines) + '\n')

    def table(cap, label, cols, head, body, foot=''):
        return ['\\begin{table}[htbp]\\centering\\small', '\\caption{' + cap + '}', '\\label{' + label + '}', '\\begin{tabular}{' + cols + '}', '\\toprule', head + '\\\\\\midrule', *body, '\\bottomrule\\end{tabular}', '\\par\\smallskip\\parbox{.97\\textwidth}{\\footnotesize ' + foot + '}' if foot else '', '\\end{table}']
    sgn = lambda t: '$-$' + t[1:] if t.startswith('-') else t
    ci = lambda v, ab: f'{sgn(f"{v:.2f}")} [{sgn(f"{ab[0]:.2f}")}, {sgn(f"{ab[1]:.2f}")}]'
    followups = P / 'summaries/review_followups.json'
    common = json.loads(followups.read_text()).get('common_radius_0p3_fixed_grid', {}) if followups.exists() else {}
    body = []
    def common_median(n):
        return f"{common[n]['median']:.2f}" if n in common else '---'

    for n, r in G['primary'].items():
        name = short[n] + ('$^\\dagger$' if r['sample'] != 'fixed' else '')
        body.append(f"{name}&{(1.0 if n.startswith(('GeoCLIP', 'SatCLIP')) else 0.3 if n.startswith('SINR') else 0.1):g}&{('U' if n.startswith(('GeoCLIP', 'SatCLIP', 'CSP')) else 'R')}&{r['median']:.2f}&{common_median(n)}&{ci(r['corr60'], r['corr_ci'])}&{ci(r['bearing'], r['bearing_ci'])}" + '\\\\')
    def wide(lines):
        lines = list(lines)
        i = next(k for k, l in enumerate(lines) if l.startswith('\\begin{tabular}'))
        j = next(k for k, l in enumerate(lines) if l.startswith('\\bottomrule\\end{tabular}'))
        lines[i] = '\\begin{adjustbox}{max width=\\textwidth}\n' + lines[i]
        lines[j] = lines[j] + '\n\\end{adjustbox}'
        return lines
    write('tables/encoder_table.tex', wide(table('Geometry of released checkpoints and association with input references. Intervals describe geographic resampling sensitivity.', 'tab:enc', '@{}lrcrrrr@{}', 'Encoder & Radius (km)&Vector&Median $A$&$A$, 0.3\\,km & $r_A$ [95\\% interval] & $C_\\theta$ [95\\% interval]', body, '$r_A$: area-weighted correlation between log output and log input-reference anisotropy. $C_\\theta$: agreement between directions of greatest stretch. U: unit-normalized output; R: raw features. Probe radii and feature conventions differ across families; $A$, 0.3\\,km gives the median anisotropy of each checkpoint re-evaluated at a common 0.3\\,km radius on the fixed grid. $^\\dagger$Two pooled randomized grids; other rows use the fixed grid. Median $A$: $|\\varphi|\\le85^\\circ$; correlation and bearing: $|\\varphi|\\le60^\\circ$. Bearings require both anisotropies to exceed $1+10^{-6}$. Probe discs crossing the antimeridian are excluded for GeoCLIP, CSP and Climplicit, whose outputs jump there. Input references are analytic Equal Earth (GeoCLIP), equally scaled longitude--latitude sinusoids (CSP), and 2:1 wrap/scaled coordinates (SINR/Climplicit); SatCLIP uses unnormalized harmonic-input profiles at 1 km from the implementation used here.')))
    body = []
    for n, phases in G['randomized_phases'].items():
        f = G['fixed_grid'][n]
        p = G['primary'][n]
        body.append('&'.join([short[n]] + [sgn(f'{v:.2f}') for v in (f['corr60'], phases[0]['corr60'], phases[1]['corr60'], p['corr60'], f['bearing'], p['bearing'])]) + '\\\\')
    write('tables/sampling_table.tex', table('Grid-phase sensitivity at $|\\varphi|\\le60^\\circ$. R0 and R1 are independent randomized grids; pooled values are primary.', 'tab:phase', '@{}lrrrrrr@{}', '&\\multicolumn{4}{c}{$r_A$}&\\multicolumn{2}{c}{$C_\\theta$}\\\\ Encoder&Fixed&R0&R1&Pooled&Fixed&Pooled', body))
    def fmt3(value):
        if value != 0 and abs(value) < 1e-3:
            mantissa, exponent = f'{value:.2e}'.split('e')
            return f'${mantissa}\\times10^{{{int(exponent)}}}$'
        return f'{value:#.3g}'.rstrip('.')

    body = []
    for n, r in G['jacobian_tails'].items():
        body.append(f"{short[n]}&{fmt3(100 * r['median_error'])}&{fmt3(100 * r['p95_error'])}&{fmt3(100 * r['high_A_p95_error'])}&{r['fallback_n']}" + '\\\\')
    write('tables/tail_table.tex', table('Area-weighted absolute relative error of finite-radius anisotropy against the Jacobian at each checkpoint\'s primary radius, in percent. High-$A$ denotes the upper decile of Jacobian anisotropy.', 'tab:tails', '@{}lrrrr@{}', 'Encoder&Median&95th percentile&High-$A$: 95th&Fallbacks', body))
    body = []
    for v in V:
        x = vals[v]
        body.append(labels[v] + '&' + '&'.join((f'{z:.3f}' for z in [*x.mean(axis=(1, 2)), x.mean()])) + '\\\\')
    write('tables/kriging_table.tex', table('Relative held-out RMSE under matched multistart fitting. Lower is better; 1 is the paired isotropic baseline. Each window averages nine paired cases; pooled averages all 36.', 'tab:krig', '@{}lrrrrr@{}', 'Method&CONUS&Europe&Australia&Trop. SAm&Pooled', body, 'Tensor kernels use determinant-normalized shape only. Target-derived tensors use privileged information. Averages are descriptive; shared geography and repeated sites preclude treating cases as independent trials.'))
    lines = []
    for vs in [V[1:6], V[6:]]:
        body = []
        for iw, w in enumerate(W):
            for it, t in enumerate(T):
                body.append(f'{w} / ' + {'ELEVATION': 'elevation', 'TREE_COVER': 'tree cover', 'SOIL_MOISTURE': 'soil moisture'}[t] + '&' + '&'.join((f'{vals[v][iw, it].mean():.3f}' for v in vs)) + '\\\\')
        heads = {'SatCLIP-L40': 'SatCLIP', 'GeoCLIP-1km': 'GeoCLIP', 'shuffled': 'Shuffled', 'oracle': 'Target', 'embedding': 'Feature', 'spatial+embedding': 'Spatial+F', 'tensor+embedding': 'Tensor+F', 'SINR': 'SINR', 'Climplicit': 'Climplicit'}
        lines += table('Per-window and per-target relative RMSE, averaged across the three paired splits: ' + ('shape-only kernels from encoder tensors and the shuffled-tensor control.' if len(vs) == 5 else 'target-derived tensor and embedding-kernel variants.'), 'tab:paired' + str(len(vs)), '@{}l' + 'r' * len(vs) + '@{}', 'Window / target&' + '&'.join((heads[v] for v in vs)), body)
    worsts = []
    for v in V[1:]:
        x = vals[v]
        ii = np.unravel_index(x.argmax(), x.shape)
        worsts.append({'method': v, 'ratio': float(x[ii]), 'window': W[ii[0]], 'target': T[ii[1]], 'split': int(ii[2])})
    lines.append('\\paragraph{Adverse cases.} The largest paired relative RMSE for each method is ' + '; '.join((f"{labels[r['method']]} {r['ratio']:.3f} ({r['window']}, {r['target'].lower().replace('_', ' ')}, split {r['split']})" for r in worsts)) + '. These are descriptive extrema, not independently replicated effects.')
    write('tables/kriging_windows.tex', lines)
    allruns = [a for r in R.values() for a in r['starts']]
    best = [min(r['starts'], key=lambda a: a['nll']) for r in R.values()]
    assert len(R) == 360 and len(allruns) == 1080
    assert all((abs(min((s['nll'] for s in r['starts'])) - r['nll']) < 1e-08 for r in R.values()))
    nested = []
    combo_nested = []
    for key, r in R.items():
        if key.split('|')[-1] in ['SatCLIP-L40', 'SINR', 'Climplicit', 'GeoCLIP-1km', 'shuffled', 'oracle']:
            iso = R[key.rsplit('|', 1)[0] + '|iso']
            nested.append(r['nll'] - iso['nll'])
            assert nested[-1] < 1e-05
    for key, r in R.items():
        if key.endswith('|tensor+embedding'):
            base = R[key.rsplit('|', 1)[0] + '|spatial+embedding']
            combo_nested.append(r['nll'] - base['nll'])
            assert combo_nested[-1] < 1e-05
    success = sum((s['success'] for s in allruns))
    selected = sum((s['success'] for s in best))
    spread = sum((r['nll_spread'] > 1 for r in R.values()))
    write('tables/optimization.tex', [f'All 360 method--case fits and 1,080 starts were completed. Optimizers reported convergence in {success} starts; {selected} of the 360 selected starts reported convergence. In {spread} fits the worst and best final training objectives differed by more than 1 NLL unit. Every selected pure tensor fit matched or improved its nested isotropic training objective, and all 36 tensor-plus-embedding fits matched or improved their contained spatial-plus-embedding solution (numerical tolerance $10^{{-5}}$). Selection was verified against the minimum stored candidate objective in all 360 fits. These checks do not establish a global optimum.'])
    mac = {'GeoCorr': G['primary']['GeoCLIP']['corr60'], 'GeoDezonal': G['geoclip_shape']['dezonal_corr'], 'GeoShapeMed': G['geoclip_shape']['median_relative_anisotropy'], 'GeoShapeP': G['geoclip_shape']['p95_relative_anisotropy'], 'CSPJitterCorr': G['primary']['CSP iNat2018']['corr60'], 'CSPJitterBearing': G['primary']['CSP fMoW']['bearing']}
    lines = [f'\\newcommand{{\\{k}}}{{{v:.2f}}}' for k, v in mac.items()]
    mac.update({'SatMid': vals['SatCLIP-L40'][:2].mean(), 'SatNew': vals['SatCLIP-L40'][2:].mean(), 'SatPooled': vals['SatCLIP-L40'].mean(), 'EmbPooled': vals['embedding'].mean(), 'SpatialEmbPooled': vals['spatial+embedding'].mean(), 'TensorEmbPooled': vals['tensor+embedding'].mean()})
    lines += [f'\\newcommand{{\\{k}}}{{{mac[k]:.3f}}}' for k in ['SatMid', 'SatNew', 'SatPooled', 'EmbPooled', 'SpatialEmbPooled', 'TensorEmbPooled']]
    write('tables/numbers.tex', lines)
    report = {'means': {v: {'original': float(x[:2].mean()), 'added': float(x[2:].mean()), 'pooled': float(x.mean()), 'windows': x.mean(axis=(1, 2)).tolist()} for v, x in vals.items()}, 'worst_cases': worsts, 'fits': len(R), 'starts': len(allruns), 'converged_starts': success, 'selected_converged': selected, 'nll_spread_above1': spread, 'max_combined_nested_nll_difference': max(combo_nested), 'max_nested_nll_difference': max(nested)}
    (P / 'summaries/kriging.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, ax = plt.subplots(figsize=(9.7, 5.2))
    ys = np.arange(len(V) - 1)[::-1]
    for j, (group, color, offset) in enumerate([('original', '#2878b5', 0.19), ('added', '#dc7c30', 0), ('pooled', '#222222', -0.19)]):
        x = [100 * (report['means'][v][group] - 1) for v in V[1:]]
        ax.scatter(x, ys + offset, c=color, s=32, marker=['o', 's', 'D'][j], label={'original': 'Original windows', 'added': 'Added windows', 'pooled': 'All four windows'}[group])
    ax.axvline(0, c='.5', lw=0.8)
    ax.set_yticks(ys, labels=[labels[v] for v in V[1:]])
    ax.grid(axis='x', alpha=0.25)
    ax.set_xlabel('Mean paired change in held-out RMSE (%)')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(P / 'figures/fig_kriging.png', dpi=200)
    plt.close(fig)
    D = np.load(ROOT / 'profiles/global.npz')
    bear = np.arange(32) * 2 * np.pi / 32
    th = np.linspace(0, 2 * np.pi, 361)
    fig, axes = plt.subplots(3, 3, figsize=(9.7, 9.5), subplot_kw={'projection': 'polar'})
    for i, (name, radius) in enumerate([('GeoCLIP', 1), ('SatCLIP vit16l40', 1), ('SINR cap1000', 0.3)]):
        for j, (lat, lon) in enumerate([(40, -105), (0, 20), (75, 20)]):
            idx = np.argmin((GLAT.ravel() - lat) ** 2 + (GLON.ravel() - lon) ** 2)
            la = GLAT.ravel()[idx]
            lo = GLON.ravel()[idx]
            p = D['prof__' + name][idx]
            p = p / np.exp(np.mean(np.log(np.maximum(p, 1e-30))))
            f = fit_ellipse(p, bear)
            curve = np.sqrt(f['lambda1'] * np.cos(th - f['theta_fit']) ** 2 + f['lambda2'] * np.sin(th - f['theta_fit']) ** 2)
            ax = axes[i, j]
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.plot(np.r_[bear, bear[0]], np.r_[p, p[0]], 'o-', ms=2, lw=0.8, label='Sampled stretch')
            ax.plot(th, curve, c='#c23b32', lw=1.1, label='Fitted stretch curve')
            ang = f['theta_fit']
            ax.plot([ang, ang + np.pi], [curve.max(), curve.max()], c='black', lw=0.8)
            ax.set_title(f"{short[name]}, {radius:g} km\n{la:g}° N, {lo:g}° E\nA = {f['A_fit']:.2f}; bearing = {np.degrees(ang):.0f}°; ν = {f['ne']:.3f}", fontsize=8, pad=14)
            ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.35)
    axes[0, 0].legend(loc='lower left', bbox_to_anchor=(-0.2, -0.35), fontsize=7, frameon=False)
    fig.tight_layout(h_pad=3)
    fig.savefig(P / 'figures/fig_indicatrix.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    coast_path = ROOT / 'reference/maps/coastlines.npz'
    if not coast_path.exists():
        raise FileNotFoundError(coast_path)
    coast = np.load(coast_path)
    f = fit_many(D['prof__GeoCLIP'], bear)
    a, theta = equal_earth_metric(GLAT.ravel(), GLON.ravel())
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, data, title, vmin, vmax, cmap in zip(axs.flat, [np.log(a), np.log(f['A_fit']), np.degrees(theta), np.degrees(f['theta_fit'])], ['Equal Earth: log anisotropy', 'GeoCLIP: log anisotropy', 'Equal Earth: major-axis bearing', 'GeoCLIP: major-axis bearing'], [0, 0, 0, 0], [2, 2, 180, 180], ['viridis', 'viridis', 'twilight', 'twilight']):
        data = data.copy()
        if title == 'GeoCLIP: major-axis bearing':
            data[f['fallback']] = np.nan
        im = ax.pcolormesh(GLON, GLAT, data.reshape(GLAT.shape), shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-85, 85)
        ax.contour(coast['lon'], coast['lat'], coast['land'], levels=[0.5], colors='.2', linewidths=0.35, alpha=0.7)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.grid(alpha=0.25, lw=0.3)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    for ax in axs[:, 1]:
        ax.scatter(GLON.ravel()[f['fallback']], GLAT.ravel()[f['fallback']], s=15, marker='x', color='#ff3282', linewidths=0.8, zorder=5)
    for ax in axs[-1]:
        ax.set_xlabel('Longitude (degrees)')
    for ax in axs[:, 0]:
        ax.set_ylabel('Latitude (degrees)')
    fig.tight_layout()
    fig.savefig(P / 'figures/geoclip_distortion_maps.png', dpi=200)
    plt.close(fig)
    print(json.dumps(report, indent=2))

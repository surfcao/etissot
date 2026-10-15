"""Cell-level feature/head error associations and paired spatial bootstrap."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import rankdata
from etissot.geometry.fitting import fit_many
from etissot.statistics.spatial import wcorr, GLAT
B = np.arange(32) * 2 * np.pi / 32

def stat(x, y, z):
    rx = rankdata(x)
    ry = rankdata(y)
    design = np.column_stack([*[rankdata(col) for col in z.T], np.ones(len(x))])
    resid = np.column_stack([rx, ry]) - design @ np.linalg.lstsq(design, np.column_stack([rx, ry]), rcond=None)[0]
    return np.array([np.corrcoef(rx, ry)[0, 1], np.corrcoef(resid.T)[0, 1]])

def run(workspace):
    """Run this analysis with explicit paths; return outputs in workspace.output."""
    ROOT = workspace.data
    P = workspace.output
    d = np.load(ROOT / 'downstream/feature_head_profiles.npz')
    e = np.load(ROOT / 'downstream/cell_outcomes.npz')
    maskpath = ROOT / 'downstream/land_mask.npz'
    if not maskpath.exists():
        raise FileNotFoundError(maskpath)
    masks = np.load(maskpath)
    rows = []
    global_corr = {}
    for tag in ['cap1000', 'cap10', 'distilled']:
        cells = e['cells__' + tag]
        land = masks[tag]
        assert land.dtype == bool and land.shape == cells.shape and (land.sum() == 13045)
        la = cells // 360 - 90 + 0.5
        z = np.column_stack([np.log1p(e['dens_cell__' + tag]), abs(la), np.log1p(e['rich_cell__' + tag])])[land]
        lc = cells[land]
        ids = (lc // 360 - 90) // 10 * 100 + (lc % 360 - 180) // 10
        blocks = np.unique(ids)
        groups = {k: np.flatnonzero(ids == k) for k in blocks}
        rng = np.random.default_rng(0)
        draws = [np.concatenate([groups[k] for k in rng.choice(blocks, len(blocks))]) for _ in range(500)]
        global_a = {}
        for space in ['feat', 'logit']:
            fit = fit_many(d[f'prof_{space}_cells__{tag}'], B)
            x = np.log(fit['A_fit'][land])
            assert np.isfinite(x).all()
            global_a[space] = np.log(fit_many(d[f'prof_{space}_global__{tag}'], B)['A_fit'])
            for metric, sign in [('brier_cell', 1), ('lauc_cell', -1)]:
                y = e[f'{metric}__{tag}'][land] * sign
                assert np.isfinite(y).all() and np.isfinite(z).all()
                point = stat(x, y, z)
                boot = np.array([stat(x[i], y[i], z[i]) for i in draws])
                ci = np.percentile(boot, [2.5, 97.5], axis=0)
                rows.append({'model': tag, 'space': space, 'metric': metric, 'n': int(land.sum()), 'fallback_sites': int(fit['fallback'][land].sum()), 'raw': float(point[0]), 'partial': float(point[1]), 'raw_ci': ci[:, 0].tolist(), 'partial_ci': ci[:, 1].tolist(), 'replicates': boot.tolist()})
                print(tag, space, metric, 'point', point, 'intervals', ci.tolist(), flush=True)
        global_corr[tag] = float(wcorr(global_a['feat'], global_a['logit'], np.cos(np.radians(GLAT.ravel()))))
    report = {'protocol': {'blocks_degrees': [10, 10], 'replicates': 500, 'seed': 0, 'land_mask': 'downstream/land_mask.npz; global-land-mask 1.0.0 at cell centres', 'estimator': 'current etissot.geometry.fitting.fit_many; flags retained', 'adjustment': 'rank-regression on log1p density, absolute latitude, log1p richness; ranks refit within each bootstrap draw', 'sampling_weight': 'one per retained land cell'}, 'rows': rows, 'global_feature_head_logA_correlation': global_corr, 'max_partial': max((r['partial'] for r in rows)), 'historical_four_site_summary': {k: np.load(ROOT / 'downstream/historical_cell_summaries.npz')[k].tolist() for k in np.load(ROOT / 'downstream/historical_cell_summaries.npz').files}}
    (P / 'summaries/downstream.json').write_text(json.dumps(report, indent=2) + '\n')

    def fmt(v, ci):
        clean = lambda q: 0.0 if abs(q) < 0.005 else q
        return f'{clean(v):.2f} [{clean(ci[0]):.2f}, {clean(ci[1]):.2f}]'
    lines = ['\\begin{table}[htbp]\\centering\\small', '\\caption{Regional error against current fitted anisotropy in SINR. Intervals describe geographic block-resampling sensitivity. Positive correlations indicate greater error at greater anisotropy.}', '\\label{tab:downfull}', '\\begin{tabular}{@{}lllrr@{}}\\toprule', 'Model&Space&Error&Raw $\\rho$&Partial $\\rho$\\\\\\midrule']
    for r in rows:
        lines.append({'cap1000': 'cap-1000', 'cap10': 'cap-10', 'distilled': 'distilled'}[r['model']] + '&' + {'feat': 'feature', 'logit': 'head'}[r['space']] + '&' + {'brier_cell': 'Brier', 'lauc_cell': '$1-$AUC'}[r['metric']] + '&' + fmt(r['raw'], r['raw_ci']) + '&' + fmt(r['partial'], r['partial_ci']) + '\\\\')
    lines += ['\\bottomrule\\end{tabular}', '\\par\\smallskip\\parbox{.97\\textwidth}{\\footnotesize Each row uses 13,045 land cells with equal cell weights. Partial correlations adjust ranks for training density, absolute latitude and richness. Percentile intervals use 500 resamples of $10^\\circ\\times10^\\circ$ blocks, seed 0; ranks and regressions are refitted in each draw.}', '\\end{table}']
    (P / 'tables/downstream_table.tex').write_text('\n'.join(lines) + '\n')
    print('Maximum partial correlation', report['max_partial'], 'global feature/head', global_corr, flush=True)

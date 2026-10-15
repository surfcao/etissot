"""Report declared checkpoint audit measurements and recompute common-radius geometry."""
import json
import numpy as np
from etissot.geometry.fitting import fit_many
from etissot.statistics.spatial import GLAT, GLON, wquantile

NAMES=['GeoCLIP', 'SatCLIP vit16l10', 'SatCLIP r18l10', 'SatCLIP r50l10', 'SatCLIP vit16l40', 'SatCLIP r18l40', 'SatCLIP r50l40', 'SINR cap1000', 'SINR cap10', 'SINR distilled', 'CSP iNat2018', 'CSP fMoW', 'Climplicit']
SHORT={'GeoCLIP': 'GeoCLIP', 'SatCLIP vit16l10': 'SatCLIP ViT-L10', 'SatCLIP r18l10': 'SatCLIP R18-L10', 'SatCLIP r50l10': 'SatCLIP R50-L10', 'SatCLIP vit16l40': 'SatCLIP ViT-L40', 'SatCLIP r18l40': 'SatCLIP R18-L40', 'SatCLIP r50l40': 'SatCLIP R50-L40', 'SINR cap1000': 'SINR cap-1000', 'SINR cap10': 'SINR cap-10', 'SINR distilled': 'SINR distilled', 'CSP iNat2018': 'CSP-iNat', 'CSP fMoW': 'CSP-fMoW', 'Climplicit': 'Climplicit'}

def num(v, fmt='{:.2f}'):
    t = fmt.format(v)
    return '$-$' + t[1:] if t.startswith('-') else t

def pct(v):
    if v is None:
        return '---'
    v = 100 * v
    if v == 0:
        return '0'
    if v < 0.01:
        e = int(np.floor(np.log10(v)))
        return f'${v / 10 ** e:.1f}\\times10^{{{e}}}$'
    return f'{v:.2f}' if v < 10 else f'{v:.1f}'

def tables(workspace, res):
    files = {}
    L = []
    if 'relu_crossing' in res:
        R = res['relu_crossing']
        L += [r'\begin{table}[htbp]\centering\small',
              r'\caption{Activation-region crossings at the 2,340 Jacobian sites. A probe disc crosses when any probe point has a different ReLU activation pattern from the center. Errors are area-weighted $|A_r/A_J-1|$ in percent.}',
              r'\label{tab:relu}', r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
              r'&&\multicolumn{2}{c}{Median error (\%)}&\multicolumn{2}{c}{95th percentile (\%)}\\',
              r'Checkpoint, radius&Crossing discs&Within&Crossing&Within&Crossing\\\midrule']
        for key, lab in [('SINR cap1000', 'SINR cap-1000, 0.3 km'), ('SINR cap10', 'SINR cap-10, 0.3 km'), ('SINR distilled', 'SINR distilled, 0.3 km')]:
            d = R[key]
            L.append(f"{lab}&{100 * d['crossing_fraction']:.1f}\\%&{pct(d['median_error_within'])}&{pct(d['median_error_crossing'])}&{pct(d['p95_error_within'])}&{pct(d['p95_error_crossing'])}\\\\")
        L += [r'\midrule', r'&Median switching units&\multicolumn{2}{c}{Lowest / highest switching tercile}&\multicolumn{2}{c}{Lowest / highest switching tercile}\\']
        for key, lab in [('GeoCLIP 1 km', 'GeoCLIP, 1 km'), ('GeoCLIP 0.3 km', 'GeoCLIP, 0.3 km')]:
            d = R[key]
            m, q = d['median_error_by_flip_tercile'], d['p95_error_by_flip_tercile']
            L.append(f"{lab}&{d['median_flipped_units']:.0f}&\\multicolumn{{2}}{{c}}{{{pct(m[0])} / {pct(m[2])}}}&\\multicolumn{{2}}{{c}}{{{pct(q[0])} / {pct(q[2])}}}\\\\")
        g1, g3 = R['GeoCLIP 1 km'], R['GeoCLIP 0.3 km']
        L += [r'\bottomrule\end{tabular}', r'\end{adjustbox}',
              r'\par\smallskip\parbox{.97\textwidth}{\footnotesize SINR Jacobians are double-precision automatic derivatives; for cap-1000 they reproduce the cached values to $'
              + f"{R['SINR cap1000']['jacobian_cache_max_relative_difference']:.0e}".replace('e-', r'\times10^{-').replace('e+', r'\times10^{') + r'}$. For all three SINR checkpoints, every site with error above 0.1\% is a crossing disc. Every GeoCLIP disc crosses at least one of its 9,216 units at both radii; the rows therefore compare terciles of the number of units whose pattern changes within the disc (Spearman correlation with error '
              + f"{g1['spearman_flips_error']:.2f} at 1 km, {g3['spearman_flips_error']:.2f} at 0.3 km). Crossing fractions by branch ($\\sigma=1,16,256$) are "
              + ', '.join(f'{100 * v:.0f}' for v in g1['crossing_fraction_by_branch_sigma_1_16_256']) + r'\% at 1 km and '
              + ', '.join(f'{100 * v:.0f}' for v in g3['crossing_fraction_by_branch_sigma_1_16_256']) + r'\% at 0.3 km.}',
              r'\end{table}']
        files['review_relu'], L = L, []
    if 'land_only' in res:
        R = res['land_only']
        L += [r'\begin{table}[htbp]\centering\small',
              r'\caption{Table~\ref{tab:enc} statistics over all sites and over land sites only. Land is defined by the 0.5$^\circ$ land mask used for the map coastlines; land covers '
              + f"{100 * R['GeoCLIP']['land_area_fraction_85']:.0f}" + r'\% of the area within $85^\circ$. Same samples, weights and exclusions as Table~\ref{tab:enc}.}',
              r'\label{tab:land}', r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',
              r'&\multicolumn{2}{c}{Median $A$}&\multicolumn{2}{c}{$r_A$}&\multicolumn{2}{c}{$C_\theta$}\\',
              r'Checkpoint&All&Land&All&Land&All&Land\\\midrule']
        for n in NAMES:
            a, b = R[n]['all'], R[n]['land']
            L.append(f"{SHORT[n]}&{num(a['median'])}&{num(b['median'])}&{num(a['corr60'])}&{num(b['corr60'])}&{num(a['bearing'])}&{num(b['bearing'])}\\\\")
        L += [r'\bottomrule\end{tabular}', r'\end{table}']
        files['review_land'], L = L, []
    if 'bearing_count' in res:
        R = res['bearing_count']
        L += [r'\begin{table}[htbp]\centering\small',
              r'\caption{Sensitivity to the number of bearings at 200 sites per checkpoint within $60^\circ$ (seed 20261016), each checkpoint at its primary radius. Profiles with 64 bearings are fitted in full and subsampled to 32 and 16. Entries are 95th percentiles over sites of the relative change in anisotropy (percent) and of the bearing change (degrees, sites with $A>1.05$), relative to 32 bearings.}',
              r'\label{tab:bearingcount}', r'\begin{tabular}{@{}lrrrr@{}}\toprule',
              r'&\multicolumn{2}{c}{$A$ change (\%)}&\multicolumn{2}{c}{Bearing change ($^\circ$)}\\',
              r'Checkpoint&16 bearings&64 bearings&16 bearings&64 bearings\\\midrule']
        for n in NAMES:
            d = R[n]
            sm = lambda v: '$<$0.01' if v < 0.005 else f'{v:.2f}'
            L.append(f"{SHORT[n]}&{sm(100 * d['p95_rel_A_16_vs_32'])}&{sm(100 * d['p95_rel_A_64_vs_32'])}&{sm(d['p95_bearing_deg_16_vs_32'])}&{sm(d['p95_bearing_deg_64_vs_32'])}\\\\")
        L += [r'\bottomrule\end{tabular}', r'\end{table}']
        files['review_bearings'], L = L, []
    dest = workspace.output / 'tables'
    dest.mkdir(parents=True, exist_ok=True)
    for name, lines in files.items():
        (dest / f'{name}.tex').write_text('\n'.join(lines) + '\n')
        print('wrote', dest / f'{name}.tex')

def run(workspace):
    source = workspace.data / 'measurements/review'
    res = json.loads((source/'review_followups.json').read_text())
    # Refit all common-radius profiles instead of accepting their saved summaries.
    with np.load(workspace.data/'profiles/common_radius_0p3.npz', allow_pickle=False) as profiles:
        lat, lon = GLAT.ravel(), GLON.ravel()
        weights = np.cos(np.radians(lat))
        for name in NAMES:
            fit = fit_many(profiles['prof__'+name], np.arange(32)*2*np.pi/32)
            keep = np.ones(len(lat), dtype=bool)
            if name in ['GeoCLIP', 'CSP iNat2018', 'CSP fMoW', 'Climplicit']:
                clearance = 6371*np.arcsin(np.abs(np.cos(np.radians(lat))*np.sin(np.pi-np.abs(np.radians(lon)))))
                keep &= (np.abs(lon)<90) | (clearance>0.3)
            res['common_radius_0p3_fixed_grid'][name] = dict(
                median=float(wquantile(fit['A_fit'][keep], .5, weights[keep])),
                p95=float(wquantile(fit['A_fit'][keep], .95, weights[keep])),
                fallback=int((fit['fallback'] & keep).sum()))
    (workspace.output/'summaries/review_followups.json').write_text(json.dumps(res,indent=1))
    tables(workspace,res)
    # The CSP site-level measurements permit independent recomputation of quantiles.
    csp = json.loads((source/'csp_precision.json').read_text())
    for model in csp['models'].values():
        rows = [r for r in model['sites'] if not r['refinement_site']]
        aj=np.array([r['AJ'] for r in rows]); high=aj>=np.quantile(aj,.9)
        summary=model['summary']
        quant=lambda values,q:float(np.sort(values)[min(int(np.ceil(q*len(values)))-1,len(values)-1)])
        summary['median_AJ']=quant(aj,.5)
        for precision,key in [('f32','A32'),('f64','A64')]:
            error=np.abs(np.array([r[key] for r in rows])/aj-1)
            for label,values,q in [('median_err',error,.5),('p95_err',error,.95),('highA_p95',error[high],.95)]:
                summary[precision+'_'+label]=quant(values,q)
    (workspace.output/'summaries/csp_precision.json').write_text(json.dumps(csp,indent=1))

    i=csp["models"]["CSP iNat2018"]["summary"]
    f=csp["models"]["CSP fMoW"]["summary"]

    row_inat = 'CSP-iNat (leaky ReLU)&' + '&'.join([
        f"{i['median_AJ']:.2f}", f"{100*i['f32_median_err']:.3f}",
        f"{100*i['f32_p95_err']:.2f}", f"{100*i['f32_highA_p95']:.2f}",
        f"{100*i['f64_median_err']:.4f}", f"{100*i['f64_p95_err']:.2f}",
        f"{100*i['f64_highA_p95']:.2f}", f"{100*i['crossing_fraction']:.0f}"+r'\%'])+r'\\'
    row_fmow = 'CSP-fMoW (GELU)&' + '&'.join([
        f"{f['median_AJ']:.2f}", f"{100*f['f32_median_err']:.4f}",
        f"{100*f['f32_p95_err']:.3f}", f"{100*f['f32_highA_p95']:.3f}",
        f"{100*f['f64_median_err']:.4f}", f"{100*f['f64_p95_err']:.4f}",
        f"{100*f['f64_highA_p95']:.4f}", '---'])+r'\\'
    template='\\begin{table}[htbp]\\centering\\small\n\\caption{CSP anisotropy error against double-precision Jacobians at 300 sites within $60^\\circ$ and the 0.1\\,km primary radius, in percent. High-$A$ denotes the upper decile of Jacobian anisotropy. Crossing is the fraction of probe discs in which at least one probe point has a different activation pattern from the center.}\n\\label{tab:cspprecision}\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{@{}lrrrrrrrr@{}}\\toprule\n&&\\multicolumn{3}{c}{Single precision (released)}&\\multicolumn{3}{c}{Double precision}&\\\\\nCheckpoint&Median $A_J$&Median&95th&High-$A$ 95th&Median&95th&High-$A$ 95th&Crossing\\\\\\midrule\n{row_inat}\n{row_fmow}\n\\bottomrule\\end{tabular}\n\\end{adjustbox}\n\\par\\smallskip\\parbox{.97\\textwidth}{\\footnotesize No site required a fallback in either precision. Crossing is undefined for the smooth GELU activation.}\n\\end{table}'
    table=template.replace("{row_inat}",row_inat).replace("{row_fmow}",row_fmow)
    (workspace.output/"tables/csp_precision_table.tex").write_text(table+"\n")

"""Generate previously handwritten supplementary empirical tables and neighbor prose."""
import json
import numpy as np

def run(workspace):
    out=workspace.output
    estimator=json.loads((out/'summaries/estimator_validity.json').read_text())
    historical=json.loads((out/'summaries/hidden_space.json').read_text())['models']
    neighbours=json.loads((out/'summaries/neighbours.json').read_text())
    lines=[r'\begin{table}[ht]\centering',r'\caption{Finite-radius fitted-anisotropy error at radius $r=1/(16A)$. All cases use 32 bearings, with no fallback.}\label{tab:counterexamples}',r'\begin{tabular}{rrr}\toprule',r'True $A$ & Fitted $A$ & Relative error (\%)\\\midrule']
    lines += [f"{r['A']}&{r['fitted']:.5f}&${r['relative_error_percent']:.3f}$"+r'\\' for r in estimator['counterexamples']]
    lines += [r'\bottomrule',r'\end{tabular}\end{table}']
    (out/'tables/counterexamples.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[ht]\centering',r'\caption{Historical five-seed exploratory means. Parentheses are across-seed standard deviations where available. Correlations use the original angular-extrema summary, not the revised primary ellipse estimator; these observations motivate hypotheses rather than test a uniquely identified mechanism.}',r'\begin{tabular}{lrrr}\toprule',r'Model&Target $R^2$&Input association&Target association\\\midrule']
    for name,label in [('rff16-mlp','RFF-16'),('geoclip-3cap','Three-capsule miniature'),('rff1-mlp','RFF-1'),('siren','SIREN')]:
        r=historical[name];m=r['mean'];sd=r['sd'];r2='---' if name=='siren' else f"{m['r2']:.3f}"
        inp=f"{m['input_correlation']:.2f}";target=f"{m['target_correlation']:.2f}"
        if inp.startswith('-'):inp='$'+inp+'$'
        if target.startswith('-'):target='$'+target+'$'
        lines.append(f"{label}&{r2}&{inp} ({sd['input_correlation']:.2f})&{target}"+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}\end{table}']
    (out/'tables/hidden_space.tex').write_text('\n'.join(lines)+'\n')
    def synthetic(latitude):return [r for r in neighbours['synthetic'] if np.isclose(r['latitude'],latitude)]
    def comma_and(values):return ', '.join(values[:-1])+' and '+values[-1]
    s60,s70=synthetic(60),synthetic(70)
    a60=comma_and([f"{r['embedding_elongation']:.2f}" for r in s60]);a70=comma_and([f"{r['embedding_elongation']:.2f}" for r in s70])
    lo60,hi60=min(r['axis_deviation_degrees'] for r in s60),max(r['axis_deviation_degrees'] for r in s60)
    lo70,hi70=min(r['axis_deviation_degrees'] for r in s70),max(r['axis_deviation_degrees'] for r in s70)
    pairs=comma_and([f"{r['embedding_median']:.2f}/{r['geodesic_median']:.2f}" for r in neighbours['gallery']])
    shares=[r['more_elongated_share']*100 for r in neighbours['gallery']]
    text=("The controlled neighbor experiment examines the geographic shape of locations selected by embedding similarity. It uses 20 queries in total, one at each combination of five reference latitudes and four reference longitudes. For each query, 400 candidate points are distributed uniformly within 6\\,km, from which 40 neighbors are retained. "
          f"At $60^\\circ$, the four reference-longitude elongations are {a60}, with major-axis deviations of approximately ${lo60:.1f}$--${hi60:.1f}^\\circ$ from the predicted perpendicular to the encoder's major-stretch axis. At $70^\\circ$, elongations are {a70}, with deviations ${lo70:.1f}$--${hi70:.1f}^\\circ$. "
          f"A separate observational comparison uses the nonuniform photographic gallery, with 300 queries per latitude band and 30 neighbors. The encoder/geodesic median elongations are {pairs}. Approximately {min(shares):.0f}--{max(shares):.0f}\\% of queries have greater elongation under encoder selection. Relative elongation differences at most $10^{{-10}}$ are treated as ties. Candidate density and content influence the gallery comparison, which therefore complements the controlled experiment without isolating the effect of encoder geometry. Both analyses concern location queries; neither evaluates image-to-location retrieval.")
    (out/'tables/neighbours.tex').write_text(text+'\n')

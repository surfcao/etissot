"""Publish matched geometry case-study figures and summaries for reproduction."""
import json
import shutil
from etissot.diagnostics.geometry_case_studies import run as analyze

def run(workspace):
    directory=workspace.output/'case_studies'
    report=analyze(workspace.data,directory)
    for name in ['geometry_comparison.png','rank_hubness.png']:
        shutil.copyfile(directory/name,workspace.output/'figures'/name)
    (workspace.output/'summaries/geometry_case_studies.json').write_text(json.dumps(report,indent=2)+'\n')
    import numpy as np
    rows=[]
    for meta in report['metadata']:
        name=meta['name']
        select=lambda pred:[r['overlap'] for r in report['local'] if r['model']==name and r['fit_radius']==.3 and r['radius']==1 and r['k']==40 and r['predictor']==pred]
        g=next(r for r in report['global_geometry'] if r['model']==name)
        h=next(r for r in report['hubness'] if r['model']==name and r['space']=='embedding' and r['k']==20)
        rows.append(f"{name}&{np.median(select('geographic')):.3f}&{np.median(select('tensor')):.3f}&{g['top_variance_share']:.3f}&{g['median_local_A']:.3f}&{h['max_over_mean']:.2f}"+r'\\')
    text=r'''\begin{table}[htbp]\centering\small
\caption{Matched geometry studies. Neighbor overlap is the median over 20 queries at a 1 km candidate radius, with $k=40$ and tensors fitted independently at 0.3 km. Global summaries use 512 equal-area locations; local anisotropy excludes flagged fits. Hub ratio is maximum neighbor occurrence divided by the mean for $k=20$.}
\label{tab:geometrycases}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Model&Geo. overlap&Tensor overlap&Top share&Local $A$&Hub ratio\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\end{table}
'''
    (workspace.output/'tables/geometry_cases.tex').write_text(text)
    from etissot.diagnostics.neighbourhoods import run as neighbourhoods
    neighbours=neighbourhoods(workspace.data,directory)
    for name in ['neighbourhood_comparison.png','neighbourhood_maps.png']+[f'neighbourhood_maps_detail_{i}.png' for i in [1,2,3]]:
        shutil.copyfile(directory/name,workspace.output/'figures'/name)
    (workspace.output/'summaries/neighbourhoods.json').write_text(json.dumps(neighbours,indent=2)+'\n')
    rows=[]
    display_labels={'SatCLIP L10 released': 'SatCLIP ViT-L10', 'SatCLIP L10 corrected': 'SatCLIP ViT-L10 input-rescaled', 'SatCLIP L40 released': 'SatCLIP ViT-L40', 'SatCLIP L40 corrected': 'SatCLIP ViT-L40 input-rescaled', 'SINR cap1000': 'SINR cap-1000', 'SINR cap10': 'SINR cap-10', 'SatCLIP r18l10': 'SatCLIP R18-L10', 'SatCLIP r18l40': 'SatCLIP R18-L40', 'SatCLIP r50l10': 'SatCLIP R50-L10', 'SatCLIP r50l40': 'SatCLIP R50-L40', 'CSP iNat2018': 'CSP-iNat', 'CSP fMoW': 'CSP-fMoW'}
    row_order=['GeoCLIP', 'SatCLIP ViT-L10', 'SatCLIP R18-L10', 'SatCLIP R50-L10', 'SatCLIP ViT-L40', 'SatCLIP R18-L40', 'SatCLIP R50-L40', 'SatCLIP ViT-L10 input-rescaled', 'SatCLIP ViT-L40 input-rescaled', 'SINR cap-1000', 'SINR cap-10', 'SINR distilled', 'CSP-iNat', 'CSP-fMoW', 'Climplicit']
    summary_rows=sorted(neighbours['summary'],key=lambda r:row_order.index(display_labels.get(r['model'],r['model'])))
    for r in summary_rows:
        rows.append(f"{display_labels.get(r['model'],r['model'])}&{r['queries']}&{r['geographic_overlap']:.3f}&{r['tensor_geographic_overlap']:.3f}&{r['tensor_overlap']:.3f}"+r'\\')
    table=r"""\begin{table}[htbp]\centering\small
\caption{Three-way neighborhood comparison. Entries are median overlaps for $k=40$ among 400 candidates within 1 km; tensors use independent 0.3 km probes. E/G compares actual embedding neighbors with geographic neighbors, T/G compares tensor-predicted neighbors with geographic neighbors, and E/T compares embedding and tensor selections directly. The same valid queries enter all three comparisons.}
\label{tab:neighbourhoods}
\begin{tabular}{@{}lrrrr@{}}\toprule
Checkpoint&Queries&E/G&T/G&E/T\\\midrule
"""+'\n'.join(rows)+r"""
\bottomrule\end{tabular}
\end{table}
"""
    (workspace.output/'tables/neighbourhoods.tex').write_text(table)
    return report

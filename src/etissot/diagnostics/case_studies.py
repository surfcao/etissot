"""Interpretation and diagnosis case studies from packaged measurements.

Run: python -m etissot.diagnostics.case_studies --data data --output outputs/case_studies
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from etissot.geometry.fitting import fit_many


def run(data, output):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(data/'measurements/neighbour_candidates.npz') as cache:
        d = {k: cache[k] for k in cache.files}
    rows = []
    selected = []
    for i, (lat, lon) in enumerate(d['query_radians']):
        sl = slice(d['offsets'][i], d['offsets'][i+1])
        candidates = d['candidate_radians'][sl]
        k = int(d['k'][i])
        angular = np.sin((candidates[:,0]-lat)/2)**2 + np.cos(lat)*np.cos(candidates[:,0])*np.sin((candidates[:,1]-lon)/2)**2
        geo = np.argsort(angular)[:k]
        emb = np.argsort(-d['similarity'][sl])[:k]
        overlap = len(np.intersect1d(geo, emb))/k
        rows.append(dict(latitude=float(np.degrees(lat)), longitude=float(np.degrees(lon)), group=int(d['group'][i]), k=k, replaced_fraction=1-overlap))
        # Select a full longitude series at a fixed latitude, without outcome-based selection.
        if d['group'][i] == -1 and np.isclose(np.degrees(lat),60):
            east = 6371*np.cos(lat)*((candidates[:,1]-lon+np.pi)%(2*np.pi)-np.pi)
            north = 6371*(candidates[:,0]-lat)
            selected.append((rows[-1], east, north, geo, emb))
    fig, axes = plt.subplots(1,4,figsize=(13,4.8),sharex=True,sharey=True)
    for ax,(row,e,n,g,b) in zip(axes,selected):
        ax.scatter(e,n,s=5,c='0.85',label='Candidate locations')
        ax.scatter(e[g],n[g],s=28,facecolors='none',edgecolors='#0072B2',label='Nearest geographically')
        ax.scatter(e[b],n[b],s=13,c='#D55E00',marker='x',label='Nearest in embedding')
        ax.plot(0,0,'k+',ms=9)
        ax.set(title=f"60°N, {row['longitude']:.0f}°E\n{row['replaced_fraction']:.1%} of neighbours differ",xlabel='East displacement (km)',aspect='equal',xlim=(-6.2,6.2),ylim=(-6.2,6.2))
        ax.grid(alpha=.15)
    axes[0].set_ylabel('North displacement (km)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(left=.055,right=.99,bottom=.24,top=.82,wspace=.12);fig.savefig(output/'geoclip_neighbourhoods.png',dpi=220,bbox_inches='tight');plt.close(fig)
    groups={str(g):{'queries':sum(r['group']==g for r in rows),'median_replaced_fraction':float(np.median([r['replaced_fraction'] for r in rows if r['group']==g]))} for g in [-1,0,1,2,3]}
    with np.load(data/'profiles/satclip_correction.npz') as cache:
        c={k:cache[k] for k in cache.files}
    bearings=np.arange(32)*2*np.pi/32
    metadata=json.loads(str(c['metadata']))
    fig,axes=plt.subplots(2,2,figsize=(9,6.5),sharex=True)
    comparisons=[]
    for col,model in enumerate([0,1]):
        for row,key in enumerate(['input_profiles','output_profiles']):
            profiles=c[key][model,:2,0]
            fits=[fit_many(x,bearings) for x in profiles]
            anisotropy=np.stack([f['A_fit'] for f in fits])
            fallback_counts=[int(f['fallback'].sum()) for f in fits]
            # All five longitudes at each latitude; show latitude medians, not geographic area summaries.
            latitudes=np.unique(c['sites'][:,0])
            for variant,(label,color) in enumerate([('Released','#0072B2'),('Corrected input, fixed network weights','#D55E00')]):
                median=[np.median(anisotropy[variant,c['sites'][:,0]==lat]) for lat in latitudes]
                axes[row,col].plot(latitudes,median,'o-',label=label,color=color,ms=4)
            axes[row,col].axhline(1,color='0.6',ls=':',lw=1)
            axes[row,col].set_title(metadata[model]['name']+(' — input' if row==0 else ' — learned output'))
            axes[row,col].set_ylabel('Median anisotropy A');axes[row,col].grid(alpha=.15)
            comparisons.append(dict(checkpoint=metadata[model]['name'],stage=key,anisotropy=anisotropy.tolist(),fallback_counts=fallback_counts))
        axes[1,col].set_xlabel('Latitude (degrees)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(output/'satclip_input_correction.png',dpi=220);plt.close(fig)
    result={'geoclip_groups':groups,'geoclip_displayed': [x[0] for x in selected],'geoclip_queries':rows,'satclip':comparisons,'scope':'Saved released-model measurements. GeoCLIP neighbour replacement is descriptive, not predictive error or causal attribution. SatCLIP shows ViT checkpoints, all 35 sites, 1 km; correction changes frozen-network inputs, not training. Latitude medians use five longitudes. Fallback flags and all six checkpoints remain in the existing correction audit.'}
    (output/'case_studies.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'groups':groups,'displayed':result['geoclip_displayed']},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data');parser.add_argument('--output',default='outputs/case_studies')
    args=parser.parse_args();run(args.data,args.output)

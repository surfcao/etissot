"""Matched GeoCLIP/SatCLIP neighbourhood audit with optional fresh collection.

python -m etissot.diagnostics.satclip_neighbours --collect --models models
Omit --collect to regenerate results from the saved SatCLIP similarities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def collect(data, models, output=None):
    from etissot.workspace import Workspace
    data = Path(data).resolve()
    destination_directory = Workspace(data, output or data.parent/'outputs/measurements').output
    destination_directory.mkdir(parents=True, exist_ok=True)
    import os
    import torch
    vendor=Path(__file__).resolve().parents[1]/'models/vendor/satclip/satclip'
    from etissot.models.loaders import load_satclip_checkpoint
    torch.set_num_threads(2)
    with np.load(data/'measurements/neighbour_candidates.npz') as cache:
        d={k:cache[k] for k in cache.files}
    positions,inverse=np.unique(np.concatenate([d['query_radians'],d['candidate_radians']]),axis=0,return_inverse=True)
    n=len(d['query_radians']);qi=inverse[:n];ci=inverse[n:]
    expected=json.loads((data/'reference/summaries/checkpoint_manifest.json').read_text())
    hashes={Path(v['checkpoint']).name:v['sha256'] for v in expected.values()}
    similarities=[];meta=[]
    for degree in [10,40]:
        paths=list(models.rglob(f'satclip-vit16-l{degree}.ckpt'))
        if len(paths)!=1:raise ValueError('Expected one matching checkpoint')
        path=paths[0];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes.get(path.name)!=digest:raise ValueError('Checkpoint checksum differs from pinned release')
        model=load_satclip_checkpoint(path,'cpu').double().eval()
        embeddings=[[],[]]
        for start in range(0,len(positions),1024):
            # SatCLIP takes longitude, latitude in degrees.
            x=torch.tensor(np.degrees(positions[start:start+1024,::-1].copy()),dtype=torch.float64)
            with torch.no_grad():
                features=model.posenc(x)
                scale=features.new_ones(degree**2);scale[[l*(l+1) for l in range(degree)]]=1/np.pi
                for v,y in enumerate([features,features*scale]):
                    output=model.nnet(y);output=torch.nn.functional.normalize(output,dim=1)
                    if not torch.isfinite(output).all():raise ValueError('Nonfinite output')
                    embeddings[v].append(output.numpy())
            if start%10240==0:print(f'L{degree}: {start}/{len(positions)} positions',flush=True)
        for v,chunks in enumerate(embeddings):
            e=np.concatenate(chunks);scores=np.empty(len(ci))
            for i in range(n):
                sl=slice(d['offsets'][i],d['offsets'][i+1]);scores[sl]=e[ci[sl]]@e[qi[i]]
            similarities.append(scores)
            meta.append(dict(name=f'SatCLIP ViT L{degree} '+('released' if v==0 else 'corrected input'),degree=degree,variant=v,checkpoint=path.name,sha256=digest))
        print(f'L{degree} complete',flush=True)
    destination=destination_directory/'satclip_neighbour_similarities.npz'
    np.savez_compressed(destination,similarities=np.stack(similarities),metadata=json.dumps(meta),source_sha256=hashlib.sha256((data/'measurements/neighbour_candidates.npz').read_bytes()).hexdigest(),protocol='Same queries/candidates/k as GeoCLIP; float64 SatCLIP; unit-normalized outputs; correction divides m=0 inputs by pi with fixed network weights; no retraining or stable-polynomial replacement.')
    return destination


def run(data,output,source=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from etissot.diagnostics.neighbours import elongation
    output.mkdir(parents=True,exist_ok=True)
    with np.load(data/'measurements/neighbour_candidates.npz') as cache:d={k:cache[k] for k in cache.files}
    with np.load(source if source is not None else data/'measurements/satclip_neighbour_similarities.npz') as cache:c={k:cache[k] for k in cache.files}
    if str(c['source_sha256'])!=hashlib.sha256((data/'measurements/neighbour_candidates.npz').read_bytes()).hexdigest():raise ValueError('Candidate cache changed')
    meta=json.loads(str(c['metadata']));rows=[];panels={}
    for i,(lat,lon) in enumerate(d['query_radians']):
        sl=slice(d['offsets'][i],d['offsets'][i+1]);points=d['candidate_radians'][sl];k=int(d['k'][i])
        h=np.sin((points[:,0]-lat)/2)**2+np.cos(lat)*np.cos(points[:,0])*np.sin((points[:,1]-lon)/2)**2
        geo_order=np.argsort(h);g=geo_order[:k]
        geographic_margin=float(h[geo_order[k]]-h[geo_order[k-1]])
        east=6371*np.cos(lat)*((points[:,1]-lon+np.pi)%(2*np.pi)-np.pi);north=6371*(points[:,0]-lat)
        selections=[]
        for model in range(4):
            order=np.argsort(-c['similarities'][model,sl]);sel=order[:k];selections.append(sel)
            scores=c['similarities'][model,sl]
            rows.append(dict(query=i,model=meta[model]['name'],group=int(d['group'][i]),latitude=float(np.degrees(lat)),longitude=float(np.degrees(lon)),k=k,replaced_fraction=float(1-len(np.intersect1d(g,sel))/k),embedding_elongation=elongation(east[sel],north[sel])[0],geographic_elongation=elongation(east[g],north[g])[0],selection_margin=float(scores[order[k-1]]-scores[order[k]]),geographic_margin=geographic_margin))
        if d['group'][i]==-1 and np.isclose(np.degrees(lat),60):panels[i]=(east,north,g,selections)
    summaries=[]
    for model in meta:
        for group in [-1,0,1,2,3]:
            r=[x for x in rows if x['model']==model['name'] and x['group']==group]
            summaries.append(dict(model=model['name'],group=group,queries=len(r),median_replaced_fraction=float(np.median([x['replaced_fraction'] for x in r])),median_elongation=float(np.median([x['embedding_elongation'] for x in r])),minimum_selection_margin=min(x['selection_margin'] for x in r),exact_embedding_boundary_ties=sum(x['selection_margin']==0 for x in r),unambiguous_queries=sum(x['selection_margin']>1e-12 and x['geographic_margin']>0 for x in r),median_replaced_unambiguous=float(np.median([x['replaced_fraction'] for x in r if x['selection_margin']>1e-12 and x['geographic_margin']>0]))))
    changes=[]
    for i in range(len(d['query_radians'])):
        sl=slice(d['offsets'][i],d['offsets'][i+1]);k=int(d['k'][i])
        for model in [0,2]:
            a=np.argsort(-c['similarities'][model,sl])[:k];b=np.argsort(-c['similarities'][model+1,sl])[:k]
            changes.append(dict(query=i,degree=meta[model]['degree'],group=int(d['group'][i]),changed_fraction=float(1-len(np.intersect1d(a,b))/k)))
    for degree,base in [(10,0),(40,2)]:
        fig,axes=plt.subplots(2,4,figsize=(13,7.8),sharex=True,sharey=True)
        for col,(i,(east,north,g,selections)) in enumerate(panels.items()):
            for row in range(2):
                ax=axes[row,col];sel=selections[base+row];fraction=1-len(np.intersect1d(g,sel))/len(g)
                ax.scatter(east,north,s=4,c='0.86');ax.scatter(east[g],north[g],s=25,facecolors='none',edgecolors='#0072B2',label='Nearest geographically');ax.scatter(east[sel],north[sel],s=15,c='#D55E00',marker='x',label='Nearest in embedding');ax.plot(0,0,'k+',ms=8)
                ax.set_title(f"{np.degrees(d['query_radians'][i,1]):.0f}°E — {fraction:.1%} differ",fontsize=10)
                ax.set_aspect('equal');ax.set_xlim(-6.2,6.2);ax.set_ylim(-6.2,6.2);ax.grid(alpha=.15)
                if row==1:ax.set_xlabel('East (km)')
                if col==0:ax.set_ylabel(('Released' if row==0 else 'Corrected input')+'\nNorth (km)')
        fig.suptitle(f'SatCLIP ViT L{degree}: 40 nearest neighbours at 60°N',fontsize=14)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
        fig.subplots_adjust(left=.065,right=.985,top=.89,bottom=.13,hspace=.27,wspace=.15)
        fig.savefig(output/f'satclip_L{degree}_neighbourhoods.png',dpi=220,bbox_inches='tight');plt.close(fig)
    result=dict(metadata=meta,summaries=summaries,queries=rows,correction_changes=changes,protocol=str(c['protocol']),scope='Descriptive neighbourhood comparisons; no predictive evaluation or inference about training. 20 controlled queries plus 1,200 selected gallery queries. All variants use the same candidates and k; finite neighbour geometry is not an infinitesimal accuracy test.')
    (output/'satclip_neighbours.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=Path('data'));parser.add_argument('--output',type=Path,default=Path('outputs/case_studies'));parser.add_argument('--models',type=Path,default=Path('models'));parser.add_argument('--collect',action='store_true');args=parser.parse_args()
    source=collect(args.data,args.models,args.output/'measurements') if args.collect else None
    run(args.data,args.output,source=source)

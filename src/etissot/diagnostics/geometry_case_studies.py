"""Matched local/global geometry studies from pinned encoders.

Fresh: python -m etissot.diagnostics.geometry_case_studies --collect
Replay: python -m etissot.diagnostics.geometry_case_studies
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr, skew
from scipy.spatial.distance import cdist
from etissot.geometry.probing import destination_point, probe_points
from etissot.geometry.fitting import fit_ellipse, fitted_tensor


def neighbour_metrics(reference, observed, k):
    """Overlap and rank severity; deterministic index ordering resolves ties."""
    a=np.argsort(reference,kind='stable');b=np.argsort(observed,kind='stable')
    rank=np.empty(len(a),int);rank[a]=np.arange(1,len(a)+1)
    return dict(overlap=float(len(np.intersect1d(a[:k],b[:k]))/k),mean_reference_rank=float(rank[b[:k]].mean()),mean_excess_rank=float(np.maximum(rank[b[:k]]-k,0).mean()),spearman=float(spearmanr(reference,observed).statistic))


def covariance_summary(features):
    z=features-features.mean(axis=0)
    eigen=np.linalg.svd(z,compute_uv=False)**2
    weights=eigen/eigen.sum();nz=weights[weights>0]
    return dict(top_variance_share=float(weights[0]),effective_rank=float(np.exp(-np.sum(nz*np.log(nz)))),mean_pair_cosine=float((np.sum(features.sum(axis=0)**2)-np.sum(features**2))/(len(features)*(len(features)-1))))


def make_design(data):
    with np.load(data/'measurements/neighbour_candidates.npz') as c:
        queries=c['query_radians'][c['group']==-1]
    rng=np.random.default_rng(20260918)
    fractions=np.sqrt(rng.uniform(size=(20,400)));angles=rng.uniform(0,2*np.pi,size=(20,400))
    radii=np.array([.3,1.,3.,6.]);fit_radii=np.array([.1,.3,1.])
    local=[];displacements=[]
    for radius in radii:
        rings=[];delta=[]
        for i,(lat,lon) in enumerate(queries):
            la,lo=destination_point(lat,lon,angles[i],radius*fractions[i]);rings.append(np.stack([la,lo],axis=1))
            delta.append(radius*fractions[i,:,None]*np.stack([np.cos(angles[i]),np.sin(angles[i])],axis=1))
        local.append(rings);displacements.append(delta)
    profiles=[]
    for radius in fit_radii:
        profiles.append([np.stack(probe_points(lat,lon,radius)[:2],axis=1) for lat,lon in queries])
    # Equal-area latitude spacing on the globe, restricted to the audit's +/-85 degrees.
    i=np.arange(512);z=(1-2*(i+.5)/512)*np.sin(np.radians(85));lon=(i*np.pi*(3-np.sqrt(5))+np.pi)%(2*np.pi)-np.pi
    globe=np.stack([np.arcsin(z),lon],axis=1)
    global_probes=np.array([np.stack(probe_points(lat,lon,1.)[:2],axis=1) for lat,lon in globe])
    return dict(queries=queries,radii=radii,fit_radii=fit_radii,candidates=np.array(local),displacements=np.array(displacements),probes=np.array(profiles),global_sites=globe,global_probes=global_probes)


def encoders(models,data):
    import torch,types,importlib,os
    from etissot.models.loaders import package_root
    from etissot.models.satclip_correction import correct_zonal_normalization
    expected=json.loads((data/'reference/summaries/checkpoint_manifest.json').read_text());hashes={Path(v['checkpoint']).name:v['sha256'] for v in expected.values()}
    root=package_root('geoclip');path=root/'model/weights/location_encoder_weights.pth'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if hashes.get(path.name)!=digest:raise ValueError('GeoCLIP checkpoint hash mismatch')
    pkg=types.ModuleType('geometry_geo');pkg.__path__=[str(root/'model')];sys.modules['geometry_geo']=pkg
    geo=importlib.import_module('geometry_geo.location_encoder').LocationEncoder().double().eval()
    yield 'GeoCLIP',lambda latlon:geo(latlon),dict(sha256=digest,precision='float64 input/weights; released float32 output accumulator')
    vendor=Path(__file__).resolve().parents[1]/'models/vendor/satclip/satclip'
    from etissot.models.loaders import load_satclip_checkpoint
    for degree in [10,40]:
        paths=list(models.rglob(f'satclip-vit16-l{degree}.ckpt'));assert len(paths)==1
        path=paths[0];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes.get(path.name)!=digest:raise ValueError('SatCLIP checkpoint hash mismatch')
        model=load_satclip_checkpoint(path,'cpu').double().eval()
        for corrected in [False,True]:
            net=correct_zonal_normalization(model) if corrected else model
            yield f'SatCLIP L{degree} '+('corrected' if corrected else 'released'),lambda latlon,net=net:net(latlon[:,[1,0]]),dict(sha256=digest,precision='float64',fixed_weight_correction=corrected)


def collect(data,models,output=None):
    from etissot.workspace import Workspace
    data = Path(data).resolve()
    destination_directory = Workspace(data, output or data.parent/'outputs/measurements').output
    destination_directory.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(2);design=make_design(data)
    fields=['queries','candidates','probes','global_sites','global_probes'];sizes=[design[k].reshape(-1,2).shape[0] for k in fields]
    positions=np.concatenate([design[k].reshape(-1,2) for k in fields]);positions[:,1]=(positions[:,1]+np.pi)%(2*np.pi)-np.pi
    saved={k:v for k,v in design.items() if k not in ['probes','global_probes']};metadata=[]
    for number,(name,encoder,meta) in enumerate(encoders(models,data)):
        output=[]
        for start in range(0,len(positions),1024):
            x=torch.tensor(np.degrees(positions[start:start+1024]),dtype=torch.float64)
            with torch.no_grad():y=torch.nn.functional.normalize(encoder(x),dim=1).double().numpy()
            if not np.isfinite(y).all():raise ValueError('Invalid features')
            output.append(y)
        all_features=np.concatenate(output);parts=np.split(all_features,np.cumsum(sizes)[:-1]);q,c,p,g,gp=parts
        c=c.reshape(4,20,400,-1);p=p.reshape(3,20,33,-1);gp=gp.reshape(512,33,-1)
        saved[f'distances_{number}']=np.linalg.norm(c-q[None,:,None,:],axis=-1)
        saved[f'profiles_{number}']=np.linalg.norm(p[:,:,1:]-p[:,:,:1],axis=-1)/design['fit_radii'][:,None,None]
        saved[f'global_features_{number}']=g
        saved[f'global_profiles_{number}']=np.linalg.norm(gp[:,1:]-gp[:,:1],axis=-1)
        metadata.append(dict(name=name,**meta));print(name,'collected',flush=True)
    saved['metadata']=json.dumps(metadata);saved['design']='seed 20260918; identical 400 candidates per query/radius/model; same 20 centres as neighbour case study; independently fitted 32-bearing tensors; equal-area 512-site globe clipped to +/-85 degrees'
    path=destination_directory/'geometry_case_studies.npz';np.savez_compressed(path,**saved);return path


def run(data,output,source=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    with np.load(source if source is not None else data/'measurements/geometry_case_studies.npz') as c:d={k:c[k] for k in c.files}
    metadata=json.loads(str(d['metadata']));bearings=np.arange(32)*2*np.pi/32;local=[];global_rows=[];hubrows=[]
    sites=d['global_sites']
    admissible=np.array([np.all(np.abs(probe_points(lat,lon,1.)[1])<=np.pi) for lat,lon in sites])
    xyz=np.stack([np.cos(sites[:,0])*np.cos(sites[:,1]),np.cos(sites[:,0])*np.sin(sites[:,1]),np.sin(sites[:,0])],axis=1)
    geographic=cdist(xyz,xyz);np.fill_diagonal(geographic,np.inf)
    for number,meta in enumerate(metadata):
        for fi,fit_radius in enumerate(d['fit_radii']):
            fits=[fit_ellipse(p,bearings) for p in d[f'profiles_{number}'][fi]]
            for ri,radius in enumerate(d['radii']):
                for qi,fit in enumerate(fits):
                    if fit['fallback']:continue
                    delta=d['displacements'][ri,qi];tensor=fitted_tensor(fit)
                    predicted=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',delta,tensor,delta),0))
                    actual=d[f'distances_{number}'][ri,qi];physical=np.linalg.norm(delta,axis=1)
                    for k in [10,20,40,80]:
                        for label,reference in [('geographic',physical),('tensor',predicted)]:
                            local.append(dict(model=meta['name'],fit_radius=float(fit_radius),radius=float(radius),query=qi,k=k,predictor=label,**neighbour_metrics(reference,actual,k)))
        features=d[f'global_features_{number}'];fits=[fit_ellipse(p,bearings) for p in d[f'global_profiles_{number}']]
        fit_valid=np.array([not f['fallback'] for f in fits]);valid=fit_valid & admissible;anisotropy=np.array([f['A_fit'] for f in fits])
        global_rows.append(dict(model=meta['name'],**covariance_summary(features),median_local_A=float(np.median(anisotropy[valid])),p95_local_A=float(np.quantile(anisotropy[valid],.95)),fallback_sites=int((~fit_valid).sum()),coordinate_boundary_exclusions=int((~admissible).sum()),local_valid_sites=int(valid.sum()),sites=512))
        distances=cdist(features,features);np.fill_diagonal(distances,np.inf)
        for k in [10,20,40]:
            for label,matrix in [('embedding',distances),('geographic',geographic)]:
                order=np.argsort(matrix,axis=1,kind='stable')[:,:k];counts=np.bincount(order.ravel(),minlength=512)
                eligible=valid & (label=='embedding')
                association=float(spearmanr(counts[eligible],anisotropy[eligible]).statistic) if eligible.any() else None
                hubrows.append(dict(model=meta['name'],space=label,k=k,max_occurrences=int(counts.max()),max_over_mean=float(counts.max()/k),skewness=float(skew(counts)),occurrences=counts.tolist(),spearman_with_local_A=association))
    result=dict(metadata=metadata,local=local,global_geometry=global_rows,hubness=hubrows,scope='Descriptive matched experiments; correction is fixed-weight. Local reference tensors use independent bearings, excluding fallback fits. Global sample includes land and ocean with equal-area sampling. Global spectrum is extrinsic variance, not intrinsic dimension; hubness correlations are descriptive, not causal.')
    (output/'geometry_case_studies.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for meta in metadata:
        name=meta['name'];values=[]
        for radius in d['radii']:
            a=[x['overlap'] for x in local if x['model']==name and x['fit_radius']==.3 and x['radius']==radius and x['k']==40 and x['predictor']=='tensor']
            b=[x['overlap'] for x in local if x['model']==name and x['fit_radius']==.3 and x['radius']==radius and x['k']==40 and x['predictor']=='geographic']
            values.append(float(np.median(np.array(a)-np.array(b))))
        axes[0].plot(d['radii'],values,'o-',label=name)
        g=next(x for x in global_rows if x['model']==name)
        axes[1].scatter(g['top_variance_share'],g['median_local_A'],label=name,s=45)
    axes[0].axhline(0,color='0.6',ls=':');axes[0].set(xscale='log',xlabel='Candidate neighbourhood radius (km)',ylabel='Median gain in neighbour overlap',title='Does the local tensor explain selection?')
    axes[0].set_xticks([.3,1,3,6], labels=['0.3','1','3','6'])
    from matplotlib.ticker import NullLocator
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[1].set(xlabel='Largest share of global feature variance',ylabel='Median local geographic anisotropy',title='Global concentration versus local stretch')
    for ax in axes:ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.14,1,1));fig.savefig(output/'geometry_comparison.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for meta in metadata:
        name=meta['name']
        values=[np.median([x['mean_excess_rank'] for x in local if x['model']==name and x['fit_radius']==.3 and x['radius']==1 and x['k']==k and x['predictor']=='geographic']) for k in [10,20,40,80]]
        axes[0].plot([10,20,40,80],values,'o-',label=name)
        h=next(x for x in hubrows if x['model']==name and x['k']==20 and x['space']=='embedding')
        counts=np.sort(h['occurrences'])/20
        axes[1].plot(np.arange(512)/511,counts,label=name)
    h=next(x for x in hubrows if x['k']==20 and x['space']=='geographic')
    axes[1].plot(np.arange(512)/511,np.sort(h['occurrences'])/20,'k--',label='Geographic baseline')
    axes[0].set(xlabel='Number of neighbours k',ylabel='Median mean excess geographic rank',title='How far do selected neighbours move?')
    axes[1].set(xlabel='Cumulative fraction of candidate locations',ylabel='Neighbour occurrences / mean',title='Unequal popularity in neighbour lists')
    for ax in axes:ax.grid(alpha=.2)
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,frameon=False)
    fig.tight_layout(rect=(0,.14,1,1));fig.savefig(output/'rank_hubness.png',dpi=200);plt.close(fig)
    print(json.dumps(global_rows,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collect',action='store_true');p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--models',type=Path,default=Path('models'));p.add_argument('--output',type=Path,default=Path('outputs/case_studies'));a=p.parse_args()
    source=collect(a.data,a.models,a.output/'measurements') if a.collect else None
    run(a.data,a.output,source=source)

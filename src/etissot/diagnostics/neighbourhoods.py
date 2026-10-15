"""Matched neighbourhood comparison across all audited encoder checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from etissot.diagnostics.geometry_case_studies import make_design, neighbour_metrics
from etissot.geometry.fitting import fit_ellipse, fitted_tensor


def collect(data, models, output=None):
    from etissot.workspace import Workspace
    data = Path(data).resolve()
    destination_directory = Workspace(data, output or data.parent/'outputs/measurements').output
    destination_directory.mkdir(parents=True, exist_ok=True)
    import torch
    from etissot.models.loaders import load_all
    torch.set_num_threads(2)
    design=make_design(data)
    fields=['queries','candidates','probes']
    sizes=[design[k].reshape(-1,2).shape[0] for k in fields]
    positions=np.concatenate([design[k].reshape(-1,2) for k in fields])
    saved={k:design[k] for k in ['queries','radii','fit_radii','displacements']}
    metadata=[]
    # Reuse precisely the same independently measured distances and probes.
    with np.load(data/'measurements/geometry_case_studies.npz') as old:
        for i,meta in enumerate(json.loads(str(old['metadata']))):
            saved[f'distances_{i}']=old[f'distances_{i}']
            saved[f'profiles_{i}']=old[f'profiles_{i}']
            metadata.append(dict(meta,normalization='L2'))
        for key in ['queries','radii','fit_radii','displacements']:
            np.testing.assert_array_equal(saved[key],old[key])
    encoders,meta=load_all(models, data/'reference/summaries/checkpoint_manifest.json')
    expected=json.loads((data/'reference/summaries/checkpoint_manifest.json').read_text())
    for name,encoder in encoders.items():
        if name in ['GeoCLIP','SatCLIP vit16l10','SatCLIP vit16l40']:continue
        digest=hashlib.sha256(Path(meta[name]['checkpoint']).read_bytes()).hexdigest()
        if digest!=expected[name]['sha256']:raise ValueError(f'Checkpoint mismatch: {name}')
        output=[]
        for start in range(0,len(positions),512):
            x=torch.tensor(positions[start:start+512],dtype=torch.float64)
            with torch.no_grad():y=encoder(x[:,0],x[:,1]).double().numpy()
            if not np.isfinite(y).all():raise ValueError(f'Invalid features: {name}')
            output.append(y)
        q,c,p=np.split(np.concatenate(output),np.cumsum(sizes)[:-1])
        c=c.reshape(4,20,400,-1);p=p.reshape(3,20,33,-1)
        i=len(metadata)
        saved[f'distances_{i}']=np.linalg.norm(c-q[None,:,None,:],axis=-1)
        saved[f'profiles_{i}']=np.linalg.norm(p[:,:,1:]-p[:,:,:1],axis=-1)/design['fit_radii'][:,None,None]
        metadata.append(dict(name=name,**meta[name],sha256=digest))
        print(name,'collected',flush=True)
    saved['metadata']=json.dumps(metadata)
    destination = destination_directory/'neighbourhoods.npz'
    np.savez_compressed(destination,**saved)
    return destination


def selected_neighbours(data, model_index, query_index, radius_index=1, fit_index=1, k=40):
    """Return independently selected geographic, embedding and tensor sets."""
    delta=data['displacements'][radius_index,query_index]
    fit=fit_ellipse(data[f'profiles_{model_index}'][fit_index,query_index],np.arange(32)*2*np.pi/32)
    physical=np.linalg.norm(delta,axis=1)
    actual=data[f'distances_{model_index}'][radius_index,query_index]
    predicted=None if fit['fallback'] else np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',delta,fitted_tensor(fit),delta),0))
    order=lambda x:np.argsort(x,kind='stable')[:k]
    return delta,order(physical),order(actual),None if predicted is None else order(predicted)


def tensor_neighbourhood_boundary(tensor, distance, samples=361):
    """North/east displacements at a fixed tensor-predicted feature distance."""
    tensor = np.asarray(tensor, dtype=float)
    if (tensor.shape != (2, 2) or not np.isfinite(tensor).all()
            or not np.allclose(tensor, tensor.T, rtol=1e-12, atol=0)
            or np.linalg.eigvalsh(tensor)[0] <= 0):
        raise ValueError('A finite symmetric positive-definite two-by-two tensor is required')
    if not isinstance(samples, (int, np.integer)) or isinstance(samples, (bool, np.bool_)) or samples < 4:
        raise ValueError('At least four boundary samples are required')
    theta=np.linspace(0,2*np.pi,samples)
    directions=np.column_stack([np.cos(theta),np.sin(theta)])
    squared_stretch=np.einsum('ni,ij,nj->n',directions,tensor,directions)
    if distance<0 or not np.isfinite(distance) or np.any(squared_stretch<=0):
        raise ValueError('A finite nonnegative distance and positive-definite tensor are required')
    return directions*(distance/np.sqrt(squared_stretch))[:,None]


def pairwise_overlaps(data, metadata):
    """All three overlaps on exactly the same valid primary comparisons."""
    values={'E/G':[],'T/G':[],'E/T':[]}
    for mi in range(len(metadata)):
        for qi in range(len(data['queries'])):
            _,g,e,t=selected_neighbours(data,mi,qi)
            if t is None:continue
            for key,a,b in [('E/G',e,g),('T/G',t,g),('E/T',e,t)]:
                values[key].append(len(np.intersect1d(a,b))/40)
    return values


def display_name(name):
    return name.replace('SatCLIP L','SatCLIP ViT L').replace('r18l','ResNet18 L').replace('r50l','ResNet50 L')


def plot_neighbourhood_maps(data, metadata, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    legends=[Line2D([],[],marker='o',ls='',mfc='none',mec='#2166ac',markersize=8,label='Geographic neighbours'),
             Line2D([],[],marker='o',ls='',color='#e68a28',markersize=5,label='Actual embedding neighbours'),
             Line2D([],[],marker='x',ls='',color='#16835c',markersize=7,label='Tensor-predicted neighbours'),
             Line2D([],[],marker='*',ls='',color='black',markersize=10,label='Query location')]
    def panel(ax,mi,qi,small=False):
        delta,g,e,t=selected_neighbours(data,mi,qi)
        east,north=delta[:,1],delta[:,0]
        ax.scatter(east,north,s=3 if small else 6,c='0.84',zorder=1)
        geographic_radius=float(np.linalg.norm(delta[g],axis=1).max())
        ax.add_patch(Circle((0,0),geographic_radius,fill=False,edgecolor='#2166ac',linestyle='--',linewidth=1.2,zorder=2))
        if not small and t is not None:
            fit=fit_ellipse(data[f'profiles_{mi}'][1,qi],np.arange(32)*2*np.pi/32)
            threshold=float(np.sort(data[f'distances_{mi}'][1,qi],kind='stable')[39])
            boundary=tensor_neighbourhood_boundary(fitted_tensor(fit),threshold)
            ax.plot(boundary[:,1],boundary[:,0],color='#e68a28',linewidth=1.3,zorder=2)
        ax.scatter(east[g],north[g],s=23 if small else 55,facecolors='none',edgecolors='#2166ac',lw=.7,zorder=2)
        ax.scatter(east[e],north[e],s=8 if small else 18,c='#e68a28',zorder=3)
        if t is not None:ax.scatter(east[t],north[t],s=16 if small else 32,c='#16835c',marker='x',linewidths=.7,zorder=4)
        ax.scatter([0],[0],marker='*',s=35 if small else 75,c='black',zorder=5)
        ax.set(xlim=(-1.02,1.02),ylim=(-1.02,1.02),aspect='equal')
        ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.tick_params(labelsize=7 if small else 9);ax.grid(alpha=.15)
        eg=len(np.intersect1d(g,e))/40
        tg=None if t is None else len(np.intersect1d(g,t))/40
        et=None if t is None else len(np.intersect1d(e,t))/40
        label=f'E/G {eg:.1%}  T/G {tg:.1%}  E/T {et:.1%}' if t is not None else f'E/G {eg:.1%}; tensor fit flagged'
        if not small and t is not None:label=f'E/G {eg:.1%}   E/T {et:.1%}'
        ax.text(.5,.025,label,transform=ax.transAxes,ha='center',fontsize=6 if small else 8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    queries=np.degrees(data['queries'])
    # Deliberately illustrative examples spanning observed accuracy, not typical sites.
    chosen=[('GeoCLIP',0,0),('SatCLIP L40 released',70,0),('SINR distilled',60,0),('CSP iNat2018',0,170),('Climplicit',60,60)]
    names=[m['name'] for m in metadata]
    fig,axes=plt.subplots(3,2,figsize=(8,10))
    for ax,(name,lat,lon) in zip(axes.ravel(),chosen):
        qi=int(np.flatnonzero(np.all(np.isclose(queries,[lat,lon]),axis=1))[0])
        panel(ax,names.index(name),qi)
        ax.set_title(display_name(name)+f'\n{lat}°N, {lon}°E',fontsize=10)
        ax.set_xlabel('East displacement (km)',fontsize=9);ax.set_ylabel('North displacement (km)',fontsize=9)
    pairs=pairwise_overlaps(data,metadata)
    slot=axes.ravel()[-1].get_subplotspec();axes.ravel()[-1].remove()
    grid=slot.subgridspec(3,1,hspace=.75)
    for row,(key,title,color) in enumerate([('E/G','Embedding vs geographic','#e68a28'),('T/G','Tensor vs geographic','#2166ac'),('E/T','Embedding vs tensor','#16835c')]):
        ax=fig.add_subplot(grid[row,0])
        shared=np.rint(np.asarray(pairs[key])*40).astype(int)
        counts=[int((shared<35).sum())]+[int((shared==n).sum()) for n in range(35,41)]
        ax.bar(np.arange(7),counts,color=color,width=.72)
        for x,count in enumerate(counts):
            if count:ax.text(x,count+6,str(count),ha='center',fontsize=7)
        ax.set(xlim=(-.6,6.6),ylim=(0,300),ylabel='Count')
        ax.set_title(title,fontsize=9,loc='left',pad=3)
        ax.set_xticks(np.arange(7),labels=['<35','35','36','37','38','39','40'])
        ax.set_yticks([0,150,300]);ax.tick_params(labelsize=8)
        if row<2:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel('Shared neighbours (out of 40)',fontsize=9)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.legend(handles=legends,loc='lower center',ncol=2,fontsize=10,frameon=False)
    fig.tight_layout(rect=(0,.065,1,1));fig.savefig(output/'neighbourhood_maps.png',dpi=220);plt.close(fig)
    # All variants at the same four longitudes: no selection by agreement.
    ordered=sorted(range(len(metadata)),key=lambda i: (['GeoCLIP','SatCLIP','SINR','CSP','Climplicit'].index(names[i].split()[0]),'corrected' in names[i]))
    for page in range(3):
        fig,axes=plt.subplots(5,4,figsize=(9,11))
        for row,mi in enumerate(ordered[page*5:(page+1)*5]):
            for col,lon in enumerate([0,60,120,170]):
                qi=int(np.flatnonzero(np.all(np.isclose(queries,[60,lon]),axis=1))[0])
                ax=axes[row,col];panel(ax,mi,qi,small=True)
                if row==0:ax.set_title(f'60°N, {lon}°E',fontsize=9)
                if col==0:ax.set_ylabel(display_name(names[mi])+'\nNorth (km)',fontsize=8)
                if row==4:ax.set_xlabel('East (km)',fontsize=8)
        fig.legend(handles=legends,loc='lower center',ncol=2,fontsize=9,frameon=False)
        fig.tight_layout(rect=(0,.055,1,1));fig.savefig(output/f'neighbourhood_maps_detail_{page+1}.png',dpi=220);plt.close(fig)


def run(data, output, source=None):
    output.mkdir(parents=True,exist_ok=True)
    with np.load(source if source is not None else data/'measurements/neighbourhoods.npz') as cache:d={k:cache[k] for k in cache.files}
    metadata=json.loads(str(d['metadata']));rows=[];validity=[]
    for i,meta in enumerate(metadata):
        for fi,fr in enumerate(d['fit_radii']):
            fits=[fit_ellipse(p,np.arange(32)*2*np.pi/32) for p in d[f'profiles_{i}'][fi]]
            validity.append(dict(model=meta['name'],fit_radius=float(fr),valid_queries=sum(not f['fallback'] for f in fits),total_queries=len(fits)))
            for ri,radius in enumerate(d['radii']):
                for qi,fit in enumerate(fits):
                    if fit['fallback']:continue
                    delta=d['displacements'][ri,qi]
                    pred=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',delta,fitted_tensor(fit),delta),0))
                    actual=d[f'distances_{i}'][ri,qi]
                    for k in [10,20,40,80]:
                        order=np.sort(actual,kind='stable')
                        gap=float(order[k]-order[k-1])
                        for label,reference in [('geographic',np.linalg.norm(delta,axis=1)),('tensor',pred)]:
                            rows.append(dict(model=meta['name'],fit_radius=float(fr),radius=float(radius),query=qi,k=k,predictor=label,boundary_gap=gap,**neighbour_metrics(reference,actual,k)))
    summary=[]
    for meta in metadata:
        selected=[r for r in rows if r['model']==meta['name'] and r['fit_radius']==.3 and r['radius']==1 and r['k']==40]
        a=np.array([r['overlap'] for r in selected if r['predictor']=='geographic'])
        b=np.array([r['overlap'] for r in selected if r['predictor']=='tensor'])
        summary.append(dict(model=meta['name'],geographic_overlap=float(np.median(a)),tensor_overlap=float(np.median(b)),median_paired_gain=float(np.median(b-a)),queries=len(a),improved_queries=int((b>a).sum()),tied_queries=int((b==a).sum()),worse_queries=int((b<a).sum()),boundary_ties=sum(r['boundary_gap']==0 for r in selected if r['predictor']=='tensor')))
    for i,item in enumerate(summary):
        overlaps=[]
        for qi in range(len(d['queries'])):
            _,g,e,t=selected_neighbours(d,i,qi)
            if t is not None:overlaps.append(len(np.intersect1d(g,t))/40)
        item['tensor_geographic_overlap']=float(np.median(overlaps))
    primary=[r['overlap'] for r in rows if r['fit_radius']==.3 and r['radius']==1 and r['k']==40 and r['predictor']=='tensor']
    levels,counts=np.unique(primary,return_counts=True)
    distribution=dict(total=len(primary),exact=sum(x==1 for x in primary),nonexact=sum(x<1 for x in primary),levels=levels.tolist(),counts=counts.tolist())
    pair_distributions={}
    for key,values in pairwise_overlaps(d,metadata).items():
        levels,counts=np.unique(values,return_counts=True)
        pair_distributions[key]=dict(total=len(values),levels=levels.tolist(),counts=counts.tolist(),median=float(np.median(values)),exact=sum(v==1 for v in values))
    result=dict(metadata=metadata,summary=summary,validity=validity,local=rows,overlap_distribution=distribution,pair_distributions=pair_distributions)
    (output/'neighbourhoods.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    summary=sorted(summary,key=lambda r: (['GeoCLIP','SatCLIP','SINR','CSP','Climplicit'].index(r['model'].split()[0]),'corrected' in r['model']))
    y=np.arange(len(summary))
    ax.barh(y-.17,[r['geographic_overlap'] for r in summary],height=.32,label='Actual embedding neighbours',color='#e68a28')
    ax.barh(y+.17,[r['tensor_geographic_overlap'] for r in summary],height=.32,label='Tensor-predicted neighbours',color='#16835c')
    ax.set_yticks(y,labels=[r['model'].replace('SatCLIP L','SatCLIP ViT L').replace('r18l','ResNet18 L').replace('r50l','ResNet50 L') for r in summary]);ax.invert_yaxis()
    ax.set(xlim=(0,1.02),xlabel='Median overlap with geographic neighbours')
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,frameon=False);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(output/'neighbourhood_comparison.png',dpi=200);plt.close(fig)
    plot_neighbourhood_maps(d,metadata,output)
    print(json.dumps(summary,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collect',action='store_true');p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--models',type=Path,default=Path('models'));p.add_argument('--output',type=Path,default=Path('outputs/case_studies'));a=p.parse_args()
    source=collect(a.data,a.models,a.output/'measurements') if a.collect else None
    run(a.data,a.output,source=source)

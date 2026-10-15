"""Location-neighbour geometry from measured candidate similarities; optional fresh collection."""
import json
from pathlib import Path
import numpy as np
from etissot.geometry.probing import destination_point, R_EARTH
from etissot.geometry.references import equal_earth_metric


def elongation(east, north):
    values, vectors = np.linalg.eigh(np.cov(np.stack([east, north])))
    return float(np.sqrt(values[1] / max(values[0], 1e-12))), float(np.degrees(np.arctan2(vectors[0, 1], vectors[1, 1])) % 180)


def query_metrics(lat, lon, candidate_lat, candidate_lon, similarity, k):
    """Recompute selections and local covariance from candidate coordinates/scores."""
    east = R_EARTH*np.cos(lat)*((candidate_lon-lon+np.pi)%(2*np.pi)-np.pi)
    north = R_EARTH*(candidate_lat-lat)
    xyz = np.stack([np.cos(candidate_lat)*np.cos(candidate_lon), np.cos(candidate_lat)*np.sin(candidate_lon), np.sin(candidate_lat)],1)
    center = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    embedded = np.argsort(-similarity)[:k]
    geographic = np.argsort(np.linalg.norm(xyz-center,axis=1))[:k]
    ea, bearing = elongation(east[embedded],north[embedded])
    ga, _ = elongation(east[geographic],north[geographic])
    reference, theta = equal_earth_metric(np.array([np.degrees(lat)]),np.array([np.degrees(lon)]))
    predicted=(np.degrees(theta[0])+90)%180
    difference=(bearing-predicted)%180
    return dict(embedding_elongation=ea,geodesic_elongation=ga,axis_deviation_degrees=float(min(difference,180-difference)),reference_anisotropy=float(reference[0]))


def run(workspace):
    """Regenerate every neighbour number in main §5.2 and supplement S6."""
    with np.load(workspace.data/'measurements/neighbour_candidates.npz') as d:
        d=dict(d)
        rows=[]
        for i,(lat,lon) in enumerate(d['query_radians']):
            sl=slice(d['offsets'][i],d['offsets'][i+1])
            metric=query_metrics(lat,lon,d['candidate_radians'][sl,0],d['candidate_radians'][sl,1],d['similarity'][sl],int(d['k'][i]))
            rows.append(dict(latitude=float(np.degrees(lat)),longitude=float(np.degrees(lon)),group=int(d['group'][i]),**metric))
    synthetic=[r for r in rows if r['group']==-1]
    gallery=[]
    for g in range(4):
        q=[r for r in rows if r['group']==g]
        gallery.append(dict(group=g,n=len(q),embedding_median=float(np.median([r['embedding_elongation'] for r in q])),geodesic_median=float(np.median([r['geodesic_elongation'] for r in q])),more_elongated_share=float(np.mean([r['embedding_elongation']>r['geodesic_elongation']+1e-10*max(r['embedding_elongation'],r['geodesic_elongation']) for r in q]))))
    report=dict(comparison_relative_tolerance=1e-10,synthetic=synthetic,gallery=gallery,queries=rows,scope='20 synthetic queries total (5 latitudes x 4 longitudes); 300 gallery queries per latitude band; candidate similarities are saved model measurements')
    (workspace.output/'summaries/neighbours.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def collect(workspace, model_dir=None):
    """Replay the original seed/order with installed GeoCLIP weights and gallery.

    Saves candidate coordinates and similarities, not formatted manuscript results.
    No model/data download is performed by this function.
    """
    import torch, importlib, types, sys, hashlib
    from scipy.spatial import cKDTree
    from etissot.models.loaders import package_root
    torch.set_num_threads(2)
    root=package_root('geoclip');pkg=types.ModuleType('neighbour_geo');pkg.__path__=[str(root/'model')];sys.modules['neighbour_geo']=pkg
    encoder=importlib.import_module('neighbour_geo.location_encoder').LocationEncoder().double().eval()
    def encode(lat,lon):
        x=torch.tensor(np.stack([np.degrees(lat),(np.degrees(lon)+180)%360-180],1),dtype=torch.float64)
        out=[]
        with torch.no_grad():
            for start in range(0,len(x),16384):
                y=encoder(x[start:start+16384]);out.append((y/y.norm(dim=1,keepdim=True)).numpy())
        return np.concatenate(out)
    rng=np.random.default_rng(0);queries=[];points=[];scores=[];groups=[];ks=[];offsets=[0]
    def save(lat,lon,clat,clon,sim,k,group):
        queries.append([lat,lon]);points.append(np.stack([clat,clon],1));scores.append(sim);ks.append(k);groups.append(group);offsets.append(offsets[-1]+len(sim))
    for latd in (0,30,45,60,70):
        for lond in (0,60,120,170):
            lat,lon=np.radians([latd,lond]);r=6*np.sqrt(rng.uniform(0,1,400));b=rng.uniform(0,2*np.pi,400)
            clat,clon=destination_point(lat,lon,b,r)
            e=encode(np.r_[lat,clat],np.r_[lon,clon]);save(lat,lon,clat,clon,e[1:]@e[0],40,-1)
    gallery_path=root/'model/gps_gallery/coordinates_100K.csv'
    coords=np.loadtxt(gallery_path,delimiter=',',skiprows=1);lat,lon=np.radians(coords).T;e=encode(lat,lon)
    xyz=np.stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)],1);tree=cKDTree(xyz);chord=2*np.sin(15/(2*R_EARTH))
    for g,(lo,hi) in enumerate(((0,20),(20,40),(40,55),(55,70))):
        band=np.flatnonzero((abs(coords[:,0])>=lo)&(abs(coords[:,0])<hi));n=0
        for q in rng.permutation(band)[:4000]:
            nb=np.array([j for j in tree.query_ball_point(xyz[q],chord) if j!=q])
            if len(nb)<80:continue
            save(lat[q],lon[q],lat[nb],lon[nb],e[nb]@e[q],30,g);n+=1
            if n==300:break
        if n!=300:raise ValueError(f'Expected 300 qualifying gallery queries, got {n}')
        print('Collected neighbour latitude band',lo,hi,flush=True)
    out=workspace.output/'measurements';out.mkdir(exist_ok=True)
    path=out/'neighbour_candidates.npz';np.savez_compressed(path,query_radians=queries,candidate_radians=np.concatenate(points),similarity=np.concatenate(scores),offsets=offsets,group=groups,k=ks)
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    (out/'neighbour_provenance.json').write_text(json.dumps(dict(seed=0,checkpoint_sha256=sha(root/'model/weights/location_encoder_weights.pth'),gallery_sha256=sha(gallery_path),synthetic_queries=20,gallery_queries=1200,candidate_count=offsets[-1],protocol='Original experiment-33 query ordering and precision; coordinates radians'),indent=2)+'\n')
    print('Saved',path,flush=True)

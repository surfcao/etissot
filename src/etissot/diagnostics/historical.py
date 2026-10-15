"""Recompute exploratory five-seed summaries from archived per-seed measurements."""
import json
import numpy as np
from etissot.statistics.spatial import GLAT, wcorr

def run(workspace):
    mask=abs(GLAT.ravel())<=60
    from scipy.special import eval_legendre
    from scipy.optimize import brentq
    degrees=np.arange(1,21);weights=(2*degrees+1)/(degrees+1)**2
    target_length=6371*brentq(lambda distance: np.sum(weights*eval_legendre(degrees,np.cos(distance)))/weights.sum()-np.exp(-1),.001,1.)
    records={}
    with np.load(workspace.data/'measurements/hidden_space_seeds.npz') as d:
        reference=np.log(d['input_anisotropy'].ravel());target=np.log(d['target_anisotropy'].ravel())
        for kind in ['rff16-mlp','geoclip-3cap','rff1-mlp','siren']:
            rows=[]
            for seed in range(5):
                x=np.log(d[f'A__{kind}__{seed}'].ravel())
                rows.append(dict(seed=seed,r2=float(d[f'r2__{kind}__{seed}']),input_correlation=float(wcorr(x,reference,mask=mask)),target_correlation=float(wcorr(x,target,mask=mask))))
            records[kind]=dict(seeds=rows,mean={key:float(np.mean([row[key] for row in rows])) for key in ['r2','input_correlation','target_correlation']},sd={key:float(np.std([row[key] for row in rows],ddof=1)) for key in ['r2','input_correlation','target_correlation']})
    (workspace.output/'summaries/hidden_space.json').write_text(json.dumps(dict(models=records,target_ensemble_correlation_length_km=target_length,scope='Reanalysis of saved per-seed angular-extrema fields and held-out R2; does not retrain miniature encoders'),indent=2)+'\n')
    return records

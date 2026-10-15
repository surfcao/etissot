"""Recompute dense tensors and regional correlation scales from directional profiles."""
import json
import numpy as np
from etissot.geometry.fitting import fit_many
from etissot.diagnostics.dense import PANELS

def run(workspace):
    report={};fields={};bear=np.arange(32)*2*np.pi/32
    with np.load(workspace.data/'profiles/dense_regional.npz') as data:
        for name,region,_,step,radius in PANELS:
            key=name+'__'+region;shape=(len(data['lat__'+key]),len(data['lon__'+key]))
            fit=fit_many(data['prof__'+key],bear);a=fit['A_fit'].reshape(shape)
            fields['A__'+key]=a;fields['B__'+key]=fit['theta_fit'].reshape(shape);fields['fallback__'+key]=fit['fallback'].reshape(shape)
            z=np.log(a);z-=z.mean(1,keepdims=True)
            acf=[float(np.mean([np.corrcoef(row[:-k],row[k:])[0,1] for row in z])) for k in range(1,z.shape[1]//2)]
            crossing=next((k+1 for k,c in enumerate(acf) if c<.5),None)
            report[key]=dict(sites=int(a.size),step_degrees=step,radius_km=radius,median_A=float(np.median(a)),fallback_count=int(fit['fallback'].sum()),row_mean_longitude_acf=acf,first_acf_below_half_degrees=None if crossing is None else crossing*step)
    np.savez_compressed(workspace.output/'profiles/dense_fitted.npz',**fields)
    (workspace.output/'summaries/dense_summary.json').write_text(json.dumps(report,indent=2)+'\n')

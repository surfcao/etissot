"""Conditional input/output concatenation sensitivity, with no power interpretation."""
import json
import numpy as np
from etissot.geometry.fitting import fit_many
from etissot.statistics.spatial import GLAT, wcorr

def run(workspace):
    bear=np.arange(32)*2*np.pi/32;mask=abs(GLAT.ravel())<=60
    alphas=np.array([0,.1,.2,.3,.4,.5,.6,.8,1.]);records={}
    with np.load(workspace.data/'profiles/global.npz') as out, np.load(workspace.data/'profiles/input_references.npz') as inp:
        for name in ['GeoCLIP','SatCLIP vit16l10','SatCLIP vit16l40','SINR cap1000','CSP fMoW','CSP iNat2018','Climplicit']:
            a=inp['in__'+name];b=out['prof__'+name]
            ca=np.median(np.exp(np.log(a).mean(1)));cb=np.median(np.exp(np.log(b).mean(1)))
            reference=np.log(fit_many(a,bear)['A_fit']);corr=[]
            for alpha in alphas:
                profile=np.sqrt((alpha*a/ca)**2+((1-alpha)*b/cb)**2)
                corr.append(float(wcorr(np.log(fit_many(profile,bear)['A_fit']),reference,mask=mask)))
            records[name]=dict(input_scale=float(ca),output_scale=float(cb),correlation=corr)
    (workspace.output/'summaries/mixtures.json').write_text(json.dumps(dict(alpha=alphas.tolist(),squared_coefficient=(alphas**2/(alphas**2+(1-alphas)**2)).tolist(),encoders=records,scope='Conditional concatenation curves, not energy shares, significance tests or detection power'),indent=2)+'\n')

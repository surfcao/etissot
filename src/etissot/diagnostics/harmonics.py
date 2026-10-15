"""Audit the vendored SatCLIP harmonic implementation against SciPy references."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import sph_harm_y, eval_legendre
from etissot.models.loaders import filemodule


def zonal_anisotropy(latitude, degree, multiplier=np.pi):
    phi=np.radians(latitude);theta=phi+np.pi/2;x=np.cos(theta);sn=np.sin(theta)
    isotropic=sum((2*l+1)*l*(l+1)/(8*np.pi) for l in range(degree));z=np.zeros_like(x)
    for l in range(1,degree):
        derivative=-l*(x*eval_legendre(l,x)-eval_legendre(l-1,x))/sn
        z+=(2*l+1)/(4*np.pi)*derivative**2
    q=1+(multiplier**2-1)*z/isotropic
    return np.sqrt(np.maximum(q,1/q))


def run(workspace):
    path=Path(__file__).resolve().parents[1]/'models/vendor/satclip/satclip/positional_encoding/spherical_harmonics_ylm.py'
    sh=filemodule('reproduction_harmonics',path).SH
    rng=np.random.default_rng(1);theta=torch.tensor(rng.uniform(.1,np.pi-.1,200));phi=torch.tensor(rng.uniform(0,2*np.pi,200))
    zonal=[]
    for degree in range(6):
        actual=(sh(0,degree,phi,theta)*torch.ones_like(theta)).numpy();expected=sph_harm_y(degree,0,theta.numpy(),phi.numpy()).real
        zonal.append(float(np.median(actual/expected)))
    np.testing.assert_allclose(zonal,np.pi,rtol=1e-12)
    theta=torch.tensor(np.radians([90.,60.,30.,20.,10.,5.]));phi=torch.ones_like(theta);errors={}
    for degree in [5,10,15,20,25,30,35,39]:
        maximum=np.zeros(len(theta))
        for m in range(-degree,degree+1):
            y=sph_harm_y(degree,abs(m),theta.numpy(),phi.numpy())
            expected=y.real if m==0 else np.sqrt(2)*(-1)**m*(y.real if m>0 else y.imag)
            if m==0:expected*=np.pi
            actual=(sh(m,degree,phi,theta)*torch.ones_like(theta)).numpy()
            # Historical code has a different nonzonal real-harmonic sign convention.
            if m!=0:expected=abs(expected)*np.sign(actual+1e-300)
            maximum=np.maximum(maximum,abs(actual-expected))
        errors[str(degree)]=maximum.tolist()
    a=zonal_anisotropy(np.arange(-85,85.01,2.5),10)
    report=dict(zonal_multiplier_by_degree=zonal,degree10_anisotropy_range=[float(a.min()),float(a.max())],latitude_degrees=[0,-30,-60,-70,-80,-85],maximum_basis_error_by_degree=errors)
    (workspace.output/'summaries/harmonics.json').write_text(json.dumps(report,indent=2)+'\n')

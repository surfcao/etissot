"""Estimator-specific finite-radius checks for revision 16; NumPy only."""
from pathlib import Path
import json
import numpy as np
from etissot.geometry.fitting import fit_ellipse, refine_radius, fitted_tensor
from etissot.geometry.probing import local_distortion
from etissot.geometry.probing import R_EARTH

def grid_band(lat, lon, timescale):
    return np.array([np.sin(lon/timescale),np.cos(lon/timescale),np.sin(lat/timescale),np.cos(lat/timescale)])

def wavelength_km(timescale):
    return 2*np.pi*timescale*R_EARTH

b=np.arange(32)*2*np.pi/32


def sinusoid(a, rotation, r):
    t=b-rotation
    return np.sqrt(np.sin(np.pi*a*r*np.cos(t))**2+np.sin(np.pi*r*np.sin(t))**2)/(np.pi*r)


def run(workspace):
    rows=[]
    for a in [1.5,2.,3.,5.,10.,20.,50.]:
        for rotation in np.linspace(0,np.pi/2,17,endpoint=False):
            initial=1/(16*a)
            f=fit_ellipse(sinusoid(a,rotation,initial),b)
            refined=refine_radius(lambda r:sinusoid(a,rotation,r),initial,b)
            error=abs(refined['fit']['A_fit']/a-1)
            assert refined['converged'] and error<0.001
            # Leading coefficient for an axis-aligned sinusoidal map; test in
            # the asymptotic regime, where the minor-eigenvalue perturbation is small.
            if rotation==0:
                k=256*a*a
                got=fit_ellipse(sinusoid(a,0,1/k),b)['A_fit']/a-1
                predicted=-np.pi**2*(a**4+7*a*a-7-a**-2)/(48*k*k)
                assert abs(got/predicted-1)<0.003,(a,got,predicted)
            rows.append({'A':a,'rotation':rotation,'initial_error':f['A_fit']/a-1,
                         'initial_residual':f['ne'],'refined_error':error,
                         'halvings':len(refined['trail'])-1})
    spherical=[]
    for lat in [60,75,80,85,89]:
        a=1/np.cos(np.radians(lat));r=wavelength_km(.0532)/(16*a)
        profile=lambda rr:local_distortion(lambda la,lo:grid_band(la,lo,.0532),np.radians(lat),np.radians(20),rr)['profile']
        result=refine_radius(profile,r,b)
        err=abs(result['fit']['A_fit']/a-1)
        assert result['converged'] and err<0.001
        spherical.append({'latitude':lat,'A':a,'refined_error':err,'halvings':len(result['trail'])-1})
    # Verify ||Ghat-G|| <= kappa_32 max|sample error| (Proposition 1) for arbitrary sampled perturbations.
    alphas=np.linspace(0,np.pi,200001)
    kappa_32=float(np.abs(1+2*np.cos(2*(b[None,:]-alphas[:,None]))).mean(1).max())
    rng=np.random.default_rng(20260917)
    for _ in range(1000):
        true=np.diag([9.,1.]); x=9*np.cos(b)**2+np.sin(b)**2
        noise=rng.uniform(-.01,.01,32)
        # Unconstrained harmonic tensor: the theorem explicitly excludes fallback.
        design=np.c_[np.ones(32),np.cos(2*b),np.sin(2*b)]
        q=np.linalg.lstsq(design,x+noise,rcond=None)[0]
        fitted=np.array([[q[0]+q[1],q[2]],[q[2],q[0]-q[1]]])
        assert np.linalg.norm(fitted-true,2)<=kappa_32*np.max(abs(noise))+1e-9
    report={'sinusoidal_cases':len(rows),'max_refined_error':max(x['refined_error'] for x in rows),
            'rows':rows,'spherical':spherical,'perturbation_checks':1000}
    (workspace.output/'summaries/estimator_validity.json').write_text(json.dumps(report,indent=2)+'\n')
    counterexamples=[]
    for a in [2,3,5,10,20]:
        k=16*a;f=fit_ellipse(sinusoid(a,0,1/k),b)
        counterexamples.append(dict(A=a,fitted=f['A_fit'],relative_error_percent=100*(f['A_fit']/a-1),residual=f['ne'],axis_error_percent=100*(np.sin(np.pi*a/k)/np.sin(np.pi/k)/a-1)))
    angles=[]
    for radius in [.1,.01]:
        frequencies=np.array([[2,1],[1,-1]])
        u=np.stack([np.cos(b),np.sin(b)])
        squared=np.sum(np.array([1,2])[:,None]*(2*np.sin(radius*(frequencies@u)/2)/radius)**2,axis=0)
        f=fit_ellipse(np.sqrt(squared),b)
        angle=(np.degrees(f['theta_fit'])+90)%180-90
        angles.append(dict(radius=radius,bearing_error_degrees=angle))
    report.update(counterexamples=counterexamples,finite_bearing_rotation=angles)
    (workspace.output/'summaries/estimator_validity.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Sinusoidal cases',len(rows),'max refined error',report['max_refined_error'])
    print('Spherical cases',spherical)



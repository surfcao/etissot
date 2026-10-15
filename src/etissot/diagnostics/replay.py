"""Replay refinement decisions and errors from cached profiles and Jacobian metrics."""
import json
import numpy as np
from etissot.geometry.fitting import refine_radius, fitted_tensor
from etissot.diagnostics.boundaries import shape_distance, clean

def run(workspace):
    data=json.loads((workspace.data/'measurements/released_checkpoints.json').read_text());b=np.arange(32)*2*np.pi/32
    for row in data['radius_cases']:
        profiles={p['radius_km']:np.array(p['profile']) for p in row['profiles']}
        result=refine_radius(lambda radius:profiles[radius],row['start_radius_km'],b)
        row.update(converged=bool(result['converged']),radius_km=result['radius'],halvings=len(result['trail'])-1,initial_fallback=bool(result['trail'][0]['fit']['fallback']),final_fallback=bool(result['fit']['fallback']),A_initial=result['trail'][0]['fit']['A_fit'],A_final=result['fit']['A_fit'],trail=result['trail'])
        if 'jacobian_metric' in row:
            j=np.array(row['jacobian_metric']);e=np.linalg.eigvalsh(j);a=float(np.sqrt(e[-1]/e[0]));g=fitted_tensor(result['fit'])
            row.update(A_jacobian=a,initial_A_error=abs(row['A_initial']/a-1),final_A_error=abs(row['A_final']/a-1),final_tensor_minor_relative_error=float(np.linalg.norm(g-j,2)/e[0]),final_shape_discrepancy=shape_distance(g,j))
    (workspace.output/'summaries/released_audit.json').write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n')

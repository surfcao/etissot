"""Evaluate radius refinement and coordinate equivalence on external checkpoints."""
from pathlib import Path
import json, hashlib
import numpy as np
import torch
from etissot.models.loaders import load_all
from etissot.geometry.probing import probe_points, R_EARTH
from etissot.geometry.fitting import fit_many, fit_ellipse, refine_radius, fitted_tensor
from etissot.statistics.spatial import GLAT, GLON
B = np.arange(32) * 2 * np.pi / 32

def tensor_inputs(lat, lon):
    return (torch.tensor(np.radians(lat), dtype=torch.float64), torch.tensor(np.radians(lon), dtype=torch.float64))

def eval_profile(fn, la, lo, r):
    lat, lon, b = probe_points(np.radians(la), np.radians(lo), r)
    with torch.no_grad():
        e = fn(torch.tensor(lat, dtype=torch.float64), torch.tensor(lon, dtype=torch.float64)).detach().numpy().astype(float)
    return np.linalg.norm(e[1:] - e[:1], axis=1) / r

def derivative(fn, la, lo):
    x, y = tensor_inputs([la], [lo])
    vn = (torch.ones_like(x) / R_EARTH, torch.zeros_like(y))
    ve = (torch.zeros_like(x), torch.ones_like(y) / (R_EARTH * np.cos(np.radians(la))))
    _, n = torch.autograd.functional.jvp(fn, (x, y), vn)
    _, e = torch.autograd.functional.jvp(fn, (x, y), ve)
    mat = np.stack([n.detach().numpy()[0], e.detach().numpy()[0]], 1).astype(float)
    return mat.T @ mat

def shape_distance(g, h):
    l, v = np.linalg.eigh(h)
    z = v * l ** (-0.5) @ v.T
    eig = np.linalg.eigvalsh(z @ g @ z)
    return float(np.sqrt(eig[-1] / eig[0]))

def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def run(workspace, model_dir):
    """Run this analysis with explicit paths; return outputs in workspace.output."""
    ROOT = workspace.data
    P = workspace.output
    torch.set_num_threads(2)
    enc, metadata = load_all(model_dir, workspace.data/'reference/summaries/checkpoint_manifest.json')
    D = np.load(ROOT / 'profiles/global.npz')
    J = np.load(ROOT / 'geometry/jacobians.npz')
    sel = np.arange(0, GLAT.size, 2)
    sel = sel[abs(GLAT.ravel()[sel]) <= 80]
    report = {'protocol': {'tolerance': 0.005, 'successive_passes': 2, 'max_halvings': 16, 'selection': 'two nearest cached median A; two maximum A; up to two fallback sites; two seam-near sites', 'seam_latitudes': [0, 45, 75], 'longitude_epsilons_degrees': [0.01, 0.0001, 1e-06]}, 'metadata': metadata, 'boundaries': {}, 'radius_cases': [], 'cache_replay': {}}
    for name, fn in enc.items():
        path = Path(metadata[name]['checkpoint'])
        metadata[name]['sha256'] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        metadata[name]['checkpoint'] = str(path.relative_to(Path(model_dir).resolve())) if path.is_relative_to(Path(model_dir).resolve()) else 'geoclip-1.2.1/model/weights/' + path.name
        idxs = [int(np.argmin((GLAT.ravel() - la) ** 2 + (GLON.ravel() - lo) ** 2)) for la, lo in [(-30, -75), (0, 20), (40, -105), (60, 60)]]
        replay = []
        for idx in idxs:
            p = eval_profile(fn, GLAT.ravel()[idx], GLON.ravel()[idx], metadata[name]['radius'])
            ref = D['prof__' + name][idx]
            replay.append(float(np.linalg.norm(p - ref) / np.linalg.norm(ref)))
        report['cache_replay'][name] = replay
        boundary = []
        with torch.no_grad():
            for la in [0, 45, 75]:
                for eps in [0.01, 0.0001, 1e-06]:
                    x, y = tensor_inputs([la, la], [180 - eps, -180 + eps])
                    e = fn(x, y)
                    en = e / e.norm(dim=1, keepdim=True)
                    boundary.append({'latitude': la, 'epsilon_degrees': eps, 'distance_normalized': float(torch.linalg.vector_norm(en[0] - en[1])), 'separation_km': float(2 * R_EARTH * np.arcsin(np.cos(np.radians(la)) * np.sin(np.radians(eps))))})
            x, y = tensor_inputs([90, 90], [0, -180])
            e = fn(x, y)
            en = e / e.norm(dim=1, keepdim=True)
            pole = float(torch.linalg.vector_norm(en[0] - en[1]))
        report['boundaries'][name] = {'antimeridian': boundary, 'pole_normalized_distance': pole}
        print('boundary/replay', name, max(replay), boundary[-1]['distance_normalized'], pole, flush=True)
    for name in ['GeoCLIP', 'SatCLIP vit16l10', 'SatCLIP vit16l40', 'SINR cap1000', 'Climplicit', 'CSP iNat2018', 'CSP fMoW']:
        fn = enc[name]
        fit = fit_many(D['prof__' + name][sel], B)
        a = J['AJ__' + name] if 'AJ__' + name in J else fit['A_fit']
        candidate = np.flatnonzero(abs(GLAT.ravel()[sel]) <= 60)
        median = np.median(a[candidate])
        chosen = []
        for group, indices in [('ordinary', candidate[np.argsort(abs(a[candidate] - median))[:2]]), ('high_anisotropy', np.argsort(a)[-2:]), ('fallback', np.flatnonzero(fit['fallback'])[:2])]:
            for index in indices:
                chosen.append((group, float(GLAT.ravel()[sel[index]]), float(GLON.ravel()[sel[index]])))
        chosen += [('near_seam', 0.0, 179.999), ('near_seam', 45.0, 179.999)]
        for group, la, lo in chosen:
            r0 = metadata[name]['radius']
            original = r0
            seam_distance = R_EARTH * np.arcsin(abs(np.cos(np.radians(la)) * np.sin(np.radians(180 - abs(lo))))) if abs(lo) > 90 else float('inf')
            pre = 0
            while r0 >= seam_distance:
                r0 /= 2
                pre += 1
            profiles = []

            def sampled(rr):
                p = eval_profile(fn, la, lo, rr)
                profiles.append({'radius_km': rr, 'profile': p.tolist(), 'zero_fraction': float(np.mean(p == 0))})
                return p
            result = refine_radius(sampled, r0, B)
            row = {'encoder': name, 'stratum': group, 'latitude': la, 'longitude': lo, 'nominal_radius_km': original, 'seam_clearance_km': None if not np.isfinite(seam_distance) else float(seam_distance), 'chart_prehalvings': pre, 'start_radius_km': r0, 'converged': bool(result['converged']), 'radius_km': result['radius'], 'halvings': len(result['trail']) - 1, 'initial_fallback': bool(result['trail'][0]['fit']['fallback']), 'final_fallback': bool(result['fit']['fallback']), 'A_initial': result['trail'][0]['fit']['A_fit'], 'A_final': result['fit']['A_fit'], 'trail': result['trail'], 'profiles': profiles}
            if not name.startswith('CSP'):
                j = derivative(fn, la, lo)
                vals = np.linalg.eigvalsh(j)
                aj = float(np.sqrt(vals[-1] / vals[0]))
                g = fitted_tensor(result['fit'])
                row.update({'jacobian_metric': j.tolist(), 'A_jacobian': aj, 'initial_A_error': abs(row['A_initial'] / aj - 1), 'final_A_error': abs(row['A_final'] / aj - 1), 'final_tensor_minor_relative_error': float(np.linalg.norm(g - j, 2) / vals[0]), 'final_shape_discrepancy': shape_distance(g, j)})
            else:
                row['jacobian_unavailable'] = 'Released coordinate embedding converts tensors through NumPy and float32.'
            report['radius_cases'].append(row)
            print('radius', name, group, la, lo, row['converged'], row['radius_km'], row.get('final_A_error'), flush=True)
            (P / 'summaries/released_audit.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False) + '\n')
    (P / 'summaries/released_audit.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False) + '\n')

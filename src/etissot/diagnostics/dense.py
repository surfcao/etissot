"""Generate directional profiles over regional grids from external checkpoints."""
from pathlib import Path
import json, time
import numpy as np
import torch
from etissot.models.loaders import load_all
from etissot.geometry.probing import destination_point
from etissot.geometry.fitting import fit_many
PANELS = [('CSP iNat2018', 'NA', (30, 50, -110, -80), 0.1, 0.1), ('CSP iNat2018', 'AF', (-10, 10, 10, 30), 0.1, 0.1), ('SINR cap1000', 'NA', (20, 60, -130, -60), 0.5, 0.3), ('Climplicit', 'NA', (20, 60, -130, -60), 0.5, 0.1), ('SatCLIP r18l40', 'NA', (20, 60, -130, -60), 0.5, 1.0)]

def run(workspace, model_dir):
    ROOT = workspace.data
    P = workspace.output
    OUT = P / 'profiles/dense_regional.npz'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    enc, meta = load_all(model_dir, workspace.data/'reference/summaries/checkpoint_manifest.json')
    cache = dict(np.load(OUT)) if OUT.exists() else {}
    bear = np.arange(32) * 2 * np.pi / 32
    t0 = time.time()
    for name, region, bounds, step, r in PANELS:
        key = name + '__' + region
        if 'prof__' + key in cache:
            continue
        la0, la1, lo0, lo1 = bounds
        la = np.arange(la0, la1 + 1e-08, step)
        lo = np.arange(lo0, lo1 + 1e-08, step)
        lat, lon = np.meshgrid(np.radians(la), np.radians(lo), indexing='ij')
        x, y = (lat.ravel(), lon.ravel())
        profiles = np.empty((len(x), 32))
        fn = enc[name]
        for i in range(0, len(x), 128):
            xx = x[i:i + 128]
            yy = y[i:i + 128]
            ringla, ringlo = destination_point(xx[:, None], yy[:, None], bear[None, :], r)
            aa = np.c_[xx, ringla]
            bb = np.c_[yy, ringlo]
            with torch.no_grad():
                z = fn(torch.tensor(aa.ravel(), dtype=torch.float64), torch.tensor(bb.ravel(), dtype=torch.float64)).detach().numpy().astype(float).reshape(len(xx), 33, -1)
            profiles[i:i + len(xx)] = np.linalg.norm(z[:, 1:] - z[:, :1], axis=2) / r
            if i % 4096 == 0:
                print(key, i, '/', len(x), 'elapsed', round(time.time() - t0), flush=True)
        fit = fit_many(profiles, bear)
        cache.update({'prof__' + key: profiles, 'lat__' + key: la, 'lon__' + key: lo, 'r__' + key: np.array(r), 'A__' + key: fit['A_fit'].reshape(lat.shape), 'B__' + key: fit['theta_fit'].reshape(lat.shape), 'fallback__' + key: fit['fallback'].reshape(lat.shape)})
        np.savez_compressed(OUT, **cache)
        print('DONE', key, len(x), round(time.time() - t0), flush=True)
    report = {}
    for name, region, _, step, r in PANELS:
        key = name + '__' + region
        z = np.log(cache['A__' + key])
        z -= z.mean(1, keepdims=True)
        acf = []
        for k in range(1, z.shape[1] // 2):
            acf.append(float(np.mean([np.corrcoef(row[:-k], row[k:])[0, 1] for row in z])))
        cross = next((i + 1 for i, c in enumerate(acf) if c < 0.5), None)
        report[key] = {'sites': int(z.size), 'step_degrees': step, 'radius_km': r, 'median_A': float(np.median(cache['A__' + key])), 'fallback_count': int(cache['fallback__' + key].sum()), 'row_mean_longitude_acf': acf, 'first_acf_below_half_degrees': None if cross is None else cross * step}
    (P / 'summaries/dense_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('All dense panels complete.', flush=True)

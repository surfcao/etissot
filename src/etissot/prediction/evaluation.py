"""Fit and resume regional prediction comparisons in the output directory."""
import json, time
import numpy as np
import torch
from pyproj import Proj
from .gaussian_process import GaussianProcess, WINDOWS, TARGETS, VARIANTS

def run(workspace, windows=None, threads=2, limit=None, replay=False):
    ROOT = workspace.data
    torch.set_num_threads(threads)
    d = {}
    for fn in ['prediction/regions_original.npz', 'prediction/regions_additional.npz']:
        with np.load(ROOT / fn) as f:
            d.update({k: f[k] for k in f.files})
    with np.load(ROOT / 'prediction/geometry_fields.npz') as f:
        d.update({k: f[k] for k in f.files if k.startswith(('A__', 'B__'))})
    replay_checks = []
    count = 0
    t0 = time.time()
    for w in windows or list(WINDOWS):
        out = workspace.output / f'prediction/fits_{w}.json'
        results = json.loads(out.read_text()) if out.exists() else {}
        saved = json.loads(workspace.fit_path(w).read_text()) if replay else {}
        la0, la1, lo0, lo1 = WINDOWS[w]
        proj = Proj(proj='lcc', lat_1=la0 + 5, lat_2=la1 - 5, lat_0=(la0 + la1) / 2, lon_0=(lo0 + lo1) / 2, ellps='WGS84')
        lat, lon = d[f'sites__{w}'].T
        x, y = proj(lon, lat)
        xn, yn = proj(lon, lat + 0.0001)
        gamma = np.arctan2(xn - x, yn - y)
        xy = d[f'xy__{w}']
        assert np.allclose(np.c_[x, y] / 1000, xy, atol=1e-06)
        e = d[f'E__{w}__SatCLIP-L40']
        for target in TARGETS:
            yall = d[f'y__{w}__{target}']
            for seed in range(3):
                rng = np.random.default_rng(100 + seed)
                idx = rng.permutation(np.flatnonzero(np.isfinite(yall)))
                tr, te = (idx[:700], idx[700:1700])
                perm = rng.permutation(len(lat))
                prefix = f'{w}|{target}|{seed}|'
                for variant in VARIANTS:
                    key = prefix + variant
                    if key in results and not replay:
                        continue
                    tensor_key = {'oracle': f'oracle-{target}', 'shuffled': 'SatCLIP-L40', 'tensor+embedding': 'SatCLIP-L40'}.get(variant, variant)
                    if variant in ['iso', 'embedding', 'spatial+embedding']:
                        a = np.ones(len(lat))
                        b = np.zeros(len(lat))
                    else:
                        a = d[f'A__{w}__{tensor_key}']
                        b = d[f'B__{w}__{tensor_key}']
                        if variant == 'shuffled':
                            a, b = (a[perm], b[perm])
                        b = (b + gamma) % np.pi
                    ee = e if 'embedding' in variant else None
                    gp = GaussianProcess(xy[tr], yall[tr], a[tr], b[tr], None if ee is None else ee[tr], variant)
                    if replay:
                        original = saved[key]
                        predicted = gp.predict(dict(original), xy[te], yall[te], a[te], b[te], None if ee is None else ee[te])
                        nll = gp.objective(np.array(original['p']))[0]
                        nll_error = abs(nll-original['nll'])
                        rmse_error = abs(predicted['rmse']-original['rmse'])
                        if nll_error > 1e-6 or rmse_error > 1e-8:
                            raise ValueError(f'Stored fit replay differs: {key}: NLL {nll_error}, RMSE {rmse_error}')
                        replay_checks.append(dict(case=key,nll=float(nll),rmse=predicted['rmse'],nll_absolute_error=nll_error,rmse_absolute_error=rmse_error))
                        count += 1
                        if count % 30 == 0: print('Replayed fitted predictions',count,flush=True)
                        continue
                    prior = lambda v: results.get(prefix + v, {}).get('p')
                    fits = gp.fit(gp.starts(prior('iso'), prior('embedding'), prior('SatCLIP-L40'), prior('spatial+embedding')))
                    fits = gp.predict(fits, xy[te], yall[te], a[te], b[te], None if ee is None else ee[te])
                    results[key] = fits
                    temporary = out.with_suffix('.tmp')
                    temporary.write_text(json.dumps(results, indent=2) + '\n')
                    temporary.replace(out)
                    count += 1
                    print(key, 'nll', round(fits['nll'], 3), 'rmse', round(fits['rmse'], 5), 'spread', round(fits['nll_spread'], 4), 'seconds', round(time.time() - t0), flush=True)
                    if limit and count >= limit:
                        return

    if replay:
        (workspace.output/'summaries/prediction_replay.json').write_text(json.dumps(dict(cases=replay_checks,refitted=False,scope='Recomputed training objective and held-out RMSE at every selected stored parameter vector'),indent=2)+'\n')

def replay(workspace):
    return run(workspace,replay=True)

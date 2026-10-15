"""CSP precision and derivative check (review, 2026-10-13).

The released CSP path builds its sinusoidal input features in float64 NumPy and
then casts them to float32 (torch.FloatTensor) before the feed-forward network.
This script compares, at the same sites:

  f32  -- the released path (float32 network), 32-bearing fit at 0.1 km;
  f64  -- the identical computation with the network and inputs in float64;
  J    -- float64 automatic-differentiation Jacobian of the unit-normalized output
          in the north--east km frame (exact derivative of the float64 function).

It also records, for the leaky-ReLU CSP-iNat network, whether any probe point's
activation pattern differs from the centre's (piecewise-linear region crossing),
and sweeps the probe radius at a few sites to expose float32 rounding.

Writes outputs/summaries/csp_precision.json and outputs/tables/csp_precision.tex.
Usage: python scripts/csp_precision_check.py   (requires torch)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get('ETISSOT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'models/csp/main'))
sys.path.insert(0, str(ROOT / 'src'))
from SpatialRelationEncoder import GridCellSpatialRelationEncoder  # noqa: E402
from module import MultiLayerFeedForwardNN  # noqa: E402
try:
    from etissot.geometry.fitting import fit_ellipse  # noqa: E402
    from etissot.geometry.probing import probe_points  # noqa: E402
except ImportError:  # minimal checkout containing only the geometry modules
    sys.path.insert(0, str(ROOT / 'src/etissot/geometry'))
    from fitting import fit_ellipse  # noqa: E402
    from probing import probe_points  # noqa: E402

R = 6371.0
BEARINGS = np.arange(32) * 2 * np.pi / 32
torch.set_num_threads(4)

CKPT = {
    'CSP iNat2018': 'models/encoders/csp/model_dir/model_inat_2018/model_inat_2018_gridcell_0.0010_32_0.1000000_1_512_leakyrelu_UNSUPER-contsoftmax_0.000500_1.000_1_1.000_TMP20.0000_1.0000_1.0000.pth.tar',
    'CSP fMoW': 'models/encoders/csp/model_dir/model_fmow/model_fmow_gridcell_0.0010_32_0.1000000_1_512_gelu_UNSUPER-contsoftmax_0.000050_1.000_1_0.100_TMP1.0000_1.0000_1.0000.pth.tar',
}
# Refinement-audit sites (supplement Table S6), lat/lon in degrees.
REFINEMENT_SITES = {
    'CSP iNat2018': [(55.0, 112.5), (10.0, 162.5), (-80.0, -57.5), (80.0, -157.5), (-75.0, 42.5), (-57.5, 112.5)],
    'CSP fMoW': [(20.0, 32.5), (47.5, 32.5), (80.0, 172.5), (77.5, 172.5)],
}


def load(name):
    c = torch.load(ROOT / CKPT[name], map_location='cpu', weights_only=False)
    p = c['params']
    ffn = MultiLayerFeedForwardNN(input_dim=4 * p['frequency_num'], output_dim=p['num_filts'],
                                  num_hidden_layers=p['num_hidden_layer'], dropout_rate=p['dropout'],
                                  hidden_dim=p['hidden_dim'], activation=p['spa_f_act'],
                                  use_layernormalize=p['use_layn'], skip_connection=p['skip_connection'],
                                  context_str='GridCellSpatialRelationEncoder')
    m = GridCellSpatialRelationEncoder(p['num_filts'], coord_dim=2, frequency_num=p['frequency_num'],
                                       max_radius=p['max_radius'], min_radius=p['min_radius'],
                                       freq_init=p['freq_init'], ffn=ffn, device='cpu')
    m.load_state_dict({k[len('spa_enc.'):]: v for k, v in c['state_dict'].items() if k.startswith('spa_enc.')})
    m.eval()
    ffn64 = MultiLayerFeedForwardNN(input_dim=4 * p['frequency_num'], output_dim=p['num_filts'],
                                    num_hidden_layers=p['num_hidden_layer'], dropout_rate=p['dropout'],
                                    hidden_dim=p['hidden_dim'], activation=p['spa_f_act'],
                                    use_layernormalize=p['use_layn'], skip_connection=p['skip_connection'],
                                    context_str='GridCellSpatialRelationEncoder')
    ffn64.load_state_dict(m.ffn.state_dict())
    ffn64 = ffn64.double().eval()
    return m, ffn64, p


def coords_deg(lat, lon):
    return np.stack([(np.degrees(lon) + 180) % 360 - 180, np.degrees(lat)], 1)


def f32(m, lat, lon):
    with torch.no_grad():
        y = m(coords_deg(lat, lon)[:, None, :]).squeeze(1).double().numpy()
    return y / np.linalg.norm(y, axis=1, keepdims=True)


def embed64(m, lon_deg, lat_deg):
    """Torch float64 re-implementation of make_input_embeds (differentiable)."""
    freq = torch.tensor(m.freq_list, dtype=torch.float64)
    parts = []
    for coord in (lon_deg, lat_deg):
        ang = coord[:, None] * freq[None, :]
        parts.append(torch.stack([torch.sin(ang), torch.cos(ang)], -1).reshape(len(coord), -1))
    return torch.cat(parts, 1)


def f64(m, ffn64, lat, lon):
    cd = coords_deg(lat, lon)
    x = embed64(m, torch.tensor(cd[:, 0]), torch.tensor(cd[:, 1]))
    # Consistency with the released NumPy embedding (before the float32 cast).
    ref = m.make_input_embeds(cd[:, None, :]).reshape(len(cd), -1)
    assert np.allclose(x.numpy(), ref, rtol=0, atol=1e-12)
    with torch.no_grad():
        y = ffn64(x[:, None, :]).squeeze(1).numpy()
    return y / np.linalg.norm(y, axis=1, keepdims=True)


def jacobian_tensor(m, ffn64, lat, lon):
    lat_t = torch.tensor([lat], dtype=torch.float64, requires_grad=True)
    lon_t = torch.tensor([lon], dtype=torch.float64, requires_grad=True)

    def g(la, lo):
        x = embed64(m, torch.rad2deg(lo), torch.rad2deg(la))
        y = ffn64(x[:, None, :]).squeeze(1)[0]
        return y / torch.linalg.norm(y)
    jl, jo = torch.autograd.functional.jacobian(g, (lat_t, lon_t))
    J = np.column_stack([jl.numpy()[:, 0] / R, jo.numpy()[:, 0] / (R * np.cos(lat))])  # north, east (per km)
    G = J.T @ J
    w = np.linalg.eigvalsh(G)
    return float(np.sqrt(w[1] / w[0]))


def activation_crossing(m, ffn64, lats, lons):
    """True if any probe point's hidden activation sign pattern differs from the centre's."""
    cd = coords_deg(lats, lons)
    x = embed64(m, torch.tensor(cd[:, 0]), torch.tensor(cd[:, 1]))[:, None, :]
    pats = []
    hooks = [mod.register_forward_hook(lambda mod, i, o: pats.append((i[0] > 0).reshape(len(cd), -1)))
             for mod in ffn64.modules() if isinstance(mod, torch.nn.LeakyReLU)]
    with torch.no_grad():
        ffn64(x)
    for h in hooks:
        h.remove()
    if not pats:
        return None
    P = torch.cat(pats, 1)
    return bool((P[1:] != P[0:1]).any())


def fit_at(fn, lat, lon, r):
    la, lo, _ = probe_points(lat, lon, r)
    y = fn(la, lo)
    s = np.linalg.norm(y[1:] - y[0], axis=1) / r
    return fit_ellipse(s, BEARINGS)


def main():
    rng = np.random.default_rng(20261015)
    n = 300
    z = rng.uniform(-np.sin(np.radians(60)), np.sin(np.radians(60)), n)
    sample = list(zip(np.degrees(np.arcsin(z)), rng.uniform(-179, 179, n)))
    out = {'sites': 'area-uniform random sample of 300 sites within 60 degrees (seed 20261015), longitudes within 179 degrees of the antimeridian', 'models': {}}
    for name in CKPT:
        m, ffn64, p = load(name)
        rows = []
        for (la_d, lo_d) in sample + REFINEMENT_SITES[name]:
            la, lo = np.radians(la_d), np.radians(lo_d)
            r32 = fit_at(lambda a, b: f32(m, a, b), la, lo, 0.1)
            r64 = fit_at(lambda a, b: f64(m, ffn64, a, b), la, lo, 0.1)
            aj = jacobian_tensor(m, ffn64, la, lo)
            pl, po, _ = probe_points(la, lo, 0.1)
            cross = activation_crossing(m, ffn64, pl, po)
            rows.append(dict(lat=float(la_d), lon=float(lo_d), refinement_site=(la_d, lo_d) in REFINEMENT_SITES[name],
                             A32=r32['A_fit'], fb32=r32['fallback'], A64=r64['A_fit'], fb64=r64['fallback'], AJ=aj,
                             crossing=cross))
        rs = [r for r in rows if not r['refinement_site']]
        e32 = np.array([abs(r['A32'] / r['AJ'] - 1) for r in rs])
        e64 = np.array([abs(r['A64'] / r['AJ'] - 1) for r in rs])
        AJ = np.array([r['AJ'] for r in rs])
        hi = AJ >= np.quantile(AJ, 0.9)
        w = np.ones(len(rs))  # sites are already area-uniform, so summaries are unweighted

        def wq(x, q, ww):
            o = np.argsort(x); c = np.cumsum(ww[o]) / ww.sum()
            return float(x[o][np.searchsorted(c, q)])
        summ = dict(activation=p['spa_f_act'], n=len(rs),
                    median_AJ=wq(AJ, .5, w),
                    f32_median_err=wq(e32, .5, w), f32_p95_err=wq(e32, .95, w), f32_highA_p95=wq(e32[hi], .95, w[hi]),
                    f64_median_err=wq(e64, .5, w), f64_p95_err=wq(e64, .95, w), f64_highA_p95=wq(e64[hi], .95, w[hi]),
                    f32_fallbacks=int(sum(r['fb32'] for r in rs)), f64_fallbacks=int(sum(r['fb64'] for r in rs)),
                    crossing_fraction=None if rs[0]['crossing'] is None else float(np.mean([r['crossing'] for r in rs])))
        # Radius sweep at refinement sites: float32 versus float64.
        sweep = []
        for (la_d, lo_d) in REFINEMENT_SITES[name][:2]:
            la, lo = np.radians(la_d), np.radians(lo_d)
            aj = jacobian_tensor(m, ffn64, la, lo)
            for r in [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125]:
                a32 = fit_at(lambda a, b: f32(m, a, b), la, lo, r)
                a64 = fit_at(lambda a, b: f64(m, ffn64, a, b), la, lo, r)
                sweep.append(dict(lat=la_d, lon=lo_d, radius_km=r, AJ=aj, A32=a32['A_fit'], A64=a64['A_fit'],
                                  err32=abs(a32['A_fit'] / aj - 1), err64=abs(a64['A_fit'] / aj - 1)))
        out['models'][name] = dict(summary=summ, sites=rows, sweep=sweep)
        print(name, json.dumps(summ, indent=1), flush=True)
        for s in sweep:
            print('  sweep', s['lat'], s['lon'], s['radius_km'], 'AJ=%.4f A32=%.4f A64=%.4f err32=%.2e err64=%.2e' % (s['AJ'], s['A32'], s['A64'], s['err32'], s['err64']))
        for r in rows:
            if r['refinement_site']:
                print('  refinement site', r)
    dest = ROOT / 'outputs/summaries'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'csp_precision.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

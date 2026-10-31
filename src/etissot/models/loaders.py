"""Load the fixed manuscript checkpoint roster with operative precision preserved.

For other encoders or versions, use models.interface.EncoderAdapter and the
public geometry API. This study loader intentionally retains its pinned roster.
"""
from pathlib import Path
import sys, types, importlib, importlib.machinery, importlib.util
import numpy as np
import torch
import hashlib
import json
import os

def filemodule(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def package_root(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ImportError(f'Install the pinned {name} package documented in docs/models.md')
    return Path(spec.origin).parent

def norm(e):
    length = e.norm(dim=1, keepdim=True)
    if not torch.isfinite(e).all() or not torch.isfinite(length).all() or torch.any(length == 0):
        raise ValueError('Unit normalization requires finite, nonzero feature vectors.')
    return e / length

def load_satclip_checkpoint(path, device='cpu'):
    """Load the trusted pinned checkpoint without leaving a Torch environment override."""
    from .vendor.satclip.satclip.load_lightweight import get_satclip_loc_encoder
    key = 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'
    previous = os.environ.get(key)
    os.environ[key] = '1'
    try:
        return get_satclip_loc_encoder(path, device)
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous

def verify_checkpoint(name, path, manifest):
    """Raise unless ``path`` matches the study manifest entry for checkpoint ``name``."""
    expected = json.loads(Path(manifest).read_text())[name]['sha256']
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'Checkpoint does not match the study manifest: {name}: {path}')
    return Path(path)

def load_geoclip_checkpoint(path):
    """Use an explicit weight file with the pinned GeoCLIP architecture."""
    pkg = types.ModuleType('audit_geo')
    pkg.__path__ = [str(package_root('geoclip') / 'model')]
    sys.modules['audit_geo'] = pkg
    model = importlib.import_module('audit_geo.location_encoder').LocationEncoder(from_pretrained=False)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.double().eval()

CSP_SOURCE_MODULES = ('SpatialRelationEncoder', 'module', 'data_utils', 'paths')


def import_csp_sources(model_root):
    """Import the CSP runtime classes after checking the four upstream sources.

    The sources are hashed against the package checkpoint manifest before any of
    them runs, imported without bytecode caches, and removed from ``sys.path`` and
    ``sys.modules`` afterwards so their generic module names do not shadow others.
    """
    folder = Path(model_root) / 'csp/main'
    manifest = json.loads((Path(__file__).with_name('checkpoints.json')).read_text())
    for entry in manifest['external_sources']:
        if entry['family'] != 'csp':
            continue
        path = Path(model_root) / entry['path']
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                raise ValueError(f'CSP source does not match the package manifest: {path}')
    saved = {name: sys.modules.pop(name) for name in CSP_SOURCE_MODULES if name in sys.modules}
    writes_bytecode = sys.dont_write_bytecode
    sys.path.insert(0, str(folder))
    sys.dont_write_bytecode = True
    try:
        # Skip cached bytecode in the model folder, which the manifest does not cover.
        loaded = {}
        for name in ('paths', 'data_utils', 'module', 'SpatialRelationEncoder'):
            spec = importlib.util.spec_from_loader(name, importlib.machinery.SourceFileLoader(name, str(folder / f'{name}.py')))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        return loaded['SpatialRelationEncoder'].GridCellSpatialRelationEncoder, loaded['module'].MultiLayerFeedForwardNN
    finally:
        sys.dont_write_bytecode = writes_bytecode
        sys.path.remove(str(folder))
        for name in CSP_SOURCE_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def load_all(model_dir, checkpoint_manifest=None):
    ROOT = Path(model_dir).resolve()
    VENDOR = Path(__file__).resolve().parent / 'vendor'
    enc = {}
    meta = {}
    expected = json.loads(Path(checkpoint_manifest).read_text()) if checkpoint_manifest is not None else None
    def check(name, path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint for {name}: {path}')
        if expected is not None:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected[name]['sha256']:
                raise ValueError(f'Checkpoint does not match the study manifest: {name}')
        return path
    geo_path = ROOT / 'encoders/geoclip/location_encoder_weights.pth'
    if not geo_path.exists():
        geo_path = package_root('geoclip') / 'model/weights/location_encoder_weights.pth'
    check('GeoCLIP', geo_path)
    geo = load_geoclip_checkpoint(geo_path)

    def geof(la, lo):
        return norm(geo(torch.stack([torch.rad2deg(la), (torch.rad2deg(lo) + 180) % 360 - 180], 1)))
    enc['GeoCLIP'] = geof
    meta['GeoCLIP'] = {'dim': 512, 'normalization': 'L2', 'radius': 1.0, 'checkpoint': str(geo_path), 'precision': 'float64 weights/input; released float32 accumulator/output'}
    sm = filemodule('audit_sinr', VENDOR / 'sinr/models.py')
    for name, tail in [('SINR cap1000', 'hard_cap_num_per_class_1000'), ('SINR cap10', 'hard_cap_num_per_class_10'), ('SINR distilled', 'distilled_from_env')]:
        path = check(name, ROOT / f'encoders/sinr/pretrained_models/model_an_full_input_enc_sin_cos_{tail}.pt')
        c = torch.load(path, map_location='cpu', weights_only=False)
        p = c['params']
        m = sm.ResidualFCNet(p['input_dim'], p['num_classes'], p['num_filts'], p['depth']).double().eval()
        m.load_state_dict(c['state_dict'])

        def fn(la, lo, m=m):
            x = torch.stack([((torch.rad2deg(lo) + 180) % 360 - 180) / 180, torch.rad2deg(la) / 90], 1)
            return m(torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x)], 1), return_feats=True)
        enc[name] = fn
        meta[name] = {'dim': p['num_filts'], 'normalization': 'raw', 'radius': 0.3, 'checkpoint': str(path), 'precision': 'float64'}
    for tag, backbone in [('vit16', 'vit16'), ('r18', 'resnet18'), ('r50', 'resnet50')]:
        for degree in [10, 40]:
            name = f'SatCLIP {tag}l{degree}'
            path = ROOT / f'encoders/satclip/satclip-{backbone}-l{degree}.ckpt'
            if not path.exists():
                candidates = list(ROOT.glob(f'**/satclip-{backbone}-l{degree}.ckpt'))
                if len(candidates) != 1:
                    raise ValueError(f'Expected exactly one checkpoint for {name}, found {len(candidates)}')
                path = candidates[0]
            check(name, path)
            model = load_satclip_checkpoint(path).double().eval()

            def fn(la, lo, m=model):
                return norm(m(torch.stack([(torch.rad2deg(lo) + 180) % 360 - 180, torch.rad2deg(la)], 1)))
            enc[name] = fn
            meta[name] = {'dim': int(fn(torch.tensor([0.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64)).shape[1]), 'normalization': 'L2', 'radius': 1.0, 'checkpoint': str(path), 'precision': 'float64'}
    GridCellSpatialRelationEncoder, MultiLayerFeedForwardNN = import_csp_sources(ROOT)
    for name, folder in [('CSP iNat2018', 'model_inat_2018'), ('CSP fMoW', 'model_fmow')]:
        candidates = list((ROOT / 'encoders/csp/model_dir' / folder).glob('*UNSUPER*.pth.tar'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one checkpoint for {name}, found {len(candidates)}')
        path = check(name, candidates[0])
        c = torch.load(path, map_location='cpu', weights_only=False)
        p = c['params']
        ffn = MultiLayerFeedForwardNN(input_dim=4 * p['frequency_num'], output_dim=p['num_filts'], num_hidden_layers=p['num_hidden_layer'], dropout_rate=p['dropout'], hidden_dim=p['hidden_dim'], activation=p['spa_f_act'], use_layernormalize=p['use_layn'], skip_connection=p['skip_connection'], context_str='GridCellSpatialRelationEncoder')
        m = GridCellSpatialRelationEncoder(p['num_filts'], coord_dim=2, frequency_num=p['frequency_num'], max_radius=p['max_radius'], min_radius=p['min_radius'], freq_init=p['freq_init'], ffn=ffn, device='cpu')
        m.load_state_dict({k[len('spa_enc.'):]: v for k, v in c['state_dict'].items() if k.startswith('spa_enc.')})
        m.eval()

        def fn(la, lo, m=m):
            coords = np.stack([(np.degrees(lo.detach().numpy()) + 180) % 360 - 180, np.degrees(la.detach().numpy())], 1)[:, None, :]
            return norm(m(coords).squeeze(1).double())
        enc[name] = fn
        meta[name] = {'dim': p['num_filts'], 'normalization': 'L2 after float32 forward', 'radius': 0.1, 'checkpoint': str(path), 'precision': 'float32 forward; outputs cast to float64'}
    siren = filemodule('audit_clim_siren', package_root('rshf') / 'climplicit/siren.py')
    path = check('Climplicit', ROOT / 'encoders/climplicit/climplicit-3341daa2.pth')
    sd = torch.load(path, map_location='cpu', weights_only=False)
    cm = siren.SirenNet(dim_in=4, dim_hidden=512, dim_out=256, num_layers=16, dropout=False, h_siren=True, residual_connections=True).double().eval()
    cm.load_state_dict({k[len('location_encoder.'):]: v for k, v in sd.items() if k.startswith('location_encoder.')})

    def clim(la, lo):
        loc = torch.stack([((torch.rad2deg(lo) + 180) % 360 - 180) / 180, torch.rad2deg(la) / 90], 1)
        out = []
        for month in [3, 6, 9, 12]:
            mo = torch.full((len(la), 1), float(month), dtype=torch.float64)
            out.append(cm(torch.cat([loc, torch.sin(mo / 12 * 2 * np.pi), torch.cos(mo / 12 * 2 * np.pi)], 1)))
        return torch.cat(out, 1)
    enc['Climplicit'] = clim
    meta['Climplicit'] = {'dim': 1024, 'normalization': 'raw; concatenate months 3,6,9,12', 'radius': 0.1, 'checkpoint': str(path), 'precision': 'float64 direct location_encoder path'}
    return (enc, meta)

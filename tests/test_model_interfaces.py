"""Encoder-adapter checks without loading or downloading checkpoints."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from etissot.models.loaders import norm, load_satclip_checkpoint


class ModelInterfaceTests(unittest.TestCase):
    def test_wrong_checkpoint_hash_is_rejected_before_model_deserialization(self):
        import json
        from etissot.models.loaders import load_all
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); weight = root/'geoclip/model/weights/location_encoder_weights.pth'
            weight.parent.mkdir(parents=True); weight.write_bytes(b'not a checkpoint')
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'GeoCLIP': {'sha256': '0'*64}}))
            with patch('etissot.models.loaders.package_root', return_value=root/'geoclip'):
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    load_all(root, checkpoint_manifest=manifest)

    def test_normalization_rejects_zero_and_nonfinite_vectors(self):
        for vectors in [torch.zeros((1, 3)), torch.tensor([[1., float('nan')]])]:
            with self.assertRaises(ValueError):
                norm(vectors)
        x = torch.tensor([[3., 4.]], dtype=torch.float64, requires_grad=True)
        y = norm(x)
        np.testing.assert_allclose(y.detach(), [[.6, .8]])
        y[0, 0].backward()
        np.testing.assert_allclose(x.grad, [[.128, -.096]], atol=1e-14)

    def test_satclip_environment_is_restored_on_success_and_failure(self):
        key = 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'
        for original in [None, '0', '1']:
            for fail in [False, True]:
                with self.subTest(original=original, fail=fail), patch.dict(os.environ, clear=False):
                    if original is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original
                    def loader(path, device):
                        self.assertEqual(os.environ[key], '1')
                        if fail:
                            raise RuntimeError('loader failed')
                        return 'loaded'
                    with patch.dict('sys.modules', {'etissot.models.vendor.satclip.satclip.load_lightweight': SimpleNamespace(get_satclip_loc_encoder=loader), 'load_lightweight': SimpleNamespace(get_satclip_loc_encoder=lambda *args: self.fail('Unrelated module must not be used'))}):
                        if fail:
                            with self.assertRaises(RuntimeError):
                                load_satclip_checkpoint('unused.ckpt')
                        else:
                            self.assertEqual(load_satclip_checkpoint('unused.ckpt'), 'loaded')
                    self.assertEqual(os.environ.get(key), original)

    def test_collection_writes_new_measurements_outside_input_data(self):
        from etissot.diagnostics.geometry_case_studies import collect
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)/'data'; (data/'measurements').mkdir(parents=True)
            reference = data/'measurements/geometry_case_studies.npz'
            reference.write_bytes(b'keep original measurements')
            design = dict(queries=np.zeros((20, 2)), candidates=np.zeros((4, 20, 400, 2)),
                probes=np.zeros((3, 20, 33, 2)), global_sites=np.zeros((512, 2)),
                global_probes=np.zeros((512, 33, 2)), radii=np.array([.3, 1., 3., 6.]),
                fit_radii=np.array([.1, .3, 1.]), displacements=np.zeros((4, 20, 400, 2)))
            def encode(x):
                return torch.column_stack([x, torch.ones(len(x))])
            with patch('etissot.diagnostics.geometry_case_studies.make_design', return_value=design), patch(
                    'etissot.diagnostics.geometry_case_studies.encoders', return_value=[('test', encode, {})]):
                path = collect(data, Path(temp)/'models', Path(temp)/'new-measurements')
            self.assertTrue(path.is_file())
            self.assertFalse(path.is_relative_to(data))
            self.assertEqual(reference.read_bytes(), b'keep original measurements')
            with self.assertRaises(ValueError):
                collect(data, Path(temp)/'models', data/'measurements')


if __name__ == '__main__':
    unittest.main()

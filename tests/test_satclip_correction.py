"""Normalization, local isotropy and function-preservation checks; no checkpoints."""
import math
from pathlib import Path
import sys
import unittest
import torch
from etissot.models.satclip_correction import correct_zonal_normalization

import etissot
VENDOR = Path(etissot.__file__).resolve().parent / 'models/vendor/satclip/satclip'
# The vendored upstream modules use absolute intra-project imports.
sys.path.insert(0, str(VENDOR))
from positional_encoding.spherical_harmonics import SphericalHarmonics


class SmallEncoder(torch.nn.Module):
    """Small nonlinear network with the same input-affine interface as SIREN."""
    def __init__(self):
        super().__init__()
        self.posenc = SphericalHarmonics(4)
        self.nnet = SmallNetwork()

    def forward(self, x):
        return self.nnet(self.posenc(x))


class SmallNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(16, 12)])
        self.output = torch.nn.Linear(12, 5)

    def forward(self, x):
        return self.output(torch.sin(self.layers[0](x)))


class SatclipCorrectionTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(123)
        return SmallEncoder().double().eval()

    def test_only_zonal_channels_are_rescaled(self):
        model = self.model()
        fixed = correct_zonal_normalization(model)
        x = torch.tensor([[20., 35.], [-105., -25.]], dtype=torch.float64)
        before = model.posenc(x)
        after = fixed.posenc(x)
        scale = torch.ones(16, dtype=torch.float64)
        scale[[0, 2, 6, 12]] = 1/math.pi
        torch.testing.assert_close(after, before*scale, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(model.posenc(x), before)
        with self.assertRaises(ValueError):
            correct_zonal_normalization(fixed)

    def test_corrected_low_degree_input_is_isotropic(self):
        basis = correct_zonal_normalization(self.model()).posenc
        for lat in [-75., -30., 0., 55., 80.]:
            x = torch.tensor([17., lat], dtype=torch.float64, requires_grad=True)
            jac = torch.autograd.functional.jacobian(lambda z: basis(z[None])[0], x)
            geographic = torch.stack([jac[:,1], jac[:,0]/math.cos(math.radians(lat))], dim=1)
            eigenvalues = torch.linalg.eigvalsh(geographic.T@geographic)
            self.assertAlmostEqual(float(torch.sqrt(eigenvalues[1]/eigenvalues[0])), 1., places=10)

    def test_compensation_preserves_features_and_jacobian(self):
        model = self.model()
        before = model.nnet.layers[0].weight.detach().clone()
        fixed = correct_zonal_normalization(model, preserve_outputs=True)
        x = torch.tensor([17., 35.], dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(fixed(x[None]), model(x[None]), rtol=1e-11, atol=1e-11)
        original_j = torch.autograd.functional.jacobian(lambda z: model(z[None]), x)
        corrected_j = torch.autograd.functional.jacobian(lambda z: fixed(z[None]), x)
        torch.testing.assert_close(corrected_j, original_j, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(model.nnet.layers[0].weight, before)
        intervention = correct_zonal_normalization(model)
        self.assertGreater(float((intervention(x[None])-model(x[None])).detach().abs().max()), 1e-5)

if __name__ == '__main__':
    unittest.main()

"""Opt-in zonal normalization correction for the pinned analytic SatCLIP path.

Original vendor code/checkpoints stay unchanged. This fixes only the extra pi
factor, not cancellation in the high-degree expanded polynomial implementation.
"""
import copy
import math
import torch
from torch import nn


class ZonalNormalizedHarmonics(nn.Module):
    """Divide m=0 channels by pi in SatCLIP's degree-major harmonic ordering."""

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.L = original.L
        self.embedding_dim = original.embedding_dim

    def forward(self, coordinates):
        features = self.original(coordinates)
        scale = features.new_ones(self.embedding_dim)
        indices = [degree * (degree + 1) for degree in range(self.L)]
        scale[indices] = 1 / math.pi
        return features * scale


def correct_zonal_normalization(model, *, preserve_outputs=False):
    """Return an independent corrected copy of a pinned analytic SIREN model.

    preserve_outputs=False holds network weights fixed: an input intervention,
    not a retrained or validated replacement for the released model.
    preserve_outputs=True rescales the matching first-layer weight columns by
    pi, preserving the original network function up to floating-point error.
    """
    basis = model.posenc
    if isinstance(basis, ZonalNormalizedHarmonics):
        raise ValueError('The harmonic normalization has already been corrected.')
    if (type(basis).__name__ != 'SphericalHarmonics'
            or not getattr(getattr(basis, 'SH', None), '__module__', '').endswith('spherical_harmonics_ylm')):
        raise ValueError('Correction requires the pinned analytic harmonic implementation.')
    if not hasattr(model.nnet, 'layers') or not len(model.nnet.layers):
        raise ValueError('Correction requires the inspected SatCLIP SIREN network.')
    corrected = copy.deepcopy(model)
    corrected.posenc = ZonalNormalizedHarmonics(corrected.posenc)
    if preserve_outputs:
        weight = corrected.nnet.layers[0].weight
        if weight.shape[1] != basis.embedding_dim:
            raise ValueError('First-layer width does not match the harmonic basis.')
        indices = [degree * (degree + 1) for degree in range(basis.L)]
        with torch.no_grad():
            weight[:, indices] *= math.pi
    return corrected

"""Measure a specified encoder's finite-radius geographic indicatrix."""
import warnings

import numpy as np

from .probing import probe_points, disc_crossings
from .fitting import fit_ellipse, fitted_tensor
from ..models.interface import EncoderAdapter, iter_feature_batches


SEAM_POLICIES = ('warn', 'error', 'ignore')


def measure_indicatrix(encoder, latitude, longitude, radius_km, n_directions=32, *, seams='warn'):
    """Evaluate ``encoder(latitude, longitude) -> (d,)`` on a geodesic ring.

    A scalar callable or an EncoderAdapter is accepted. Coordinates and returned
    bearings are radians; radius is kilometers.
    The callable must return the selected representation, including any desired
    normalization. It must handle its longitude convention consistently.
    Probe discs that contain a pole or cross the antimeridian are flagged in the
    result (``contains_pole``, ``crosses_antimeridian``); ``seams`` chooses
    whether such discs warn (default), raise ValueError, or pass silently, e.g.
    for an encoder known to be continuous across both. An EncoderAdapter's
    ``domain_check`` can reject further discs. This finite-radius fit does not
    certify smoothness or differential accuracy; inspect ``fit['ne']``.

    ``tensor`` and ``principal_stretches`` are None on a flagged fallback.
    ``unconstrained_tensor`` preserves the actual harmonic fit in that case;
    ``fit`` retains the existing estimator's fallback diagnostics.
    """
    return measure_locations(encoder, [latitude], [longitude], radius_km,
                             n_directions=n_directions, seams=seams)[0]


def measure_locations(encoder, latitudes, longitudes, radius_km, n_directions=32, *, batch_size=1024, seams='warn'):
    """Measure arbitrary sites, using scalar or per-site radii in kilometres.

    Coordinates are aligned one-dimensional arrays in radians. A batched
    EncoderAdapter evaluates complete rings in chunks; a scalar callable
    remains supported. Results preserve site order and adapter provenance.
    No study data, named model roster or model checkpoint is required.
    ``seams`` ('warn', 'error' or 'ignore') handles probe discs that contain a
    pole or cross the antimeridian, as in measure_indicatrix.
    """
    latitudes, longitudes = [np.asarray(v, dtype=float) for v in (latitudes, longitudes)]
    if (latitudes.ndim != 1 or longitudes.shape != latitudes.shape
            or not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all()
            or np.any(abs(latitudes) >= np.pi / 2)):
        raise ValueError('Finite aligned one-dimensional coordinates away from the poles are required.')
    radii = np.asarray(radius_km, dtype=float)
    if radii.ndim == 0:
        if not np.isfinite(radii) or radii <= 0:
            raise ValueError("Radius must be finite and positive.")
        radii = np.full(latitudes.shape, radii.item())
    if radii.shape != latitudes.shape or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Radius must be finite and positive, scalar or one value per location.')
    for value, minimum, label in [(n_directions, 8, 'directions'), (batch_size, 1, 'batch size')]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
            raise ValueError(f'{label} must be an integer of at least {minimum}.')
    if not callable(encoder):
        raise ValueError('Provide a scalar callable or an EncoderAdapter.')
    if seams not in SEAM_POLICIES:
        raise ValueError(f'seams must be one of {SEAM_POLICIES}.')
    if len(latitudes) == 0:
        return []
    crossings = [disc_crossings(a, b, r) for a, b, r in zip(latitudes, longitudes, radii)]
    flagged = [i for i, c in enumerate(crossings) if c['contains_pole'] or c['crosses_antimeridian']]
    if flagged and seams != 'ignore':
        message = (f'{len(flagged)} probe disc(s) contain a pole or cross the antimeridian '
                   f'(first at site index {flagged[0]}); the fit there may measure a coordinate '
                   "discontinuity rather than local geometry. Use seams='ignore' only for an "
                   'encoder that is continuous across both.')
        if seams == 'error':
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    rings = []
    for latitude, longitude, radius in zip(latitudes, longitudes, radii):
        if isinstance(encoder, EncoderAdapter):
            encoder.validate_disc(float(latitude), float(longitude), float(radius))
        lats, lons, bearings = probe_points(latitude, longitude, radius, n_directions)
        rings.append(np.column_stack([lats, lons]))
    coordinates = np.concatenate(rings)
    # Keep feature memory bounded by the chunk size and one probe ring.
    results, pieces = [], []
    block, buffered, site = n_directions + 1, 0, 0
    for features in iter_feature_batches(encoder, coordinates[:, 0], coordinates[:, 1], batch_size):
        offset = 0
        while offset < len(features):
            take = min(block - buffered, len(features) - offset)
            pieces.append(features[offset:offset+take])
            buffered += take
            offset += take
            if buffered == block:
                vectors = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
                result = _fit_indicatrix(vectors, radii[site], bearings)
                result.update(latitude=float(latitudes[site]), longitude=float(longitudes[site]),
                              radius_km=float(radii[site]), result_schema_version=1,
                              **crossings[site],
                              encoder=encoder.describe() if isinstance(encoder, EncoderAdapter) else None)
                results.append(result)
                site += 1
                pieces, buffered = [], 0
    return results


def _fit_indicatrix(features, radius_km, bearings):
    profile = np.linalg.norm(features[1:] - features[0], axis=1) / radius_km
    if not np.all(np.isfinite(profile)):
        raise ValueError("Feature increments exceed finite numerical range")
    if not np.any(profile > 0):
        raise ValueError("Encoder has no measurable change at this probe radius")
    fit = fit_ellipse(profile, bearings)
    design = np.column_stack([np.ones_like(bearings), np.cos(2 * bearings),
                              np.sin(2 * bearings), np.cos(bearings), np.sin(bearings)])
    a0, a2, b2 = np.linalg.lstsq(design, profile ** 2, rcond=None)[0][:3]
    unconstrained = np.array([[a0 + a2, b2], [b2, a0 - a2]])
    valid = not fit["fallback"] and np.isfinite(fit["A_fit"])
    return {"feature_dimension": features.shape[1],
            "bearings": bearings, "profile": profile, "fit": fit,
            "unconstrained_tensor": unconstrained,
            "tensor": fitted_tensor(fit) if valid else None,
            "principal_stretches": np.sqrt([fit["lambda1"], fit["lambda2"]]) if valid else None}

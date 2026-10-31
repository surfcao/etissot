"""Spherical geodesic sampling and finite-lag probes."""
import numpy as np
R_EARTH = 6371.0

def destination_point(lat, lon, bearing, r, R=R_EARTH):
    """Geodesic destination on the sphere; all angles radians, r in km.
    Vectorized over `bearing`."""
    lat, lon, bearing, r, R = [np.asarray(v, dtype=float) for v in (lat, lon, bearing, r, R)]
    if (any(not np.isfinite(v).all() for v in (lat, lon, bearing, r, R))
            or np.any(abs(lat) > np.pi / 2) or np.any(r < 0) or np.any(R <= 0)):
        raise ValueError('Finite angles, valid latitude, nonnegative distance and positive Earth radius are required.')
    d = r / R
    sine = np.sin(lat) * np.cos(d) + np.cos(lat) * np.sin(d) * np.cos(bearing)
    lat2 = np.arcsin(np.clip(sine, -1, 1))
    lon2 = lon + np.arctan2(np.sin(bearing) * np.sin(d) * np.cos(lat), np.cos(d) - np.sin(lat) * np.sin(lat2))
    return (lat2, lon2)

def disc_crossings(lat, lon, r, R=R_EARTH):
    """Report whether the probe disc of radius r (km) around (lat, lon) in radians
    contains a pole or crosses the antimeridian (longitude +/-pi).

    These are coordinate discontinuities: an encoder may give different outputs
    for equivalent coordinates there, and the finite-radius fit then measures
    that jump rather than local geometry.
    """
    lat, lon, r = float(lat), float(lon), float(r)
    d = r / R
    contains_pole = (np.pi / 2 - abs(lat)) <= d
    if contains_pole:
        return {'contains_pole': True, 'crosses_antimeridian': True}
    # Largest longitude offset of a spherical cap that does not contain a pole.
    extent = np.arcsin(min(1.0, np.sin(d) / np.cos(lat)))
    wrapped = (lon + np.pi) % (2 * np.pi) - np.pi
    return {'contains_pole': False, 'crosses_antimeridian': bool(abs(wrapped) + extent >= np.pi)}

def probe_points(lat, lon, r, n_dir=32):
    """Center + ring of n_dir geodesic neighbours at radius r.
    Returns (lats, lons) arrays of length n_dir+1; index 0 is the center."""
    if (not isinstance(n_dir, (int, np.integer)) or isinstance(n_dir, (bool, np.bool_)) or n_dir < 1):
        raise ValueError('Number of directions must be a positive integer.')
    if any(np.ndim(v) != 0 for v in (lat, lon, r)):
        raise ValueError('A probe ring requires scalar coordinates and radius.')
    if not np.isfinite(r) or r <= 0 or r >= np.pi * R_EARTH:
        raise ValueError('Probe radius must be positive and smaller than half the spherical circumference.')
    bearings = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    la2, lo2 = destination_point(lat, lon, bearings, r)
    return (np.concatenate([[lat], la2]), np.concatenate([[lon], lo2]), bearings)

def summarize(emb, r, bearings):
    """Summarize one probe ring; emb is (n_dir+1, d) with row 0 the center.

    Anisotropy and bearing are taken from the harmonic tensor fit, so they agree
    with measure_indicatrix. Prefer measure_locations for new code.
    """
    emb, bearings = np.asarray(emb, dtype=float), np.asarray(bearings, dtype=float)
    if (emb.ndim != 2 or emb.shape[0] != bearings.size + 1 or emb.shape[1] == 0
            or bearings.ndim != 1 or bearings.size == 0 or not np.isfinite(emb).all()
            or not np.isfinite(bearings).all() or not np.isfinite(r) or r <= 0):
        raise ValueError('Finite feature rows matching the probe ring and a positive radius are required.')
    s = np.linalg.norm(emb[1:] - emb[0], axis=1) / r
    if not np.isfinite(s).all() or not np.any(s > 0):
        raise ValueError('No finite measurable directional change at this radius.')
    from .fitting import fit_ellipse  # local import: fitting does not depend on probing
    fit = fit_ellipse(s, bearings)
    s_pos = np.clip(s, 1e-30, None)
    # Anisotropy and bearing come from the fitted tensor (Equations 5-6), as in
    # measure_indicatrix; the bearing is the principal axis modulo pi.
    return {'anisotropy': fit['A_fit'], 'scale': float(np.exp(np.mean(np.log(s_pos)))), 'bearing': fit['theta_fit'], 'fallback': fit['fallback'], 'profile': s}

def local_distortion(f, lat, lon, r, n_dir=32):
    """f: scalar-call encoder f(lat, lon) -> (d,) array."""
    las, los, bearings = probe_points(lat, lon, r, n_dir)
    emb = np.stack([np.asarray(f(a, b), dtype=float) for a, b in zip(las, los)])
    return summarize(emb, r, bearings)

def local_distortion_batched(f_batch, lats, lons, radii, n_dir=32):
    """One batched encoder call for a whole grid of probe locations.

    f_batch: (lat_arr, lon_arr) -> (N, d) array, angles in radians.
    lats, lons, radii: 1-D arrays, one entry per probe location
                       (radii may differ per location/band).
    Returns list of summary dicts.
    """
    lats, lons, radii = [np.atleast_1d(np.asarray(v, dtype=float)) for v in (lats, lons, radii)]
    if any(v.ndim != 1 for v in (lats, lons, radii)) or not (lats.shape == lons.shape == radii.shape):
        raise ValueError('Latitude, longitude and radius arrays must have identical one-dimensional shapes.')
    if lats.size == 0:
        return []
    P, block = (len(lats), n_dir + 1)
    all_la = np.empty(P * block)
    all_lo = np.empty(P * block)
    for i, (la, lo, r) in enumerate(zip(lats, lons, radii)):
        las, los, bearings = probe_points(la, lo, r, n_dir)
        all_la[i * block:(i + 1) * block] = las
        all_lo[i * block:(i + 1) * block] = los
    emb = np.asarray(f_batch(all_la, all_lo), dtype=float)
    if emb.ndim != 2 or emb.shape[0] != P * block or emb.shape[1] == 0:
        raise ValueError('Batched encoder must return one nonempty feature row per requested point.')
    out = []
    for i, r in enumerate(radii):
        out.append(summarize(emb[i * block:(i + 1) * block], r, bearings))
    return out

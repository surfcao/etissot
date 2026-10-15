"""Harmonic tensor fitting, fallback diagnostics and empirical refinement."""
import numpy as np

def _validated_bearings(bearings):
    bearings = np.asarray(bearings, dtype=float)
    if bearings.ndim != 1 or bearings.size < 5 or not np.isfinite(bearings).all():
        raise ValueError('At least five finite bearings are required in a one-dimensional array.')
    ordered = np.sort(bearings % (2 * np.pi))
    gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
    if not np.allclose(gaps, 2 * np.pi / bearings.size, rtol=0, atol=1e-10):
        raise ValueError('Bearings must be equally spaced over a complete circle.')
    return bearings

def fit_ellipse(profile, bearings, legacy=False, tol=1e-06):
    """Fit a nonnegative stretch profile on a complete, equally spaced ring.

    Invalid, nonfinite and entirely zero profiles are rejected. Fallback
    diagnostics retain the historical sampled-minimum substitution.
    """
    bearings = _validated_bearings(bearings)
    profile = np.asarray(profile, dtype=float)
    if (profile.shape != bearings.shape or not np.isfinite(profile).all()
            or np.any(profile < 0) or not np.any(profile > 0)):
        raise ValueError('Profile must match the bearings and contain finite nonnegative, nonzero stretches.')
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('Fallback tolerance must be finite and nonnegative.')
    with np.errstate(over='ignore', under='ignore'):
        s2 = profile ** 2
    if not np.isfinite(s2).all() or not np.any(s2 > 0):
        raise ValueError('Squared stretches are outside the measurable floating-point range.')
    X = np.column_stack([np.ones_like(bearings), np.cos(2 * bearings), np.sin(2 * bearings), np.cos(bearings), np.sin(bearings)])
    coef, *_ = np.linalg.lstsq(X, s2, rcond=None)
    a0, a2, b2, a1, b1 = coef
    amp = np.hypot(a2, b2)
    lam1 = a0 + amp
    lam2 = a0 - amp
    fallback = False
    if legacy:
        lam2 = max(lam2, s2.min())
    elif not lam2 > 0:
        lam2 = max(s2.min(), 1e-12 * lam1)
        fallback = True
    else:
        n = len(bearings)
        allowed = lam2 * (1 + (lam1 / lam2 - 1) * np.sin(np.pi / n) ** 2)
        if s2.min() > allowed * (1 + tol):
            lam2 = s2.min()
            fallback = True
    resid = s2 - X[:, :3] @ coef[:3]
    residual_scale = np.max(abs(resid))
    rms = residual_scale * np.sqrt(np.mean((resid / residual_scale) ** 2)) if residual_scale > 0 else 0.0
    return {'A_fit': float(np.sqrt(lam1 / lam2)), 'theta_fit': float(0.5 * np.arctan2(b2, a2) % np.pi), 'ne': float(rms / a0), 'asym': float(np.hypot(a1, b1) / a0), 'fallback': fallback, 'lambda1': float(lam1), 'lambda2': float(lam2), 'residual_minor': float(rms / lam2) if lam2 > 0 else float('inf')}

def fit_many(profiles, bearings, legacy=False):
    bearings = _validated_bearings(bearings)
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2 or profiles.shape[1] != bearings.size:
        raise ValueError('Profiles must be a two-dimensional array with one column per bearing.')
    out = {k: np.empty(len(profiles)) for k in ('A_fit', 'theta_fit', 'ne', 'asym', 'lambda1', 'lambda2', 'residual_minor')}
    out['fallback'] = np.zeros(len(profiles), bool)
    for i, p in enumerate(profiles):
        r = fit_ellipse(p, bearings, legacy=legacy)
        for k in ('A_fit', 'theta_fit', 'ne', 'asym', 'lambda1', 'lambda2', 'residual_minor'):
            out[k][i] = r[k]
        out['fallback'][i] = r['fallback']
    return out

def fitted_tensor(fit):
    """Reconstruct the fitted tensor in the bearing basis (north, east)."""
    theta = fit['theta_fit']
    u = np.array([np.cos(theta), np.sin(theta)])
    return fit['lambda2'] * np.eye(2) + (fit['lambda1'] - fit['lambda2']) * np.outer(u, u)

def refine_radius(profile_at_radius, initial_radius, bearings, tolerance=0.005, max_halvings=16):
    """Empirical tensor convergence check, NOT a universal error certificate.

    Halve the radius until two consecutive fitted-tensor changes, measured in
    operator norm relative to the finer tensor's minor eigenvalue, are below
    tolerance. Both levels must avoid fallback. Returns the full audit trail.
    An independent Jacobian/analytic reference is required to certify accuracy;
    aliasing or floating-point cancellation can produce false plateaus.
    """
    if not np.isfinite(initial_radius) or initial_radius <= 0:
        raise ValueError('Initial radius must be finite and positive.')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Convergence tolerance must be finite and nonnegative.')
    if (not isinstance(max_halvings, (int, np.integer))
            or isinstance(max_halvings, (bool, np.bool_)) or max_halvings < 0):
        raise ValueError('Maximum halvings must be a nonnegative integer.')
    bearings = _validated_bearings(bearings)
    trail = []
    previous = None
    streak = 0
    for level in range(max_halvings + 1):
        radius = initial_radius / 2 ** level
        fit = fit_ellipse(profile_at_radius(radius), bearings)
        tensor = fitted_tensor(fit)
        change = float('inf')
        if previous is not None and fit['lambda2'] > 0:
            change = np.linalg.norm(tensor - previous[0], 2) / fit['lambda2']
        valid = previous is not None and (not fit['fallback']) and (not previous[1]['fallback']) and np.isfinite(change)
        streak = streak + 1 if valid and change <= tolerance else 0
        trail.append({'radius': radius, 'fit': fit, 'minor_relative_change': float(change)})
        if streak >= 2:
            return {'converged': True, 'radius': radius, 'fit': fit, 'trail': trail}
        previous = (tensor, fit)
    return {'converged': False, 'radius': radius, 'fit': fit, 'trail': trail}

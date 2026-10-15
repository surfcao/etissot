"""Analytic Equal Earth metric and determinant-normalised shapes."""
import numpy as np

def shapes(a, theta):
    u = np.stack([np.cos(theta), np.sin(theta)], 1)
    return np.eye(2)[None] / a[:, None, None] + (a - 1 / a)[:, None, None] * u[:, :, None] * u[:, None, :]

def equal_earth_metric(lat, lon):
    """Analytic Equal Earth derivative, common Earth-radius scale omitted."""
    phi, lam = (np.radians(lat), np.radians(lon))
    t = np.arcsin(np.sqrt(3) / 2 * np.sin(phi))
    ct = np.cos(t)
    a1, a2, a3, a4 = (1.340264, -0.081106, 0.000893, 0.003796)
    den = a1 + 3 * a2 * t * t + 7 * a3 * t ** 6 + 9 * a4 * t ** 8
    dp = 6 * a2 * t + 42 * a3 * t ** 5 + 72 * a4 * t ** 7
    dt = np.sqrt(3) / 2 * np.cos(phi) / ct
    xn = 2 * np.sqrt(3) / 3 * lam * (-np.sin(t) * den - ct * dp) / den ** 2 * dt
    yn = den * dt
    xe = 2 * np.sqrt(3) / 3 * ct / den / np.cos(phi)
    g = np.empty((len(phi), 2, 2))
    g[:, 0, 0] = xn * xn + yn * yn
    g[:, 1, 1] = xe * xe
    g[:, 0, 1] = g[:, 1, 0] = xn * xe
    val, vec = np.linalg.eigh(g)
    return (np.sqrt(val[:, 1] / val[:, 0]), np.arctan2(vec[:, 1, 1], vec[:, 0, 1]) % np.pi)

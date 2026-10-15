"""Fit a synthetic directional profile without study data or checkpoints."""
import numpy as np
from etissot import fit_ellipse, fitted_tensor
bearings = np.linspace(0, 2*np.pi, 32, endpoint=False)
profile = np.sqrt(9*np.cos(bearings-0.4)**2 + np.sin(bearings-0.4)**2)
fit = fit_ellipse(profile, bearings)
print("Anisotropy:", fit["A_fit"])
print("Bearing (radians):", fit["theta_fit"])
print("Fitted metric:\n", fitted_tensor(fit))

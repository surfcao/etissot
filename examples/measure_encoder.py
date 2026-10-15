"""Measure a custom encoder at one location without the study data."""
import numpy as np
from etissot import measure_indicatrix


def encoder(latitude, longitude):
    # Replace with your model adapter; coordinates here are radians.
    return np.array([2 * np.sin(latitude), np.sin(longitude)])


result = measure_indicatrix(encoder, latitude=0.0, longitude=0.0, radius_km=0.3)
print("Anisotropy:", result["fit"]["A_fit"])
print("Bearing (degrees clockwise from north):", np.rad2deg(result["fit"]["theta_fit"]))
print("Fallback:", result["fit"]["fallback"])
print("Tensor in north/east coordinates:", result["tensor"])

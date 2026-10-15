"""Run with only NumPy and etissot; no study inputs or checkpoints.

Replace model_output with the chosen model's batched forward function. Keep its
own framework, precision, device and evaluation mode in that function. Declare
which output is measured and identify its checkpoint in metadata.
"""
import numpy as np
from etissot import EncoderAdapter, measure_locations


def model_output(lonlat_degrees):
    longitude, latitude = np.radians(lonlat_degrees).T
    features = np.column_stack([np.cos(latitude)*np.cos(longitude),
                                np.cos(latitude)*np.sin(longitude),
                                np.sin(latitude)])
    return {'location_features': features}


encoder = EncoderAdapter(
    model_output,
    name='example-spherical-encoder',
    input_units='degrees',
    coordinate_order='lonlat',
    wrap_longitude=True,
    normalization='raw',
    feature_transform=lambda output: output['location_features'],
    metadata={'version': 'example-v1', 'representation': 'three Cartesian components'},
)

results = measure_locations(encoder, np.radians([0., 30., 60.]),
                            np.radians([0., -105., 170.]), radius_km=[.3, 1., .5])
for result in results:
    print(f"Latitude {np.degrees(result['latitude']):g}°: A={result['fit']['A_fit']:.8f}, "
          f"features={result['feature_dimension']}, radius={result['radius_km']:g} km")
print(results[0]['encoder'])

"""Analytic and interface tests for arbitrary future encoders."""
import subprocess
import sys
import unittest
import numpy as np
from etissot import EncoderAdapter, measure_indicatrix, measure_locations


def sphere(coordinates):
    lat, lon = coordinates.T
    return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


class EncoderExtensionTests(unittest.TestCase):
    def test_coordinate_conversion_wrapping_and_feature_selection(self):
        calls = []
        def model(coordinates):
            calls.append(coordinates.copy())
            lon, lat = np.radians(coordinates).T
            return {'features': sphere(np.column_stack([lat, lon])), 'unused': 99}
        adapter = EncoderAdapter(model, name='future-model', input_units='degrees',
            coordinate_order='lonlat', wrap_longitude=True,
            feature_transform=lambda output: output['features'])
        actual = adapter.encode_locations(np.radians([30., -20.]), np.radians([200., -190.]))
        np.testing.assert_allclose(calls[0], [[-160., 30.], [170., -20.]], atol=1e-12)
        np.testing.assert_allclose(actual, sphere(np.radians([[30., -160.], [-20., 170.]])))

    def test_batched_sites_match_scalar_measurements_and_preserve_order(self):
        calls = []
        def model(coordinates):
            calls.append(len(coordinates))
            return sphere(coordinates)
        adapter = EncoderAdapter(model)
        lat, lon, radii = [.2, .5, -.8], [.1, 1., -2.], [.3, 1., 2.]
        results = measure_locations(adapter, lat, lon, radii, batch_size=40)
        self.assertEqual(calls, [40, 40, 19])
        for result, la, lo, radius in zip(results, lat, lon, radii):
            reference = measure_indicatrix(lambda a, b: sphere(np.array([[a, b]]))[0], la, lo, radius)
            np.testing.assert_allclose(result['tensor'], reference['tensor'], rtol=1e-12, atol=1e-18)
            self.assertEqual((result['latitude'], result['longitude'], result['radius_km']), (la, lo, radius))

        buffer = np.empty((40, 3))
        def reused_batch(coordinates):
            buffer[:len(coordinates)] = sphere(coordinates)
            return buffer[:len(coordinates)]
        reused_results = measure_locations(EncoderAdapter(reused_batch), lat, lon, radii, batch_size=40)
        scalar_buffer = np.empty(3)
        def reused_scalar(a, b):
            scalar_buffer[:] = sphere(np.array([[a, b]]))[0]
            return scalar_buffer
        scalar_results = measure_locations(reused_scalar, lat, lon, radii, batch_size=40)
        for reference, reused, scalar in zip(results, reused_results, scalar_results):
            np.testing.assert_allclose(reused['profile'], reference['profile'])
            np.testing.assert_allclose(scalar['profile'], reference['profile'])

    def test_arbitrary_feature_dimensions_and_radius_direction_settings(self):
        for dimension in [3, 64, 257]:
            weights = np.arange(1, dimension+1, dtype=float)
            adapter = EncoderAdapter(lambda x: np.column_stack([2*x[:, 0], x[:, 1]]) @
                np.vstack([np.cos(weights), np.sin(weights)]))
            result = measure_indicatrix(adapter, 0., 0., .1, n_directions=48)
            self.assertEqual(result['feature_dimension'], dimension)
            self.assertEqual(result['profile'].shape, (48,))
            self.assertFalse(result['fit']['fallback'])

    def test_normalization_is_explicit_and_handles_large_vectors(self):
        raw = EncoderAdapter(lambda x: 1e200*sphere(x))
        unit = EncoderAdapter(lambda x: 1e200*sphere(x), normalization='unit')
        coordinates = np.array([[.2, .3], [.4, .5]])
        np.testing.assert_allclose(unit.encode_locations(*coordinates.T), sphere(coordinates), atol=1e-15)
        self.assertGreater(abs(raw.encode_locations(*coordinates.T)).max(), 1e100)
        with self.assertRaises(ValueError):
            EncoderAdapter(lambda x: np.zeros((len(x), 3)), normalization='unit').encode_locations([0.], [0.])

    def test_domain_checks_run_before_any_model_evaluation(self):
        calls = []
        adapter = EncoderAdapter(lambda x: calls.append(x), domain_check=lambda lat, lon, radius: lon < 1.)
        with self.assertRaisesRegex(ValueError, 'rejected'):
            measure_locations(adapter, [0., 0.], [0., 2.], 1.)
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(ValueError, 'Boolean'):
            measure_indicatrix(EncoderAdapter(sphere, domain_check=lambda *args: None), 0., 0., 1.)

    def test_provenance_records_output_conventions_and_model_version(self):
        metadata = {'version': 'new-checkpoint-v2', 'representation': 'pre-head features'}
        adapter = EncoderAdapter(sphere, name='my-model', metadata=metadata)
        metadata['version'] = 'changed externally'
        result = measure_indicatrix(adapter, 0., 0., 1.)
        self.assertEqual(result['encoder']['metadata']['version'], 'new-checkpoint-v2')
        result['encoder']['metadata']['version'] = 'changed result'
        self.assertEqual(adapter.describe()['metadata']['version'], 'new-checkpoint-v2')
        self.assertEqual(result['result_schema_version'], 1)

    def test_feature_dimension_cannot_change_between_batches(self):
        calls = []
        def model(x):
            calls.append(len(x))
            return np.ones((len(x), len(calls)+2))
        with self.assertRaisesRegex(ValueError, 'dimension changed'):
            measure_indicatrix(EncoderAdapter(model), 0., 0., 1., n_directions=1100)

    def test_invalid_coordinates_output_configuration_and_metadata(self):
        for kwargs in [dict(input_units='grads'), dict(coordinate_order='xy'),
                       dict(normalization='automatic'), dict(metadata={'version': np.nan})]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EncoderAdapter(sphere, **kwargs)
        with self.assertRaises(ValueError):
            measure_locations(EncoderAdapter(sphere), [0., .5], [0.], 1.)
        with self.assertRaises(ValueError):
            measure_locations(EncoderAdapter(sphere), [0.], [0.], [1., 2.])
        with self.assertRaises(ValueError):
            measure_indicatrix(EncoderAdapter(lambda x: np.ones((len(x), 2), dtype=complex)), 0., 0., 1.)
        self.assertEqual(measure_locations(EncoderAdapter(sphere), [], [], 1.), [])

    def test_core_does_not_import_frameworks_or_study_dependencies(self):
        code = '''
import importlib.abc, sys, numpy
class BlockStudyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'matplotlib', 'pyproj', 'PIL', 'geoclip', 'rshf'}:
            raise ImportError('Unexpected study dependency: '+fullname)
sys.meta_path.insert(0, BlockStudyImports())
from etissot import EncoderAdapter, measure_indicatrix
import numpy as np
encoder = EncoderAdapter(lambda x: np.column_stack([2*np.sin(x[:, 0]), np.sin(x[:, 1])]))
assert abs(measure_indicatrix(encoder, 0., 0., .3)['fit']['A_fit']-2.) < 1e-6
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()

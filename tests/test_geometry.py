"""Analytic geometry and fitter tests; no study files or checkpoints needed."""
import unittest
import numpy as np
from pyproj import Proj
from etissot.geometry.fitting import fit_ellipse, fit_many, fitted_tensor, refine_radius
from etissot.geometry.probing import (
    R_EARTH, destination_point, probe_points, local_distortion, local_distortion_batched,
)
from etissot.geometry.references import equal_earth_metric, shapes

BEARINGS = np.linspace(0, 2 * np.pi, 32, endpoint=False)


def ellipse(ratio=4., rotation=.3, scale=1.):
    return scale * np.sqrt(ratio**2 * np.cos(BEARINGS-rotation)**2
                           + np.sin(BEARINGS-rotation)**2)


def sphere(lat, lon):
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], axis=-1)


class HarmonicFittingTests(unittest.TestCase):
    def test_isotropic_profile_recovers_metric(self):
        fit = fit_ellipse(np.full(32, 2.5), BEARINGS)
        self.assertFalse(fit['fallback'])
        self.assertAlmostEqual(fit['A_fit'], 1.)
        np.testing.assert_allclose(fitted_tensor(fit), np.eye(2)*6.25, atol=1e-13)
        # No assertion on bearing: the orientation of an isotropic metric is undefined.

    def test_scale_changes_tensor_but_not_anisotropy_or_orientation(self):
        reference = fit_ellipse(ellipse(), BEARINGS)
        for scale in [1e-4, .2, 12., 1e4]:
            with self.subTest(scale=scale):
                fit = fit_ellipse(ellipse(scale=scale), BEARINGS)
                self.assertAlmostEqual(fit['A_fit'], reference['A_fit'], places=10)
                np.testing.assert_allclose(fitted_tensor(fit)/scale**2,
                                           fitted_tensor(reference), rtol=1e-12, atol=1e-12)

    def test_minor_axis_between_samples_does_not_trigger_floor(self):
        rotation = np.pi/32
        profile = ellipse(ratio=10., rotation=rotation)
        fit = fit_ellipse(profile, BEARINGS)
        self.assertFalse(fit['fallback'])
        self.assertAlmostEqual(fit['A_fit'], 10., places=10)
        self.assertAlmostEqual(fit['lambda2'], 1., places=10)
        self.assertGreater(profile.min()**2, fit['lambda2'])
        self.assertLess(fit_ellipse(profile, BEARINGS, legacy=True)['A_fit'], 10.)

    def test_batched_fits_preserve_order_and_flags(self):
        profiles = [ellipse(2.), ellipse(7., .7), np.ones(32)]
        fits = fit_many(profiles, BEARINGS)
        np.testing.assert_allclose(fits['A_fit'], [2., 7., 1.], atol=1e-12)
        np.testing.assert_array_equal(fits['fallback'], [False, False, False])
        for i, profile in enumerate(profiles):
            single = fit_ellipse(profile, BEARINGS)
            for key in fits:
                with self.subTest(row=i, key=key):
                    self.assertEqual(fits[key][i], single[key])

    def test_empty_batch_has_consistent_empty_outputs(self):
        fits = fit_many(np.empty((0, 32)), BEARINGS)
        self.assertTrue(all(value.shape == (0,) for value in fits.values()))
        self.assertEqual(fits['fallback'].dtype, np.dtype(bool))

    def test_nonelliptical_spike_is_flagged(self):
        profile = np.ones(32)
        profile[0] = 100.
        fit = fit_ellipse(profile, BEARINGS)
        self.assertTrue(fit['fallback'])
        self.assertEqual(fit['lambda2'], 1.)
        self.assertTrue(np.isfinite(fit['A_fit']))

    def test_invalid_profiles_and_sampling_are_rejected(self):
        cases = [(np.zeros(32), BEARINGS), (-np.ones(32), BEARINGS),
                 (np.full(32, np.nan), BEARINGS), (np.full(32, 1e200), BEARINGS),
                 (np.ones(31), BEARINGS), (np.ones(32), np.zeros(32)),
                 (np.ones(32), BEARINGS + np.linspace(0, .01, 32))]
        for profile, bearings in cases:
            with self.subTest(profile=profile[0]), self.assertRaises(ValueError):
                fit_ellipse(profile, bearings)

    def test_large_output_scale_does_not_overflow_residual_diagnostics(self):
        fit = fit_ellipse(ellipse(scale=1e100), BEARINGS)
        self.assertAlmostEqual(fit['A_fit'], 4., places=10)
        self.assertTrue(np.isfinite([fit['ne'], fit['asym'], fit['residual_minor']]).all())

    def test_permuted_complete_grid_preserves_tensor(self):
        order = np.random.default_rng(12).permutation(32)
        reference = fitted_tensor(fit_ellipse(ellipse(), BEARINGS))
        actual = fitted_tensor(fit_ellipse(ellipse()[order], BEARINGS[order]))
        np.testing.assert_allclose(actual, reference, atol=1e-13)

    def test_first_harmonic_reports_asymmetry(self):
        fit = fit_ellipse(np.sqrt(4 + .4*np.cos(BEARINGS)), BEARINGS)
        self.assertAlmostEqual(fit['asym'], .1, places=12)
        self.assertGreater(fit['ne'], 0.)


class RadiusRefinementTests(unittest.TestCase):
    def test_invalid_refinement_parameters_are_rejected_before_evaluation(self):
        from unittest.mock import Mock
        profile = Mock()
        for kwargs in [dict(initial_radius=0), dict(initial_radius=np.nan),
                       dict(max_halvings=-1), dict(max_halvings=1.5),
                       dict(tolerance=-.1), dict(tolerance=np.nan)]:
            arguments = dict(initial_radius=1., bearings=BEARINGS)
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                refine_radius(profile, **arguments)
        profile.assert_not_called()

    def test_stable_profile_requires_two_successive_passes(self):
        radii = []
        def profile(radius):
            radii.append(radius)
            return ellipse()
        result = refine_radius(profile, 8., BEARINGS)
        self.assertTrue(result['converged'])
        self.assertEqual(radii, [8., 4., 2.])
        self.assertEqual(result['radius'], 2.)
        self.assertTrue(np.isinf(result['trail'][0]['minor_relative_change']))

    def test_unstable_profile_exhausts_budget_and_retains_trail(self):
        result = refine_radius(lambda radius: ellipse(scale=radius), 8., BEARINGS,
                               max_halvings=3)
        self.assertFalse(result['converged'])
        self.assertEqual([x['radius'] for x in result['trail']], [8., 4., 2., 1.])

    def test_failed_change_resets_success_streak(self):
        scales = iter([1., 1., 2., 2., 2.])
        result = refine_radius(lambda _: ellipse(scale=next(scales)), 16., BEARINGS,
                               max_halvings=4)
        self.assertTrue(result['converged'])
        self.assertEqual(len(result['trail']), 5)
        self.assertEqual(result['radius'], 1.)

    def test_fallback_cannot_be_accepted_as_convergence(self):
        profile = np.ones(32)
        profile[0] = 100.
        result = refine_radius(lambda _: profile, 1., BEARINGS, max_halvings=3)
        self.assertFalse(result['converged'])
        self.assertTrue(all(item['fit']['fallback'] for item in result['trail']))


class GeodesicTests(unittest.TestCase):
    def test_mismatched_batch_cannot_reach_encoder(self):
        from unittest.mock import Mock
        encoder = Mock()
        with self.assertRaises(ValueError):
            local_distortion_batched(encoder, [0., .4], [0.], [1., 1.])
        encoder.assert_not_called()
        with self.assertRaises(ValueError):
            local_distortion_batched(lambda lat, lon: np.ones((len(lat)-1, 3)),
                                     [0.], [0.], [1.])

    def test_empty_batch_does_not_call_encoder(self):
        from unittest.mock import Mock
        encoder = Mock()
        self.assertEqual(local_distortion_batched(encoder, [], [], []), [])
        encoder.assert_not_called()

    def test_invalid_geodesic_inputs_are_rejected(self):
        for lat, radius, earth in [(2., 1., R_EARTH), (0., -1., R_EARTH),
                                   (0., 1., 0.), (0., np.inf, R_EARTH)]:
            with self.subTest(lat=lat, radius=radius), self.assertRaises(ValueError):
                destination_point(lat, 0., 0., radius, earth)
        with self.assertRaises(ValueError):
            probe_points(0., 0., np.pi*R_EARTH)

    def test_cardinal_destinations_at_equator(self):
        distance = 100.
        lat, lon = destination_point(0., 0., np.array([0., np.pi/2]), distance)
        np.testing.assert_allclose(lat, [distance/R_EARTH, 0.], atol=1e-15)
        np.testing.assert_allclose(lon, [0., distance/R_EARTH], atol=1e-15)

    def test_zero_distance_keeps_centre(self):
        lat, lon = destination_point(.4, -.7, BEARINGS, 0.)
        np.testing.assert_allclose(lat, .4, atol=1e-15)
        np.testing.assert_allclose(lon, -.7, atol=1e-15)

    def test_probe_radius_via_independent_haversine(self):
        for latitude, longitude, radius in [(0., 0., 1.), (45., 179.9, 100.), (89., 30., 200.)]:
            with self.subTest(latitude=latitude):
                la, lo = np.radians([latitude, longitude])
                lat, lon, bearings = probe_points(la, lo, radius)
                self.assertEqual(len(lat), 33)
                self.assertEqual((lat[0], lon[0]), (la, lo))
                self.assertEqual(len(bearings), 32)
                h = np.sin((lat[1:]-la)/2)**2 + np.cos(la)*np.cos(lat[1:])*np.sin((lon[1:]-lo)/2)**2
                actual = 2*R_EARTH*np.arcsin(np.sqrt(h))
                np.testing.assert_allclose(actual, radius, rtol=1e-10, atol=1e-8)

    def test_unit_sphere_encoder_has_known_chord_stretch(self):
        radius = 20.
        result = local_distortion(sphere, .5, -.9, radius)
        expected = 2*np.sin(radius/(2*R_EARTH))/radius
        np.testing.assert_allclose(result['profile'], expected, rtol=1e-11)
        self.assertAlmostEqual(result['anisotropy'], 1., places=10)

    def test_batched_encoder_is_called_once_and_preserves_locations(self):
        lats = np.array([0., .4, 1.])
        lons = np.array([0., 2., -1.])
        radii = np.array([1., 10., 100.])
        calls = []
        def encoder(lat, lon):
            calls.append(len(lat))
            return sphere(lat, lon)
        results = local_distortion_batched(encoder, lats, lons, radii)
        self.assertEqual(calls, [99])
        for result, lat, lon, radius in zip(results, lats, lons, radii):
            np.testing.assert_allclose(result['profile'], local_distortion(sphere, lat, lon, radius)['profile'])


class ReferenceGeometryTests(unittest.TestCase):
    def test_shapes_are_positive_definite_with_unit_determinant(self):
        anisotropy = np.array([1., 2., 20.])
        tensors = shapes(anisotropy, np.array([0., .2, 1.]))
        eigenvalues = np.linalg.eigvalsh(tensors)
        self.assertTrue((eigenvalues > 0).all())
        np.testing.assert_allclose(np.linalg.det(tensors), 1., atol=1e-12)
        np.testing.assert_allclose(np.sqrt(eigenvalues[:, 1]/eigenvalues[:, 0]), anisotropy)

    def test_equal_earth_matches_independent_projection_derivative(self):
        projection = Proj(proj='eqearth', R=1)
        for lat, lon in [(0., 0.), (30., -120.), (-60., 140.), (80., 20.)]:
            with self.subTest(lat=lat, lon=lon):
                eps = 1e-4
                north = (np.array(projection(lon, lat+eps))-projection(lon, lat-eps))/(2*np.radians(eps))
                east = (np.array(projection(lon+eps, lat))-projection(lon-eps, lat))/(2*np.radians(eps)*np.cos(np.radians(lat)))
                singular = np.linalg.svd(np.column_stack([north, east]), compute_uv=False)
                actual, _ = equal_earth_metric(np.array([lat]), np.array([lon]))
                np.testing.assert_allclose(actual[0], singular[0]/singular[1], rtol=1e-8)

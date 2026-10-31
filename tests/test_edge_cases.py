"""Edge cases found in the October 2026 code review."""
import unittest
import warnings

import numpy as np

from etissot import EncoderAdapter, fit_ellipse, measure_indicatrix, measure_locations
from etissot.geometry.probing import disc_crossings, local_distortion
from etissot.statistics.spatial import GLAT, block_bootstrap_corr

BEARINGS = np.arange(32) * 2 * np.pi / 32


def ellipse_profile(bearings, a, theta):
    """Stretch profile of an exact ellipse with anisotropy a and major axis theta."""
    return np.sqrt(a ** 2 * np.cos(bearings - theta) ** 2 + np.sin(bearings - theta) ** 2)


class FittingEdgeCases(unittest.TestCase):
    def test_bearing_lies_in_half_open_range(self):
        self.assertEqual(fit_ellipse(np.ones(32), BEARINGS)['theta_fit'], 0.0)
        for a in np.linspace(1.1, 20, 50):
            theta = fit_ellipse(ellipse_profile(BEARINGS, a, 0.0), BEARINGS)['theta_fit']
            self.assertTrue(0 <= theta < np.pi)
            self.assertLess(min(theta, np.pi - theta), 1e-9)

    def test_scale_invariance_within_range_and_clear_error_outside(self):
        base = fit_ellipse(ellipse_profile(BEARINGS, 3.0, 0.7), BEARINGS)
        for factor in (1e-100, 1e100):
            fit = fit_ellipse(factor * ellipse_profile(BEARINGS, 3.0, 0.7), BEARINGS)
            self.assertAlmostEqual(fit['A_fit'], base['A_fit'], places=9)
            self.assertAlmostEqual(fit['theta_fit'], base['theta_fit'], places=9)
        for factor in (1e-200, 1e200):
            with self.assertRaisesRegex(ValueError, 'rescale'):
                fit_ellipse(factor * ellipse_profile(BEARINGS, 3.0, 0.7), BEARINGS)

    def test_odd_rings_use_the_tighter_fallback_bound(self):
        for n in (9, 15, 31):
            bearings = np.arange(n) * 2 * np.pi / n
            for theta in np.linspace(0, np.pi, 13):
                fit = fit_ellipse(ellipse_profile(bearings, 3.0, theta), bearings)
                self.assertFalse(fit['fallback'])
                self.assertAlmostEqual(fit['A_fit'], 3.0, places=8)
        # A sampled minimum between the odd- and even-ring bounds must fall back.
        n = 9
        bearings = np.arange(n) * 2 * np.pi / n
        lam1, lam2 = 9.0, 1.0
        y = lam2 + (lam1 - lam2) * np.sin(bearings) ** 2  # exact squared stretches, minor axis east
        lifted = np.maximum(y, lam2 * (1 + (lam1 / lam2 - 1) * np.sin(np.pi / (2 * n)) ** 2) * 1.5)
        self.assertTrue(fit_ellipse(np.sqrt(lifted), bearings)['fallback'])

    def test_legacy_summary_agrees_with_tensor_fit(self):
        def encoder(lat, lon):
            return np.array([3 * 6371 * lat, 6371 * np.cos(0.4) * lon])
        summary = local_distortion(encoder, 0.4, 0.2, 0.5)
        result = measure_indicatrix(encoder, 0.4, 0.2, 0.5)
        self.assertAlmostEqual(summary['anisotropy'], result['fit']['A_fit'])
        self.assertAlmostEqual(summary['bearing'], result['fit']['theta_fit'])
        self.assertLess(summary['bearing'], np.pi)


class SeamAndPoleChecks(unittest.TestCase):
    def setUp(self):
        self.encoder = EncoderAdapter(lambda c: c.copy(), name='identity-degrees',
                                      input_units='degrees', wrap_longitude=True)

    def test_crossings_are_detected(self):
        self.assertTrue(disc_crossings(0.3, np.radians(179.99), 5.0)['crosses_antimeridian'])
        self.assertFalse(disc_crossings(0.3, np.radians(179.0), 5.0)['crosses_antimeridian'])
        self.assertTrue(disc_crossings(np.radians(89.99), 0.0, 5.0)['contains_pole'])
        self.assertFalse(disc_crossings(np.radians(89.0), 0.0, 5.0)['contains_pole'])

    def test_policy_warns_raises_or_ignores(self):
        args = (self.encoder, [0.3], [np.radians(179.99)], 5.0)
        with self.assertWarns(UserWarning):
            result = measure_locations(*args)[0]
        self.assertTrue(result['crosses_antimeridian'])
        with self.assertRaises(ValueError):
            measure_locations(*args, seams='error')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            measure_locations(*args, seams='ignore')
        with self.assertRaises(ValueError):
            measure_locations(*args, seams='sometimes')

    def test_interior_discs_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = measure_locations(self.encoder, [0.3, -1.2], [0.1, 3.0], 5.0)
        self.assertFalse(any(r['crosses_antimeridian'] or r['contains_pole'] for r in result))


class AdapterLayout(unittest.TestCase):
    def test_lonlat_radians_are_contiguous(self):
        seen = []

        def encode(coordinates):
            seen.append(coordinates.flags['C_CONTIGUOUS'] and all(s > 0 for s in coordinates.strides))
            return coordinates.copy()

        encoder = EncoderAdapter(encode, name='lonlat', coordinate_order='lonlat')
        measure_locations(encoder, [0.3], [0.2], 1.0)
        self.assertTrue(seen and all(seen))


class BootstrapPool(unittest.TestCase):
    def test_nan_and_mask_exclusions_agree(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=GLAT.size)
        b = a + rng.normal(size=GLAT.size)
        keep = np.abs(GLAT.ravel()) <= 40
        by_mask = block_bootstrap_corr(a, b, mask=keep, n_boot=200, seed=1)
        by_nan = block_bootstrap_corr(np.where(keep, a, np.nan), b, n_boot=200, seed=1)
        np.testing.assert_allclose(by_mask, by_nan)



class SecondReview(unittest.TestCase):
    """Cases from the second October 2026 review."""

    def test_constant_variable_has_no_correlation(self):
        from etissot.statistics.spatial import wcorr
        self.assertTrue(np.isnan(wcorr(np.full(5, 0.1), np.arange(5.), np.array([0.3, 0.7, 1.1, 0.2, 0.9]))))

    def test_float32_bearings_are_accepted(self):
        profile = ellipse_profile(BEARINGS, 2.0, 0.4)
        fit = fit_ellipse(profile, BEARINGS.astype(np.float32))
        self.assertAlmostEqual(fit['A_fit'], 2.0, places=5)

    def test_subnormal_squared_stretches_raise(self):
        with self.assertRaises(ValueError):
            fit_ellipse(1e-161 * ellipse_profile(BEARINGS, 2.0, 0.4), BEARINGS)

    def test_refinement_keeps_trail_when_a_level_cannot_be_fitted(self):
        from etissot import refine_radius
        def profile(radius):
            if radius < 0.01:
                return np.zeros(BEARINGS.size)  # identical features at tiny radii
            return ellipse_profile(BEARINGS, 2.0 + radius, 0.4)
        result = refine_radius(profile, 0.05, BEARINGS, tolerance=0.0)
        self.assertFalse(result['converged'])
        self.assertEqual(result['radius'], result['trail'][-1]['radius'])
        self.assertLess(result['failed_level']['radius'], 0.01)
        self.assertIsNotNone(result['fit'])

    def test_equal_earth_metric_accepts_scalars(self):
        from etissot.geometry.references import equal_earth_metric
        a, theta = equal_earth_metric(30.0, 10.0)
        self.assertEqual(a.shape, (1,))
        self.assertGreater(a[0], 1.0)

    def test_combined_kernel_without_priors_is_explained(self):
        from etissot.prediction.gaussian_process import GaussianProcess
        rng = np.random.default_rng(0)
        x = rng.normal(size=(10, 2)) * 100
        gp = GaussianProcess(x, rng.normal(size=10), np.ones(10), np.zeros(10), rng.normal(size=(10, 3)), 'spatial+embedding')
        with self.assertRaisesRegex(ValueError, 'earlier fits'):
            gp.starts()


class CspSourceCheck(unittest.TestCase):
    def test_modified_csp_source_is_refused_before_it_runs(self):
        import tempfile
        from pathlib import Path
        from etissot.models.loaders import import_csp_sources
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root, 'csp/main')
            folder.mkdir(parents=True)
            for name in ('SpatialRelationEncoder', 'module', 'data_utils', 'paths'):
                (folder / f'{name}.py').write_text("raise RuntimeError('unverified source executed')\n")
            with self.assertRaisesRegex(ValueError, 'does not match the package manifest'):
                import_csp_sources(root)

if __name__ == '__main__':
    unittest.main()

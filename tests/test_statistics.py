"""Known weighted statistics and deterministic geographic resampling."""
import unittest
import numpy as np
from etissot.statistics.spatial import wcorr, wquantile, wmedian, block_bootstrap_corr, GLAT, GLON


class WeightedStatisticsTests(unittest.TestCase):
    def test_correlation_rejects_invalid_weights_masks_and_dimensions(self):
        for weights in [[-1, 2, 1], [1, np.nan, 1], [1, np.inf, 1], [0, 0, 0], [1]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                wcorr([1, 2, 3], [3, 4, 2], weights)
        with self.assertRaises(ValueError):
            wcorr([1, 2, 3], [3, 4, 2], [1, 1, 1], mask=[1])

    def test_constant_correlation_is_explicitly_undefined(self):
        self.assertTrue(np.isnan(wcorr([1, 1, 1], [1, 2, 3], [1, 1, 1])))
        with self.assertRaises(ValueError):
            block_bootstrap_corr(np.ones(GLAT.shape), GLAT, n_boot=5)

    def test_small_blocks_do_not_collide_between_latitude_bands(self):
        from etissot.statistics.spatial import block_ids
        ids = block_ids(2.5, 1.).reshape(GLAT.shape)
        self.assertEqual(np.unique(ids).size, GLAT.size)
        with self.assertRaises(ValueError):
            block_ids(0, 30)

    def test_invalid_bootstrap_budget_is_rejected(self):
        for budget in [0, -1, 1.5]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                block_bootstrap_corr(GLAT, GLON, n_boot=budget)

    def test_correlation_agrees_with_explicit_integer_weight_replication(self):
        x = np.array([1., 3., 8., 4.])
        y = np.array([7., 2., 4., 9.])
        weights = np.array([1, 4, 2, 3])
        expected = np.corrcoef(np.repeat(x, weights), np.repeat(y, weights))[0, 1]
        self.assertAlmostEqual(wcorr(x, y, weights), expected, places=14)

    def test_mask_zero_weights_and_nonfinite_pairs_are_excluded(self):
        x = [1., 2., 3., np.nan, 99., -99.]
        y = [2., 4., 6., 7., -100., 100.]
        self.assertAlmostEqual(wcorr(x, y, [1., 1., 1., 1., 0., 1.],
                                     mask=[1, 1, 1, 1, 1, 0]), 1.)

    def test_correlation_is_invariant_to_positive_weight_scaling(self):
        x, y, weights = [1., 2., 4.], [8., 3., 5.], np.array([1., 3., 2.])
        self.assertAlmostEqual(wcorr(x, y, weights), wcorr(x, y, 100*weights))
        self.assertAlmostEqual(wcorr(x, [-v for v in y], weights), -wcorr(x, y, weights))

    def test_weighted_quantile_uses_midpoint_interpolation(self):
        x, weights = [20., 0., 10.], [1., 1., 2.]
        np.testing.assert_allclose(wquantile(x, [0., .25, .5, 1.], weights), [0., 10/3, 10., 20.])
        self.assertEqual(wmedian(x, weights), 10.)

    def test_quantile_ignores_nonfinite_observations(self):
        self.assertEqual(wmedian([np.nan, 0., 10., np.inf], [100., 1., 1., 100.]), 5.)

    def test_quantile_ignores_zero_weight_observations(self):
        np.testing.assert_array_equal(wquantile([0, 1, 2], [0, .25, .75, 1], [0, 1, 0]), [1]*4)

    def test_quantile_rejects_invalid_weights_and_empty_samples(self):
        for values, weights in [([1, 2], [-1, 2]), ([1, 2], [1, np.nan]),
                                ([1, 2], [1, np.inf]), ([1, 2], [0, 0]),
                                ([np.nan], [1]), ([], []), ([1, 2], [1])]:
            with self.subTest(values=values, weights=weights), self.assertRaises(ValueError):
                wmedian(values, weights)

    def test_quantile_rejects_invalid_probabilities(self):
        for q in [-.1, 1.1, np.nan, np.inf]:
            with self.subTest(q=q), self.assertRaises(ValueError):
                wquantile([1, 2], q, [1, 1])

    def test_block_bootstrap_is_deterministic_and_contains_affine_truth(self):
        x = GLAT + .2*GLON
        y = 3*x + 7
        first = block_bootstrap_corr(x, y, n_boot=25, seed=31)
        second = block_bootstrap_corr(x, y, n_boot=25, seed=31)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, [1., 1., 1.], atol=1e-14)

    def test_masked_bootstrap_matches_weighted_point_correlation(self):
        x = GLAT + GLON
        y = GLAT**2 - .7*GLON
        mask = abs(GLAT) < 45
        result = block_bootstrap_corr(x, y, mask=mask, n_boot=20, seed=4)
        self.assertAlmostEqual(result[0], wcorr(x, y, mask=mask))
        self.assertLessEqual(result[1], result[2])
        self.assertTrue(np.isfinite(result).all())

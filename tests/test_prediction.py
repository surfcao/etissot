"""Independent small-matrix GP checks without fitting the study dataset."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from etissot.prediction.gaussian_process import GaussianProcess, T


class GaussianProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        rng = np.random.default_rng(53)
        self.x = rng.normal(size=(7, 2))*100
        self.y = rng.normal(size=7)
        self.a = rng.uniform(1, 4, 7)
        self.b = rng.uniform(0, np.pi, 7)
        self.e = rng.normal(size=(7, 3))

    def model(self, kind):
        return GaussianProcess(self.x, self.y, self.a, self.b,
                               self.e if 'embedding' in kind else None, kind)

    @staticmethod
    def covariance(model, p):
        return model.kernel(T(p), model.dx, model.dy, model.a, model.b,
                            model.a, model.b, model.ed).numpy()

    def test_prediction_rejects_unaligned_or_missing_test_features(self):
        model = self.model('embedding')
        fit = dict(p=[0., 0., -2.])
        with self.assertRaises(ValueError):
            model.predict(fit, self.x[:2], self.y[:2], self.a[:2], self.b[:2], None)
        with self.assertRaises(ValueError):
            model.predict(fit, self.x[:2], self.y[:1], self.a[:2], self.b[:2], self.e[:2])

    def test_invalid_variants_and_training_data_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model('misspelled')
        with self.assertRaises(ValueError):
            GaussianProcess(self.x, self.y, self.a, self.b, None, 'embedding')
        with self.assertRaises(ValueError):
            GaussianProcess(self.x[:-1], self.y, self.a, self.b, None, 'iso')
        with self.assertRaises(ValueError):
            GaussianProcess(self.x, self.y, np.zeros(7), self.b, None, 'SatCLIP-L40')

    def test_nonfinite_optimizer_result_cannot_replace_valid_start(self):
        model = self.model('iso')
        start = np.array([np.log(150.), .2, -2.])
        failed = SimpleNamespace(x=np.full(3, np.nan), fun=np.nan,
                                 success=False, message='nonfinite', nit=1)
        with patch('etissot.prediction.gaussian_process.minimize', return_value=failed):
            result = model.fit([start])
        np.testing.assert_array_equal(result['p'], start)
        self.assertTrue(np.isfinite(result['nll']))

    def test_no_valid_fit_raises_instead_of_returning_penalty_as_a_fit(self):
        model = self.model('iso')
        with self.assertRaises(RuntimeError):
            model.fit([])
        bad = np.full(3, np.nan)
        failed = SimpleNamespace(x=bad, fun=1e20, success=False, message='bad', nit=0)
        with patch('etissot.prediction.gaussian_process.minimize', return_value=failed):
            with self.assertRaises(RuntimeError):
                model.fit([bad])

    def test_isotropic_covariance_matches_closed_form_matern(self):
        model = self.model('iso')
        p = [np.log(180.), np.log(1.7), -2.]
        distance = np.linalg.norm(self.x[:, None, :]-self.x[None, :, :], axis=-1)
        z = np.sqrt(3)*distance/180.
        expected = 1.7*(1+z)*np.exp(-z)
        np.testing.assert_allclose(self.covariance(model, p), expected, atol=1e-14)

    def test_covariances_are_symmetric_positive_definite(self):
        for kind in ['iso', 'embedding', 'SatCLIP-L40', 'spatial+embedding', 'tensor+embedding']:
            with self.subTest(kind=kind):
                model = self.model(kind)
                p = [np.log(100.), .2, -2.]
                if model.tensor:
                    p += [.6]
                if model.is_combo:
                    p += [np.log(2.), -.4]
                covariance = self.covariance(model, p)
                np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)
                self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.)

    def test_zero_tensor_exponent_nests_spatial_and_combined_models(self):
        for base_kind, tensor_kind in [('iso', 'SatCLIP-L40'), ('spatial+embedding', 'tensor+embedding')]:
            with self.subTest(base=base_kind):
                p = np.array([np.log(200.), .2, -2.] + ([np.log(2.), -.3] if 'embedding' in base_kind else []))
                base, tensor = self.model(base_kind), self.model(tensor_kind)
                q = np.insert(p, 3, 0.)
                np.testing.assert_allclose(self.covariance(base, p), self.covariance(tensor, q), atol=1e-14)
                self.assertAlmostEqual(base.objective(p)[0], tensor.objective(q)[0], places=11)

    def test_likelihood_gradient_matches_central_difference(self):
        for kind in ['iso', 'SatCLIP-L40', 'tensor+embedding']:
            with self.subTest(kind=kind):
                model = self.model(kind)
                p = np.array([np.log(150.), .2, -2.] + ([.7] if model.tensor else [])
                             + ([np.log(2.), -.4] if model.is_combo else []))
                _, actual = model.objective(p)
                step = np.eye(len(p))*1e-5
                expected = [(model.objective(p+d)[0]-model.objective(p-d)[0])/2e-5 for d in step]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_optimizer_cannot_discard_a_better_initial_point(self):
        model = self.model('iso')
        start = np.array([np.log(150.), .2, -2.])
        initial = model.objective(start)[0]
        failed = SimpleNamespace(x=start+1., fun=initial+10., success=False, message='test failure', nit=1)
        with patch('etissot.prediction.gaussian_process.minimize', return_value=failed):
            result = model.fit([start])
        np.testing.assert_array_equal(result['p'], start)
        self.assertEqual(result['nll'], initial)
        self.assertFalse(result['starts'][0]['success'])

    def test_prediction_scores_match_direct_linear_algebra(self):
        model = self.model('iso')
        p = np.array([np.log(150.), .2, -2.])
        xte, yte = self.x[:2]+15., self.y[:2]+.1
        def kernel(x, y):
            z = np.sqrt(3)*np.linalg.norm(x[:, None, :]-y[None, :, :], axis=-1)/np.exp(p[0])
            return np.exp(p[1])*(1+z)*np.exp(-z)
        covariance = kernel(self.x, self.x)+(np.exp(p[2])+1e-6)*np.eye(len(self.x))
        cross = kernel(xte, self.x)
        train = (self.y-self.y.mean())/(self.y.std()+1e-9)
        truth = (yte-self.y.mean())/(self.y.std()+1e-9)
        mean = cross @ np.linalg.solve(covariance, train)
        variance = np.maximum(np.exp(p[1])-np.sum(cross*np.linalg.solve(covariance, cross.T).T, axis=1), 1e-9)+np.exp(p[2])
        expected_rmse = np.sqrt(np.mean((mean-truth)**2))
        expected_nlpd = np.mean(.5*np.log(2*np.pi*variance)+.5*(mean-truth)**2/variance)
        result = model.predict({'p': p.tolist()}, xte, yte, np.ones(2), np.zeros(2), None)
        self.assertAlmostEqual(result['rmse'], expected_rmse, places=12)
        self.assertAlmostEqual(result['nlpd'], expected_nlpd, places=12)

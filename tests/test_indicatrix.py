import unittest
import numpy as np
from etissot import measure_indicatrix


class IndicatrixTests(unittest.TestCase):
    def test_spherical_cartesian_encoder_is_isotropic(self):
        def encoder(lat, lon):
            return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        result = measure_indicatrix(encoder, 0.7, 0.3, 1.0)
        self.assertFalse(result["fit"]["fallback"])
        self.assertAlmostEqual(result["fit"]["A_fit"], 1.0, places=8)
        np.testing.assert_allclose(result["tensor"], result["unconstrained_tensor"], atol=1e-16)

    def test_directional_stretch_and_output_rescaling(self):
        def encoder(lat, lon):
            return np.array([2*np.sin(lat), np.sin(lon)])
        a = measure_indicatrix(encoder, 0., 0., 0.3)
        b = measure_indicatrix(lambda lat, lon: 5*encoder(lat, lon), 0., 0., 0.3)
        self.assertAlmostEqual(a["fit"]["A_fit"], 2., places=7)
        self.assertAlmostEqual(a["fit"]["A_fit"], b["fit"]["A_fit"], places=12)
        np.testing.assert_allclose(b["tensor"], 25*a["tensor"], rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(b["principal_stretches"], 5*a["principal_stretches"], rtol=1e-12)

    def test_invalid_or_constant_outputs_are_rejected(self):
        with self.assertRaises(ValueError):
            measure_indicatrix(lambda lat, lon: np.ones(3), 0., 0., 1.)
        with self.assertRaises(ValueError):
            measure_indicatrix(lambda lat, lon: np.array([np.nan]), 0., 0., 1.)
        with self.assertRaises(ValueError):
            measure_indicatrix(lambda lat, lon: np.array([lat]), 0., 0., 0.)

    def test_fallback_is_not_exposed_as_a_predictive_tensor(self):
        def encoder(lat, lon):
            if lat == 0 and lon == 0:
                return np.array([0.])
            return np.array([100. if lat > 0.99/6371 else 1.])
        result = measure_indicatrix(encoder, 0., 0., 1.)
        self.assertTrue(result["fit"]["fallback"])
        self.assertIsNone(result["tensor"])
        self.assertIsNone(result["principal_stretches"])
        self.assertLess(np.linalg.eigvalsh(result["unconstrained_tensor"])[0], 0.)


if __name__ == "__main__":
    unittest.main()

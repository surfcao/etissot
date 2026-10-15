"""Small independent checks of package APIs and path safety."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from etissot import fit_ellipse, fitted_tensor
from etissot.workspace import Workspace

class PublicAPI(unittest.TestCase):
    def test_known_rotated_metric(self):
        theta = np.linspace(0, 2*np.pi, 32, endpoint=False)
        rotation = .4
        u = np.array([np.cos(rotation), np.sin(rotation)])
        metric = np.eye(2)+8*np.outer(u,u)
        directions = np.column_stack([np.cos(theta), np.sin(theta)])
        profile = np.sqrt(np.einsum("ni,ij,nj->n", directions, metric, directions))
        fit = fit_ellipse(profile, theta)
        self.assertFalse(fit["fallback"])
        np.testing.assert_allclose(fitted_tensor(fit), metric, atol=1e-12)
        self.assertAlmostEqual(fit["A_fit"], 3.)

    def test_outputs_cannot_overlap_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                Workspace(Path(d), Path(d)/"outputs")
            with self.assertRaises(ValueError):
                Workspace(Path(d)/"data", Path(d))

    def test_output_fit_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();data=root/"data";output=root/"outputs"
            (data/"reference").mkdir(parents=True)
            work=Workspace(data,output).prepare()
            self.assertEqual(work.fit_path("CONUS"),data/"prediction/fits_CONUS.json")
            target=output/"prediction/fits_CONUS.json";target.write_text("{}")
            self.assertEqual(work.fit_path("CONUS"),target)

if __name__ == "__main__":
    unittest.main()

"""Analytic checks for rank severity and global feature summaries."""
import unittest
import numpy as np
from etissot.diagnostics.geometry_case_studies import neighbour_metrics,covariance_summary

class GeometryCaseStudyTests(unittest.TestCase):
    def test_invalid_boundary_tensor_cannot_produce_a_nan_ellipse(self):
        from etissot.diagnostics.neighbourhoods import tensor_neighbourhood_boundary
        for tensor in [np.full((2, 2), np.nan), np.diag([1., -1.]),
                       np.array([[1., 2.], [0., 1.]]), np.eye(3)]:
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                tensor_neighbourhood_boundary(tensor, 1.)

    def test_intrusion_severity_distinguishes_equal_overlap(self):
        reference=np.arange(6,dtype=float)
        near=neighbour_metrics(reference,np.array([0,2,1,3,4,5]),2)
        far=neighbour_metrics(reference,np.array([0,2,3,4,5,1]),2)
        self.assertEqual(near['overlap'],far['overlap'])
        self.assertGreater(far['mean_excess_rank'],near['mean_excess_rank'])

    def test_independent_tensor_predicts_linear_embedding_neighbours(self):
        from etissot.geometry.fitting import fit_ellipse, fitted_tensor
        theta=np.arange(32)*2*np.pi/32
        directions=np.column_stack([np.cos(theta),np.sin(theta)])
        jacobian=np.array([[3.,1.],[0.,1.],[1.,0.]])
        stretches=np.linalg.norm(directions@jacobian.T,axis=1)
        fit=fit_ellipse(stretches,theta)
        self.assertFalse(fit['fallback'])
        candidates=np.random.default_rng(27).normal(size=(100,2))
        actual=np.linalg.norm(candidates@jacobian.T,axis=1)
        predicted=np.sqrt(np.einsum('ni,ij,nj->n',candidates,fitted_tensor(fit),candidates))
        np.testing.assert_allclose(actual,predicted,rtol=1e-12)
        self.assertEqual(neighbour_metrics(predicted,actual,10)['overlap'],1.)
        self.assertLess(neighbour_metrics(np.linalg.norm(candidates,axis=1),actual,10)['overlap'],1.)

    def test_geographic_neighbourhood_ellipse_uses_inverse_stretches(self):
        from etissot.diagnostics.neighbourhoods import tensor_neighbourhood_boundary
        g=np.diag([9.,1.])
        points=tensor_neighbourhood_boundary(g,6.,samples=5)
        np.testing.assert_allclose(points[0],[2.,0.],atol=1e-12)
        np.testing.assert_allclose(points[1],[0.,6.],atol=1e-12)
        np.testing.assert_allclose(np.einsum('ni,ij,nj->n',points,g,points),36.)
        np.testing.assert_allclose(tensor_neighbourhood_boundary(g*25,30.,samples=5),points,atol=1e-12)

    def test_global_summary_known_circle_and_rotation(self):
        x=np.array([[1.,0],[0,1],[-1,0],[0,-1]])
        q=np.array([[1.,1],[-1,1]])/np.sqrt(2)
        a=covariance_summary(x);b=covariance_summary(x@q)
        self.assertAlmostEqual(a['top_variance_share'],.5)
        self.assertAlmostEqual(a['effective_rank'],2)
        self.assertAlmostEqual(a['mean_pair_cosine'],-1/3)
        for key in a:self.assertAlmostEqual(a[key],b[key])

if __name__=='__main__':unittest.main()

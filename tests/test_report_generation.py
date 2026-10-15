"""Exercise a cached checkpoint-report path without real models or study data."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from etissot.workspace import Workspace
from etissot.diagnostics.satclip_correction import run


class CheckpointReportTests(unittest.TestCase):
    def test_satclip_report_preserves_latex_reference_and_measured_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = Workspace(root/'data', root/'output')
            (workspace.data/'profiles').mkdir(parents=True)
            for name in ['summaries','tables']:
                (workspace.output/name).mkdir(parents=True,exist_ok=True)
            metadata = [dict(name='vit16-L10', corrected_center_cosine=[1.,1.],
                             compensated_max_output_difference=0.)]
            np.savez(workspace.data/'profiles/satclip_correction.npz',
                     metadata=np.array(json.dumps(metadata)), sites=np.array([[0.,0.],[30.,0.]]),
                     radii=np.array([1.,.5]), output_profiles=np.ones((1,3,2,2,32)),
                     input_profiles=np.ones((1,2,2,2,32)))
            report = run(workspace)
            self.assertAlmostEqual(report['checkpoints'][0]['radii']['1.0']['output_released']['median_A'],1.)
            text = (workspace.output/'tables/satclip_correction.tex').read_text()
            self.assertIn(r'\ref{supp:geometry}',text)
            self.assertIn('1.000&1.000&1.000&1.000',text)

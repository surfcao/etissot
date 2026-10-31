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
            bearings = np.arange(32)*2*np.pi/32
            ellipse = lambda a: np.sqrt((a*np.cos(bearings))**2+np.sin(bearings)**2)
            names = [f'{b}-L{d}' for b in ['vit16','resnet18','resnet50'] for d in [10,40]]
            metadata = [dict(name=n, corrected_center_cosine=[1.,1.], compensated_max_output_difference=0.) for n in names]
            sites, radii = np.array([[0.,0.],[30.,0.]]), np.array([1.,.5])
            outputs = np.empty((6,3,2,2,32)); inputs = np.empty((6,2,2,2,32))
            corrected_l40 = iter([1.4, 1.6, 1.55])  # one reduction only: no consistent reduction
            for i, name in enumerate(names):
                l40 = name.endswith('L40'); corrected = next(corrected_l40) if l40 else 1.4
                for j in range(2):
                    inputs[i,0,j] = ellipse(1.2)
                    inputs[i,1,j] = ellipse((1.001 if j == 0 else 1.002) if l40 else 1.0)
                    outputs[i,0,j] = outputs[i,2,j] = ellipse(1.5)
                    outputs[i,1,j] = ellipse(corrected)
            def write(outputs):
                np.savez(workspace.data/'profiles/satclip_correction.npz', metadata=np.array(json.dumps(metadata)),
                         sites=sites, radii=radii, output_profiles=outputs, input_profiles=inputs)
            write(outputs)
            report = run(workspace)
            self.assertAlmostEqual(report['checkpoints'][0]['radii']['1.0']['output_released']['median_A'],1.5)
            text = (workspace.output/'tables/satclip_correction.tex').read_text()
            self.assertIn(r'\ref{supp:geometry}',text)
            self.assertIn('ViT-L10&1.200&1.000&1.500&1.400',text)
            # The prose states that the rescaling does not consistently reduce L40 anisotropy.
            reduced = outputs.copy()
            for i in (1, 3, 5):
                reduced[i,1] = ellipse(1.3)
            write(reduced)
            with self.assertRaisesRegex(ValueError, 'no longer match'):
                run(workspace)

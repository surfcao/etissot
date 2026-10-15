"""Guard against false reproduction through prefilled outputs and stale paper blocks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from etissot.workspace import Workspace
from etissot.validation.manuscript import compare_generated_blocks
from etissot.diagnostics.neighbours import query_metrics
from etissot.validation.dense import compare_dense_fit
from etissot.validation.environment import inspect_environment
from unittest.mock import patch
import json
from PIL import Image
from etissot.validation.manuscript import verify, load_documents

class ReproducibilityTests(unittest.TestCase):
    def test_relocated_document_is_required_by_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['main.tex', 'supplement.tex']:
                (root/name).write_text('journal')
            contract = dict(documents=['main.tex', 'supplement.tex', 'additional_results.tex'])
            with self.assertRaises(FileNotFoundError):
                load_documents(root, contract)
            (root/'additional_results.tex').write_text('retained evidence')
            self.assertEqual(load_documents(root, contract)['additional_results.tex'], 'retained evidence')

    def test_combined_manuscript_does_not_require_a_separate_supplement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'main.tex').write_text('main and supplement')
            (root/'additional_results.tex').write_text('package evidence')
            documents = load_documents(root, dict(documents=['main.tex', 'additional_results.tex']))
            self.assertEqual(set(documents), {'main.tex', 'additional_results.tex'})
            self.assertEqual(documents['main.tex'], 'main and supplement')

    def test_document_contract_cannot_omit_journal_or_escape_folder(self):
        for names in [['additional_results.tex'], ['main.tex', 'supplement.tex', '../other.tex'],
                      ['main.tex', 'supplement.tex', 'main.tex']]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                load_documents(Path('.'), dict(documents=names))

    def test_numerical_mode_reports_pixel_differences_but_requires_figures(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); manuscript=root/'manuscript'; manuscript.mkdir()
            workspace=Workspace(root/'data',root/'out')
            for folder in [workspace.data/'reference',workspace.output/'tables',workspace.output/'figures',manuscript/'figures']:
                folder.mkdir(parents=True,exist_ok=True)
            text='% BEGIN GENERATED: example\n42\n% END GENERATED: example\n'
            for i in range(15):
                name=f'{i}.png'; text+='\\includegraphics{'+name+'}\n'
                Image.new('RGB',(2,2),'white').save(manuscript/'figures'/name)
                Image.new('RGB',(2,2),'black').save(workspace.output/'figures'/name)
            text = text.replace('\\includegraphics{14.png}\n', '')
            (manuscript/'main.tex').write_text(text); (manuscript/'supplement.tex').write_text('')
            extra='% BEGIN GENERATED: retained\n99\n% END GENERATED: retained\n\\includegraphics{14.png}\n'
            (manuscript/'additional_results.tex').write_text(extra)
            (workspace.output/'tables/retained.tex').write_text('99')
            (manuscript/'numbers.tex').write_text(''); (workspace.output/'tables/numbers.tex').write_text('')
            (workspace.output/'tables/example.tex').write_text('42')
            (workspace.data/'reference/manuscript_claims.json').write_text(json.dumps(dict(numerical_claims=[],coverage=[],scope='test',documents=['main.tex','supplement.tex','additional_results.tex'])))
            with self.assertRaisesRegex(ValueError,'pixels differ'):
                verify(workspace,manuscript)
            report=verify(workspace,manuscript,exact_figures=False)
            self.assertFalse(report['figure_pixels_verified'])
            self.assertEqual(report['validation_mode'],'numerical_only')
            self.assertIn('retained', report['generated_blocks'])
            self.assertEqual(report['document_inventory']['additional_results.tex']['figures'], 1)
            contract_path = workspace.data/'reference/manuscript_claims.json'
            contract = json.loads(contract_path.read_text())
            contract['numerical_claims'] = [dict(id='finite-result', document='main.tex',
                summary='synthetic.json', path=['value'], expected=1., atol=.01)]
            contract_path.write_text(json.dumps(contract))
            (workspace.output/'summaries').mkdir()
            for bad in [float('nan'), float('inf'), -float('inf')]:
                (workspace.output/'summaries/synthetic.json').write_text(json.dumps(dict(value=bad)))
                with self.subTest(value=bad), self.assertRaisesRegex(ValueError, 'nonfinite'):
                    verify(workspace, manuscript, exact_figures=False)
            contract['numerical_claims'] = []
            contract_path.write_text(json.dumps(contract))
            (manuscript/'additional_results.tex').write_text(extra.replace('99', '98'))
            with self.assertRaisesRegex(ValueError, 'stale generated block'):
                verify(workspace,manuscript,exact_figures=False)
            (manuscript/'additional_results.tex').write_text(extra)
            (workspace.output/'figures/0.png').unlink()
            with self.assertRaisesRegex(ValueError,'Missing generated figure'):
                verify(workspace,manuscript,exact_figures=False)

    def test_dense_comparison_accepts_roundoff_and_equivalent_axes(self):
        fit = dict(A_fit=np.array([2.]), theta_fit=np.array([0.]), fallback=np.array([False]))
        compare_dense_fit([2.+2e-12], [np.pi-1e-12], [False], fit)

    def test_dense_comparison_rejects_real_changes(self):
        fit = dict(A_fit=np.array([2.]), theta_fit=np.array([0.]), fallback=np.array([False]))
        for a, b, f in [([2.001], [0], [False]), ([2], [.001], [False]), ([2], [0], [True])]:
            with self.subTest(a=a, b=b, f=f), self.assertRaises(AssertionError):
                compare_dense_fit(a, b, f, fit)

    def test_environment_records_pin_mismatches_and_missing_requirements(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'requirements.txt'; path.write_text('numpy==2.3.5\n')
            with patch('etissot.validation.environment.version', return_value='2.0.0'):
                self.assertFalse(inspect_environment(path)['matches_pins'])
            self.assertIsNone(inspect_environment(Path(temp)/'missing')['matches_pins'])

    def test_clean_prepare_does_not_copy_finished_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for folder in ['figures','tables','summaries','baselines']:
                p=root/'data/reference'/folder/'sentinel';p.parent.mkdir(parents=True);p.write_text('not generated')
            w=Workspace(root/'data',root/'out').prepare(seed_reference=False)
            for folder in ['figures','tables','summaries']:
                self.assertFalse((w.output/folder/'sentinel').exists())
            self.assertTrue((w.output/'baselines/sentinel').exists())

    def test_manuscript_check_rejects_missing_generated_table(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'Missing or stale'):
                compare_generated_blocks({'main.tex':'% BEGIN GENERATED: result\n42\n% END GENERATED: result'},temp)

    def test_manuscript_check_rejects_changed_number(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'tables';folder.mkdir();(folder/'result.tex').write_text('43')
            with self.assertRaisesRegex(ValueError,'Missing or stale'):
                compare_generated_blocks({'main.tex':'% BEGIN GENERATED: result\n42\n% END GENERATED: result'},temp)

    def test_manuscript_check_accepts_generated_content_and_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'tables';folder.mkdir();(folder/'result.tex').write_text('42\n')
            document='% BEGIN GENERATED: result\n42\n'+r'\relax\space% Preserve the original input-file boundary.'+'\n% END GENERATED: result'
            self.assertEqual(compare_generated_blocks({'main.tex':document},temp),['result'])

    def test_neighbour_shape_for_complete_isotropic_ring(self):
        b=np.arange(32)*2*np.pi/32
        result=query_metrics(0.,0.,.001*np.cos(b),.001*np.sin(b),np.arange(32.),32)
        self.assertAlmostEqual(result['embedding_elongation'],1.,places=12)
        self.assertAlmostEqual(result['geodesic_elongation'],1.,places=12)

if __name__=='__main__':unittest.main()

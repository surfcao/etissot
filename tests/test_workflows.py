"""Temporary-directory workflow tests; heavy scientific modules are mocked."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from etissot.workspace import Workspace
from etissot.pipeline import run
from etissot.cli import main


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.workspace = Workspace(root/'data', root/'outputs')
        reference = self.workspace.data/'reference/summaries'
        reference.mkdir(parents=True)
        (reference/'example.json').write_text('{"value": 1}')

    def test_optimized_python_cannot_report_skipped_numerical_checks_as_passed(self):
        import subprocess, sys
        result = subprocess.run([sys.executable, '-O', '-c',
            'from etissot.pipeline import run; run(None, check_only=True)'],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('without optimization', result.stderr)

    def test_missing_optional_study_dependencies_fail_before_workspace_creation(self):
        with patch('importlib.util.find_spec', return_value=None), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['check', '--data', str(self.workspace.data), '--output', str(self.workspace.output)])
        self.assertFalse(self.workspace.output.exists())

    def test_prepare_preserves_existing_outputs_and_reference(self):
        work = self.workspace.prepare()
        output = work.output/'summaries/example.json'
        self.assertEqual(json.loads(output.read_text()), {'value': 1})
        output.write_text('{"value": 2}')
        work.prepare()
        self.assertEqual(json.loads(output.read_text()), {'value': 2})
        self.assertEqual(json.loads((work.data/'reference/summaries/example.json').read_text()), {'value': 1})

    def test_missing_reference_data_produces_clear_error(self):
        root = Path(self.temp.name)
        work = Workspace(root/'missing', root/'another-output')
        with self.assertRaisesRegex(FileNotFoundError, 'Missing study reference data'):
            work.prepare()
        self.assertFalse(work.output.exists())

    def test_unknown_region_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown region'):
            self.workspace.fit_path('../outside')

    def test_selected_analysis_logs_output_and_success(self):
        def analysis(workspace):
            print('synthetic analysis completed')
        with patch('etissot.pipeline.import_module', return_value=SimpleNamespace(run=analysis)) as loader:
            report = run(self.workspace, steps=['shape'])
        loader.assert_called_once_with('etissot.diagnostics.shape')
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['status'], 'passed')
        self.assertIn('synthetic analysis completed', (self.workspace.output/'logs/diagnostics.shape.log').read_text())
        self.assertEqual(json.loads((self.workspace.output/'run_report.json').read_text()), report)

    def test_failure_is_recorded_and_later_steps_do_not_run(self):
        def fail(workspace):
            raise RuntimeError('synthetic failure')
        with patch('etissot.pipeline.import_module', return_value=SimpleNamespace(run=fail)) as loader:
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                run(self.workspace, steps=['shape', 'downstream'])
        self.assertEqual(loader.call_count, 1)
        report = json.loads((self.workspace.output/'run_report.json').read_text())
        self.assertEqual(report[0]['status'], 'failed')
        self.assertEqual(report[0]['error'], 'synthetic failure')

    def test_check_only_dispatches_validators_not_generators(self):
        with patch('etissot.pipeline.import_module', return_value=SimpleNamespace(run=lambda _: None)) as loader:
            report = run(self.workspace, check_only=True)
        self.assertEqual(len(report), 3)
        self.assertTrue(all(call.args[0].startswith('etissot.validation.') for call in loader.call_args_list))

    def test_cli_passes_requested_analysis_and_paths(self):
        with patch('etissot.pipeline.run', return_value=[] ) as runner, patch.dict('os.environ'), contextlib.redirect_stdout(io.StringIO()):
            main(['analyze', 'shape', '--data', str(self.workspace.data), '--output', str(self.workspace.output)])
        runner.assert_called_once()
        self.assertEqual(runner.call_args.args[0], self.workspace)
        self.assertEqual(runner.call_args.kwargs, {'check_only': False, 'steps': ['shape']})

    def test_cli_accepts_every_registered_analysis(self):
        from etissot.pipeline import ANALYSES
        for name in ANALYSES:
            with self.subTest(name=name), patch('etissot.pipeline.run', return_value=[]) as runner, patch.dict('os.environ'), contextlib.redirect_stdout(io.StringIO()):
                main(['analyze', name, '--data', str(self.workspace.data), '--output', str(self.workspace.output)])
                self.assertEqual(runner.call_args.kwargs['steps'], [name])

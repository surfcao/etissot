"""Checkpoint acquisition, preservation, packaging and distribution boundaries."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from etissot.cli import main
from etissot.models.checkpoints import load_manifest, download_models, verify_models, bundle_models


class Response(io.BytesIO):
    def geturl(self):
        return 'https://publisher.example/checkpoint'


class CheckpointAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / 'models'
        self.manifest = self.root / 'checkpoints.json'
        self.contents = {'one': b'first checkpoint', 'two': b'second checkpoint', 'held': b'held checkpoint'}
        self.data = dict(schema_version=1, package_version='test', checkpoints=[], external_sources=[])
        for identifier, raw in self.contents.items():
            self.data['checkpoints'].append(dict(
                id=identifier, family='test', path='weights/' + identifier + '.pth',
                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                source=dict(kind='file', url='https://publisher.example/' + identifier),
                license=dict(id='test-only', redistribution=identifier != 'held',
                             evidence_files=['license.txt'],
                             evidence_sha256={'license.txt': hashlib.sha256(b'Test-only license evidence for synthetic bytes.').hexdigest()},
                             note='Test fixture: no terms established for held')))
        self.save()
        (self.root / 'license.txt').write_text('Test-only license evidence for synthetic bytes.')

    def save(self):
        self.manifest.write_text(json.dumps(self.data))

    def install(self):
        for entry in self.data['checkpoints']:
            path = self.models / entry['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.contents[entry['id']])

    def response(self, request, **kwargs):
        return Response(self.contents[request.full_url.rsplit('/', 1)[1]])

    def test_catalog_matches_exact_scientific_checkpoint_contract(self):
        catalog = load_manifest()
        reference = Path(__file__).resolve().parents[1] / 'data/reference/summaries/checkpoint_manifest.json'
        expected = json.loads(reference.read_text())
        self.assertEqual(len(catalog['checkpoints']), 13)
        self.assertEqual(len(catalog['external_sources']), 4)
        self.assertEqual(sum(item['license']['redistribution'] for item in catalog['checkpoints']), 8)
        for entry in catalog['checkpoints']:
            original = expected[entry['label']]
            self.assertEqual(entry['sha256'], original['sha256'])
            self.assertEqual(entry['normalization'], original['normalization'])
            self.assertEqual(entry['feature_dimension'], original['dim'])
            self.assertEqual(entry['radius_km'], original['radius'])

    def test_manifest_rejects_escaping_paths_and_non_https_sources(self):
        for path in ['../escape', '/absolute', 'C:\\weights', 'weights/../../escape']:
            with self.subTest(path=path):
                self.data['checkpoints'][0]['path'] = path
                self.save()
                with self.assertRaises(ValueError):
                    load_manifest(self.manifest)
        self.data['checkpoints'][0]['path'] = 'weights/one.pth'
        self.data['checkpoints'][0]['source']['url'] = 'http://publisher.example/one'
        self.save()
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)
        self.data['checkpoints'][0]['source']['url'] = 'https://publisher.example/one'
        for version in ['../../escape', '0.2/../../escape', None]:
            with self.subTest(version=version):
                self.data['package_version'] = version
                self.save()
                with self.assertRaises(ValueError):
                    load_manifest(self.manifest)

    def test_verification_distinguishes_missing_matching_and_changed_bytes(self):
        report = verify_models(self.models, manifest=self.manifest)
        self.assertEqual({entry['status'] for entry in report['checkpoints']}, {'missing'})
        self.install()
        self.assertEqual(verify_models(self.models, manifest=self.manifest)['status'], 'passed')
        (self.models / 'weights/one.pth').write_bytes(b'altered checkpoint')
        report = verify_models(self.models, manifest=self.manifest)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['checkpoints'][0]['status'], 'mismatch')

    def test_historical_alias_is_verified_without_duplicate_download(self):
        self.data['checkpoints'][0]['aliases'] = ['legacy/one.pth']; self.save()
        path = self.models / 'legacy/one.pth'
        path.parent.mkdir(parents=True); path.write_bytes(self.contents['one'])
        with patch('etissot.models.checkpoints.urlopen') as network:
            result = download_models(self.models, manifest=self.manifest, encoders=['one'])
        network.assert_not_called()
        self.assertEqual(result['files'][0]['acquisition'], 'existing')
        self.assertFalse((self.models / 'weights/one.pth').exists())

    def test_valid_download_installs_exact_bytes_and_records_receipt(self):
        with patch('etissot.models.checkpoints.urlopen', side_effect=self.response):
            result = download_models(self.models, manifest=self.manifest)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(verify_models(self.models, manifest=self.manifest)['status'], 'passed')
        self.assertEqual(json.loads((self.models / 'acquisition.json').read_text())['status'], 'passed')

    def test_bad_or_interrupted_download_leaves_no_checkpoint_or_partial_file(self):
        from http.client import IncompleteRead
        class Interrupted(Response):
            def read(self, *args):
                raise OSError('Connection interrupted')
        class Truncated(Response):
            def read(self, *args):
                raise IncompleteRead(b'partial bytes', 100)
        for response in [Response(b'wrong'), Response(b'x' * 500), Interrupted(b''), Truncated(b'')]:
            with self.subTest(response=type(response).__name__), patch('etissot.models.checkpoints.urlopen', return_value=response):
                result = download_models(self.models, manifest=self.manifest, encoders=['one'])
            self.assertEqual(result['status'], 'failed')
            self.assertFalse((self.models / 'weights/one.pth').exists())
            self.assertFalse(list(self.models.rglob('.etissot-*')))

    def test_existing_mismatch_needs_replace_and_bad_replacement_preserves_original(self):
        self.install()
        path = self.models / 'weights/one.pth'; path.write_bytes(b'keep original')
        with patch('etissot.models.checkpoints.urlopen') as network:
            result = download_models(self.models, manifest=self.manifest, encoders=['one'])
        network.assert_not_called()
        self.assertEqual(result['status'], 'failed')
        with patch('etissot.models.checkpoints.urlopen', return_value=Response(b'bad replacement')):
            download_models(self.models, manifest=self.manifest, encoders=['one'], replace=True)
        self.assertEqual(path.read_bytes(), b'keep original')
        with patch('etissot.models.checkpoints.urlopen', side_effect=self.response):
            result = download_models(self.models, manifest=self.manifest, encoders=['one'], replace=True)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(path.read_bytes(), self.contents['one'])

    def test_dry_run_does_not_download_or_create_directories(self):
        with patch('etissot.models.checkpoints.urlopen') as network:
            report = download_models(self.models, manifest=self.manifest, dry_run=True)
        network.assert_not_called()
        self.assertFalse(self.models.exists())
        self.assertEqual({entry['status'] for entry in report['files']}, {'planned'})

    def zip_manifest(self, wrong_member=False):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, 'w') as archive:
            for key, content in self.contents.items():
                archive.writestr('original/' + key, b'wrong' if wrong_member and key == 'one' else content)
            archive.writestr('../escape.txt', b'not extracted')
        data = raw.getvalue()
        for item in self.data['checkpoints']:
            item['source'] = dict(kind='zip', url='https://publisher.example/weights.zip',
                member='original/' + item['id'], archive_sha256=hashlib.sha256(data).hexdigest(), archive_bytes=len(data))
        self.save()
        return data

    def test_shared_zip_downloaded_once_and_only_selected_members_installed(self):
        data = self.zip_manifest()
        with patch('etissot.models.checkpoints.urlopen', return_value=Response(data)) as network:
            result = download_models(self.models, manifest=self.manifest, encoders=['one', 'two'])
        self.assertEqual(network.call_count, 1)
        self.assertEqual(result['status'], 'passed')
        self.assertFalse((self.models / 'weights/held.pth').exists())
        self.assertFalse((self.root / 'escape.txt').exists())

    def test_zip_with_wrong_member_is_rejected(self):
        data = self.zip_manifest(wrong_member=True)
        with patch('etissot.models.checkpoints.urlopen', return_value=Response(data)):
            result = download_models(self.models, manifest=self.manifest, encoders=['one'])
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((self.models / 'weights/one.pth').exists())

    def test_symlink_cannot_write_outside_model_directory(self):
        self.models.mkdir(); outside = self.root / 'outside'; outside.mkdir()
        (self.models / 'weights').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            download_models(self.models, manifest=self.manifest)
        self.assertFalse(list(outside.iterdir()))

    def test_bundle_contains_only_eligible_unmodified_weights_and_license_evidence(self):
        self.install()
        output = self.root / 'models.zip'
        report = bundle_models(self.models, output, manifest=self.manifest, resource_root=self.root)
        self.assertEqual(report['included'], ['one', 'two'])
        self.assertEqual(report['omitted'][0]['id'], 'held')
        with zipfile.ZipFile(output) as archive:
            prefix = 'etissot-models-test/'
            self.assertNotIn(prefix + 'weights/held.pth', archive.namelist())
            self.assertEqual(archive.read(prefix + 'weights/one.pth'), self.contents['one'])
            self.assertIn(prefix + 'license.txt', archive.namelist())
            for line in archive.read(prefix + 'SHA256SUMS.txt').decode().splitlines():
                expected, name = line.split(maxsplit=1)
                self.assertEqual(hashlib.sha256(archive.read(prefix + name)).hexdigest(), expected)
        with self.assertRaises(ValueError):
            bundle_models(self.models, self.root / 'held.zip', manifest=self.manifest, encoders=['held'], resource_root=self.root)

    def test_missing_evidence_or_altered_weight_prevents_bundle_creation(self):
        self.install()
        (self.root / 'license.txt').unlink()
        with self.assertRaises(ValueError):
            bundle_models(self.models, self.root / 'missing.zip', manifest=self.manifest, resource_root=self.root)
        (self.root / 'license.txt').write_text('Test-only license evidence for synthetic bytes.')
        (self.models / 'weights/one.pth').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            bundle_models(self.models, self.root / 'changed.zip', manifest=self.manifest, resource_root=self.root)
        self.assertFalse((self.root / 'changed.zip').exists())

    def test_changed_license_evidence_cannot_be_bundled(self):
        self.install()
        (self.root / 'license.txt').write_text('altered license')
        with self.assertRaisesRegex(ValueError, 'License evidence differs'):
            bundle_models(self.models, self.root / 'changed-license.zip', manifest=self.manifest, resource_root=self.root)

    def test_receipt_symlink_and_bundle_input_collision_are_rejected(self):
        self.models.mkdir()
        outside = self.root / 'outside.json'; outside.write_text('keep')
        (self.models / 'acquisition.json').symlink_to(outside)
        with self.assertRaises(ValueError):
            download_models(self.models, manifest=self.manifest)
        self.assertEqual(outside.read_text(), 'keep')
        self.install()
        checkpoint = self.models / 'weights/one.pth'
        with self.assertRaisesRegex(ValueError, 'input checkpoint'):
            bundle_models(self.models, checkpoint, manifest=self.manifest, resource_root=self.root, replace=True)
        self.assertEqual(checkpoint.read_bytes(), self.contents['one'])

    def test_external_source_files_are_obtained_and_verified_for_selected_family(self):
        entry = dict(self.data['checkpoints'][0], id='source', path='runtime/source.py')
        entry.pop('license')
        self.data['external_sources'] = [entry]; self.save()
        with patch('etissot.models.checkpoints.urlopen', side_effect=self.response):
            result = download_models(self.models, manifest=self.manifest, encoders=['two'])
        self.assertEqual(result['status'], 'passed')
        self.assertEqual((self.models / 'runtime/source.py').read_bytes(), self.contents['one'])
        report = verify_models(self.models, manifest=self.manifest, encoders=['two'])
        self.assertEqual(report['external_sources'][0]['status'], 'passed')

    def test_cli_model_inventory_does_not_require_study_dependencies(self):
        with patch('importlib.util.find_spec', return_value=None), redirect_stdout(io.StringIO()) as output:
            main(['models', 'list', '--manifest', str(self.manifest)])
        self.assertIn('held', output.getvalue())
        self.assertFalse(self.models.exists())
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['models', 'verify', '--models', str(self.models), '--manifest', str(self.manifest)])
        self.assertEqual(error.exception.code, 1)

    def test_acquisition_module_never_imports_model_frameworks(self):
        code = '''
import importlib.abc, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'matplotlib', 'pyproj', 'geoclip', 'rshf'}:
            raise ImportError('Unexpected framework import: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
from etissot.models.checkpoints import load_manifest
assert len(load_manifest()['checkpoints']) == 13
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_study_loader_prefers_explicit_geoclip_weights(self):
        from etissot.models.loaders import load_all
        path = self.models / 'encoders/geoclip/location_encoder_weights.pth'
        path.parent.mkdir(parents=True); path.write_bytes(b'explicit')
        with patch('etissot.models.loaders.load_geoclip_checkpoint', return_value=object()) as loader, \
             patch('etissot.models.loaders.package_root', side_effect=AssertionError('Installed weights must not be used')), \
             patch('etissot.models.loaders.filemodule', return_value=SimpleNamespace()):
            with self.assertRaises(FileNotFoundError):
                load_all(self.models)
        loader.assert_called_once_with(path.resolve())


if __name__ == '__main__':
    unittest.main()

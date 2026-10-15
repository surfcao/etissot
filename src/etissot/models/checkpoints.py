"""Acquire and verify the study checkpoints without importing or loading models.

Sources and redistribution decisions belong to the versioned manifest. Downloads
are explicit, streamed, checked before installation, and never deserialize weights.
"""
from importlib import util
from http.client import HTTPException
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import zipfile

RESOURCE_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = RESOURCE_ROOT / 'checkpoints.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative_path(value):
    """Reject paths that could leave the chosen checkpoint directory."""
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or '\x00' in value:
        raise ValueError(f'Invalid relative path: {value!r}')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or str(path) == '.':
        raise ValueError(f'Invalid relative path: {value!r}')
    return path


def under(root, relative):
    root = Path(root).resolve()
    target = root.joinpath(*relative_path(relative).parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'Path escapes the checkpoint directory: {relative}')
    return target


def load_manifest(path=None):
    data = json.loads(Path(path or DEFAULT_MANIFEST).read_text())
    if data.get('schema_version') != 1 or not data.get('checkpoints'):
        raise ValueError('Unsupported or empty checkpoint manifest')
    version = data.get('package_version')
    if not isinstance(version, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', version):
        raise ValueError('Invalid package version for the checkpoint bundle prefix')
    ids, paths = set(), set()
    for item in data['checkpoints'] + data.get('external_sources', []):
        if item['id'] in ids or item['path'] in paths:
            raise ValueError('Duplicate checkpoint/source id or destination')
        ids.add(item['id']); paths.add(item['path'])
        relative_path(item['path'])
        for alias in item.get('aliases', []):
            relative_path(alias)
        if not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError(f'Invalid SHA-256 for {item["id"]}')
        if type(item['bytes']) is not int or item['bytes'] <= 0:
            raise ValueError(f'Invalid file size for {item["id"]}')
        source = item['source']
        if source['kind'] not in {'file', 'zip'}:
            raise ValueError(f'Unsupported source for {item["id"]}')
        if urlparse(source['url']).scheme != 'https' or not urlparse(source['url']).netloc:
            raise ValueError('Checkpoint sources must use HTTPS')
        if source['kind'] == 'zip':
            relative_path(source['member'])
            if not re.fullmatch('[0-9a-f]{64}', source['archive_sha256']):
                raise ValueError('ZIP sources require an archive SHA-256')
            if type(source['archive_bytes']) is not int or source['archive_bytes'] <= 0:
                raise ValueError('ZIP sources require a positive archive size')
        if 'license' in item:
            license_info = item['license']
            if type(license_info['redistribution']) is not bool:
                raise ValueError('Redistribution must be an explicit Boolean')
            for evidence in license_info.get('evidence_files', []):
                relative_path(evidence)
            evidence_hashes = license_info.get('evidence_sha256', {})
            if license_info['redistribution'] and (not license_info.get('id') or
                    not license_info.get('evidence_files') or
                    set(evidence_hashes) != set(license_info['evidence_files'])):
                raise ValueError('Eligible checkpoints require complete license evidence and hashes')
            if any(not re.fullmatch('[0-9a-f]{64}', checksum) for checksum in evidence_hashes.values()):
                raise ValueError('Invalid license-evidence SHA-256')
    return data


def select_checkpoints(manifest, families=None, encoders=None):
    families = set(families or [])
    encoders = set(encoders or [])
    items = manifest['checkpoints']
    if families - {item['family'] for item in items}:
        raise ValueError('Unknown encoder family')
    if encoders - {item['id'] for item in items}:
        raise ValueError('Unknown encoder id')
    selected = [item for item in items if (not families or item['family'] in families)
                and (not encoders or item['id'] in encoders)]
    if not selected:
        raise ValueError('No encoders match the selection')
    return selected


def source_items(manifest, checkpoints):
    families = {item['family'] for item in checkpoints}
    return [item for item in manifest.get('external_sources', []) if item['family'] in families]


def locate(model_dir, item):
    """Prefer a supplied checkpoint; recognize documented historical locations."""
    for name in [item['path']] + item.get('aliases', []):
        path = under(model_dir, name)
        if path.exists():
            return path
    package = item.get('installed_package')
    if package:
        spec = util.find_spec(package['name'])
        if spec is not None and spec.origin is not None:
            candidate = under(Path(spec.origin).parent, package['path'])
            if candidate.exists():
                return candidate
    return under(model_dir, item['path'])


def inspect_file(item, path):
    result = dict(id=item['id'], path=str(path), expected_sha256=item['sha256'])
    if not path.is_file():
        result.update(status='missing')
    else:
        actual = digest(path)
        result.update(sha256=actual, bytes=path.stat().st_size,
                      status='passed' if actual == item['sha256'] and path.stat().st_size == item['bytes'] else 'mismatch')
    return result


def verify_models(model_dir, *, manifest=None, families=None, encoders=None):
    """Verify checkpoint bytes and selected external source; assess no runtime."""
    data = load_manifest(manifest)
    selected = select_checkpoints(data, families, encoders)
    checkpoints = [inspect_file(item, locate(model_dir, item)) for item in selected]
    sources = [inspect_file(item, under(model_dir, item['path'])) for item in source_items(data, selected)]
    return dict(status='passed' if all(item['status'] == 'passed' for item in checkpoints + sources) else 'failed',
                checkpoints=checkpoints, external_sources=sources, runtime_readiness='not assessed',
                manifest_sha256=digest(manifest or DEFAULT_MANIFEST))


def _copy_checked(stream, destination, expected_hash, expected_bytes):
    """Check a bounded stream in a sibling temporary file before atomic install."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.etissot-', delete=False) as target:
        temporary = Path(target.name)
        try:
            size, checksum = 0, hashlib.sha256()
            while block := stream.read(1024 * 1024):
                size += len(block)
                if size > expected_bytes:
                    raise ValueError('Downloaded file exceeds the declared size')
                checksum.update(block)
                target.write(block)
            if size != expected_bytes or checksum.hexdigest() != expected_hash:
                raise ValueError('Downloaded file does not match the declared size and SHA-256')
            target.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _fetch(url, destination, expected_hash, expected_bytes, timeout):
    request = Request(url, headers={'User-Agent': 'eTissot checkpoint acquisition'})
    with urlopen(request, timeout=timeout) as response:
        if urlparse(response.geturl()).scheme != 'https':
            raise ValueError('Checkpoint download redirected away from HTTPS')
        _copy_checked(response, destination, expected_hash, expected_bytes)


def download_models(model_dir, *, manifest=None, families=None, encoders=None,
                    replace=False, dry_run=False, timeout=60):
    """Obtain exact originals; existing mismatches require explicit replacement.

    ZIPs are downloaded once per call. Only declared members are streamed out;
    no archive is extracted wholesale. Individual failures remain in the receipt.
    """
    if not isinstance(timeout, (int, float)) or not 0 < timeout < float('inf'):
        raise ValueError('Timeout must be positive and finite')
    data = load_manifest(manifest)
    selected = select_checkpoints(data, families, encoders)
    items = selected + source_items(data, selected)
    root = Path(model_dir).resolve()
    receipt = under(root, 'acquisition.json') if not dry_run else None
    results = []
    with tempfile.TemporaryDirectory(prefix='etissot-acquisition-') as scratch:
        archives = {}
        for item in items:
            record = dict(id=item['id'], path=str(under(root, item['path'])), url=item['source']['url'])
            try:
                found = locate(root, item) if item in selected else under(root, item['path'])
                current = inspect_file(item, found)
                if current['status'] == 'passed':
                    record.update(current, acquisition='existing')
                elif current['status'] == 'mismatch' and not replace:
                    raise ValueError('Existing file differs from the manifest; use --replace to replace it')
                elif dry_run:
                    record.update(status='planned', acquisition='download')
                else:
                    destination = under(root, item['path'])
                    source = item['source']
                    if source['kind'] == 'file':
                        _fetch(source['url'], destination, item['sha256'], item['bytes'], timeout)
                    else:
                        key = (source['url'], source['archive_sha256'])
                        if key not in archives:
                            archive = Path(scratch) / f'{len(archives)}.zip'
                            _fetch(source['url'], archive, source['archive_sha256'], source['archive_bytes'], timeout)
                            archives[key] = archive
                        with zipfile.ZipFile(archives[key]) as archive:
                            matches = [entry for entry in archive.infolist() if entry.filename == source['member']]
                            if len(matches) != 1 or matches[0].file_size != item['bytes']:
                                raise ValueError('ZIP member is missing, duplicated, or has the wrong size')
                            with archive.open(matches[0]) as stream:
                                _copy_checked(stream, destination, item['sha256'], item['bytes'])
                    record.update(inspect_file(item, destination), acquisition='downloaded')
            except (OSError, HTTPException, ValueError, KeyError, zipfile.BadZipFile) as error:
                record.update(status='failed', error=str(error))
            results.append(record)
    report = dict(status='passed' if all(item['status'] in {'passed', 'planned'} for item in results) else 'failed',
                  dry_run=bool(dry_run), files=results, manifest_sha256=digest(manifest or DEFAULT_MANIFEST),
                  runtime_readiness='not assessed')
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, prefix='.etissot-receipt-', mode='w', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(json.dumps(report, indent=2) + '\n')
                stream.close()
                temporary.replace(receipt)
            finally:
                temporary.unlink(missing_ok=True)
    return report


def bundle_models(model_dir, output, *, manifest=None, families=None, encoders=None,
                  resource_root=None, replace=False):
    """Bundle only verified files with explicit, recorded redistribution terms."""
    data = load_manifest(manifest)
    selected = select_checkpoints(data, families, encoders)
    eligible = [item for item in selected if item['license']['redistribution']]
    omitted = [dict(id=item['id'], reason=item['license']['note']) for item in selected if item not in eligible]
    if (families or encoders) and omitted:
        raise ValueError('Selection includes checkpoints without established redistribution terms: ' +
                         ', '.join(item['id'] for item in omitted))
    if not eligible:
        raise ValueError('No checkpoints have established redistribution terms')
    output = Path(output).resolve()
    if output.exists() and not replace:
        raise FileExistsError(output)
    resources = Path(resource_root or RESOURCE_ROOT).resolve()
    members = {}
    for item in eligible:
        path = locate(model_dir, item)
        if inspect_file(item, path)['status'] != 'passed':
            raise ValueError(f'Checkpoint is missing or differs from the manifest: {item["id"]}')
        members[item['path']] = path
        evidence = item['license'].get('evidence_files', [])
        if not item['license'].get('id') or not evidence:
            raise ValueError(f'Missing license evidence for {item["id"]}')
        for name in evidence:
            path = under(resources, name)
            if not path.is_file():
                raise ValueError(f'Missing license evidence: {name}')
            if digest(path) != item['license']['evidence_sha256'][name]:
                raise ValueError(f'License evidence differs from the catalog: {name}')
            members[name] = path
    if any(output == path.resolve() for path in members.values()):
        raise ValueError('Bundle output cannot replace an input checkpoint or license file')
    info = dict(schema_version=1, name='eTissot optional encoder checkpoints',
                included=[item['id'] for item in eligible], omitted=omitted,
                original_weights_unmodified=True, manifest_sha256=digest(manifest or DEFAULT_MANIFEST))
    text_members = {
        'checkpoints.json': Path(manifest or DEFAULT_MANIFEST).read_text(),
        'BUNDLE.json': json.dumps(info, indent=2) + '\n',
        'README.md': '# eTissot optional encoder checkpoints\n\n'
            'Extract this directory and use it as --models in eTissot. Original checkpoint bytes are unchanged. '
            'BUNDLE.json lists included models and omissions; checkpoints.json records exact sources, hashes, '
            'representation conventions and publisher terms. The full study also needs any omitted checkpoints '
            'and the dependencies described in the main package docs/models.md.\n\n'
            'Obtain missing files with `etissot models download --models /path/to/this/directory`. '
            'Verify with `etissot models verify --models /path/to/this/directory`. These commands never load '
            'weights; verification does not establish runtime compatibility. Preserve the supplied model cards '
            'and license notices when sharing this bundle.\n',
    }
    prefix = 'etissot-models-' + data['package_version']
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.etissot-bundle-', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        sums = []
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(members.items()):
                archive.write(path, prefix + '/' + name)
                sums.append(digest(path) + '  ' + name + '\n')
            for name, text in text_members.items():
                raw = text.encode()
                archive.writestr(prefix + '/' + name, raw)
                sums.append(hashlib.sha256(raw).hexdigest() + '  ' + name + '\n')
            archive.writestr(prefix + '/SHA256SUMS.txt', ''.join(sums))
        # Read back every member and compare both manifest and expected model bytes.
        with zipfile.ZipFile(temporary) as archive:
            for line in sums:
                expected, name = line.split(maxsplit=1)
                with archive.open(prefix + '/' + name.strip()) as source:
                    actual = hashlib.file_digest(source, 'sha256').hexdigest()
                if actual != expected:
                    raise ValueError(f'Bundle member checksum mismatch: {name}')
            for item in eligible:
                with archive.open(prefix + '/' + item['path']) as source:
                    if hashlib.file_digest(source, 'sha256').hexdigest() != item['sha256']:
                        raise ValueError(f'Checkpoint changed during bundling: {item["id"]}')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = digest(output)
    Path(str(output) + '.sha256').write_text(checksum + '  ' + output.name + '\n')
    return dict(status='passed', archive=str(output), sha256=checksum, bytes=output.stat().st_size,
                verified_files=len(sums), included=info['included'], omitted=omitted)

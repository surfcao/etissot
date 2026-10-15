"""Verify frozen share archives and each manifest member without extracting them."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    if (ROOT / 'SHA256SUMS.txt').exists():
        lines = (ROOT / 'SHA256SUMS.txt').read_text().splitlines()
        for line in lines:
            expected, relative = line.split(maxsplit=1)
            with (ROOT / relative).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise SystemExit(f'File checksum mismatch: {relative}')
        print(f'Verified {len(lines)} release files')
        return
    names = ['etissot-0.2.0.zip', 'etissot-manuscript-tex.zip']
    companion = ROOT / 'releases/etissot-models-0.2.0.zip'
    if companion.exists():
        names.append(companion.name)
    for name in names:
        path = ROOT / "releases" / name
        expected = Path(str(path) + ".sha256").read_text().split()[0]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise SystemExit(f"Archive checksum mismatch: {name}")
        with zipfile.ZipFile(path) as archive:
            manifests = [p for p in archive.namelist() if p.endswith("/SHA256SUMS.txt")]
            manifests = [p for p in manifests if len(Path(p).parts) == 2]
            if len(manifests) != 1:
                raise SystemExit(f"Expected one manifest: {name}")
            manifest = manifests[0]
            prefix = manifest.rsplit("/", 1)[0] + "/"
            lines = archive.read(manifest).decode().splitlines()
            for line in lines:
                digest, relative = line.split(maxsplit=1)
                with archive.open(prefix + relative) as content:
                    actual = hashlib.file_digest(content, 'sha256').hexdigest()
                if actual != digest:
                    raise SystemExit(f"Member checksum mismatch: {name}: {relative}")
            if name == companion.name:
                info = json.loads(archive.read(prefix + 'BUNDLE.json'))
                catalog = json.loads(archive.read(prefix + 'checkpoints.json'))
                records = {entry['id']: entry for entry in catalog['checkpoints']}
                for identifier in info['included']:
                    entry = records[identifier]
                    if not entry['license']['redistribution']:
                        raise SystemExit(f'Checkpoint lacks recorded redistribution terms: {identifier}')
                    with archive.open(prefix + entry['path']) as stream:
                        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if actual != entry['sha256']:
                        raise SystemExit(f'Checkpoint differs from study catalog: {identifier}')
                    for relative, expected in entry['license']['evidence_sha256'].items():
                        with archive.open(prefix + relative) as stream:
                            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                        if actual != expected:
                            raise SystemExit(f'License evidence differs from study catalog: {relative}')
                for entry in catalog['checkpoints']:
                    present = prefix + entry['path'] in archive.namelist()
                    if present != (entry['id'] in info['included']):
                        raise SystemExit(f'Companion inventory mismatch: {entry["id"]}')
        print(f"Verified {name}: {len(lines)} manifest files")
    with zipfile.ZipFile(ROOT/'releases/etissot-0.2.0.zip') as archive:
        wheels = [name for name in archive.namelist() if '/dist/' in name and name.endswith('.whl')]
        if len(wheels) != 1:
            raise SystemExit('Expected exactly one wheel in the sharing archive')
        expected = hashlib.sha256(archive.read(wheels[0])).hexdigest()
        path = ROOT/'releases'/Path(wheels[0]).name
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise SystemExit('Standalone wheel differs from the verified sharing archive')
        print(f'Verified standalone wheel: {path.name}')

if __name__ == "__main__":
    main()

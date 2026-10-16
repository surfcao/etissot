"""Build a clean, shareable distribution from functional source and named study inputs."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PROJECT=tomllib.loads((ROOT/'pyproject.toml').read_text())['project']
VERSION=PROJECT['version']
NAME=f"{PROJECT['name']}-{VERSION}"

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def stage(destination, include_manuscript=False):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for name in ['src','tests','examples']+(['manuscript'] if include_manuscript else []):
        excluded = ['__pycache__', '*.pyc', '.DS_Store', 'build', '*.egg-info']
        if name == 'manuscript':
            excluded += ['output', '*.pdf', '*.aux', '*.bbl', '*.blg', '*.fls', '*.fdb_latexmk',
                         '*.synctex.gz', '*.toc', '*.out', '*.log', '*_backup_*', 'review_*.md', 'README.md', 'backups', 'notes']
        shutil.copytree(ROOT/name,destination/name,ignore=shutil.ignore_patterns(*excluded))
        if name == 'manuscript':
            for auxiliary in ['main2.tex', 'tags', 'main.pdf', 'SHA256SUMS.txt', 'SOURCE_PROVENANCE.json', 'VERIFICATION.json']:
                (destination/name/auxiliary).unlink(missing_ok=True)
            (destination/name/'SOURCE_NOTICE.txt').write_text('Unmodified working manuscript sources and figures are included for numerical verification. Stale PDFs and historical manuscript manifests are omitted. The author requested review without manuscript edits. This software archive is not a submission-ready manuscript package.\n')
            texts='\n'.join((destination/name/f).read_text() for f in ['main.tex','additional_results.tex'])
            texts=re.sub(r'(?m)^\s*%.*\n?', '', texts)
            used=set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',texts))
            for figure in (destination/name/'figures').glob('*.png'):
                if figure.name not in used:
                    figure.unlink()

    for name in ['README.md','LICENSE','RIGHTS_AND_ATTRIBUTION.md','requirements-reproduce.txt','requirements-figures.txt','pyproject.toml','MANIFEST.in']:
        shutil.copy2(ROOT/name,destination/name)
    (destination/'docs').mkdir();(destination/'docs/validation').mkdir()
    for p in (ROOT/'docs').iterdir():
        if p.is_file() and p.name not in ('writing_style.md','revision_2026-10-13.md'):shutil.copy2(p,destination/'docs'/p.name)
    validation_names = {'reproduction_v0.2.0.json'}
    for name in sorted(validation_names):
        path = ROOT/'docs/validation'/name
        if path.exists():shutil.copy2(path,destination/'docs/validation'/name)
    shutil.copytree(ROOT/'data',destination/'data',ignore=shutil.ignore_patterns('raw','.DS_Store','__pycache__'))
    (destination/'scripts').mkdir()
    shutil.copy2(ROOT/'scripts/verify_releases.py',destination/'scripts/verify_releases.py')
    shutil.copy2(ROOT/'scripts/validate_review_fixes.py',destination/'scripts/validate_review_fixes.py')
    shutil.copy2(ROOT/'scripts/build_release.py',destination/'scripts/build_release.py')
    for name in ['make_revision_figures.py','review_followups.py','neighbourhood_nulls.py','csp_precision_check.py']:
        shutil.copy2(ROOT/'scripts'/name,destination/'scripts'/name)
    with tempfile.TemporaryDirectory(prefix='ld-wheel-build-') as temp:
        build=Path(temp)/'source';build.mkdir()
        shutil.copytree(destination/'src',build/'src')
        for name in ['pyproject.toml','README.md','LICENSE','RIGHTS_AND_ATTRIBUTION.md','MANIFEST.in']:shutil.copy2(destination/name,build/name)
        subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','--wheel-dir',str(destination/'dist'),str(build)],check=True)
    return destination

def package(source):
    source=Path(source).resolve();manifest=source/'SHA256SUMS.txt'
    paths=sorted(p for p in source.rglob('*') if p.is_file() and p!=manifest and '.DS_Store' not in p.parts and '__pycache__' not in p.parts)
    manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(source).as_posix()}\n' for p in paths))
    archive=ROOT/'releases'/f'{NAME}.zip';temporary=archive.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths+[manifest]:z.write(p,str(Path(NAME)/p.relative_to(source)))
    temporary.replace(archive)
    Path(str(archive)+'.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
    for wheel in (source/'dist').glob('*.whl'):
        shutil.copy2(wheel, ROOT/'releases'/wheel.name)
        (ROOT/'releases'/(wheel.name+'.sha256')).write_text(f'{digest(wheel)}  {wheel.name}\n')
    return dict(archive=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,verified_files=len(paths),wheel=f'{NAME}-py3-none-any.whl')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path);p.add_argument('--pack',type=Path);p.add_argument('--with-manuscript',action='store_true',help='include the manuscript sources (after publication)');a=p.parse_args()
    if a.stage:stage(a.stage,a.with_manuscript)
    elif a.pack:print(json.dumps(package(a.pack),indent=2))
    else:
        with tempfile.TemporaryDirectory(prefix='ld-release-') as temp:print(json.dumps(package(stage(Path(temp)/NAME,a.with_manuscript)),indent=2))

if __name__=='__main__':main()

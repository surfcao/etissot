"""Record the runtime separately from scientific reproduction outcomes."""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def inspect_environment(requirements=Path('requirements-reproduce.txt')):
    requirements = Path(requirements)
    packages = {}
    if requirements.is_file():
        for line in requirements.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, expected = line.split('==')
            try:
                actual = version(name)
            except PackageNotFoundError:
                actual = None
            packages[name] = dict(expected=expected, actual=actual, matches=actual == expected)
    return dict(python=platform.python_version(), executable=sys.executable,
                platform=platform.platform(), supported_python=sys.version_info >= (3, 11),
                requirements=str(requirements.resolve()), requirements_found=requirements.is_file(),
                packages=packages,
                matches_pins=all(p['matches'] for p in packages.values()) if packages else None)

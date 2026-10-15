"""Explicit, separate input data and generated-output locations."""
from dataclasses import dataclass
from pathlib import Path
import shutil

@dataclass(frozen=True)
class Workspace:
    data: Path
    output: Path

    def __post_init__(self):
        object.__setattr__(self, 'data', Path(self.data).expanduser().resolve())
        object.__setattr__(self, 'output', Path(self.output).expanduser().resolve())
        if self.output == self.data or self.output.is_relative_to(self.data) or self.data.is_relative_to(self.output):
            raise ValueError('Data and output directories must not overlap.')

    def prepare(self, seed_reference=True):
        """Seed reference summaries for checks without changing supplied data."""
        reference = self.data / 'reference'
        if not reference.is_dir():
            raise FileNotFoundError(f'Missing study reference data: {reference}')
        self.output.mkdir(parents=True, exist_ok=True)
        for source in reference.rglob('*'):
            if source.is_file() and (seed_reference or source.relative_to(reference).parts[0] == 'baselines'):
                target = self.output / source.relative_to(reference)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
        for name in ['prediction', 'profiles', 'figures', 'summaries', 'tables', 'logs']:
            (self.output / name).mkdir(exist_ok=True)
        return self

    def fit_path(self, region):
        """Prefer a new fit in outputs, otherwise use the distributed reference."""
        if region not in {'CONUS', 'Europe', 'Australia', 'TropSAm'}:
            raise ValueError(f'Unknown region: {region}')
        candidate = self.output / 'prediction' / f'fits_{region}.json'
        return candidate if candidate.exists() else self.data / 'prediction' / candidate.name

    def fit_files(self):
        return [self.fit_path(region) for region in ['CONUS', 'Europe', 'Australia', 'TropSAm']]

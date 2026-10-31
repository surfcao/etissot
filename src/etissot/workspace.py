"""Explicit, separate input data and generated-output locations."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil

SEED_RECORD = '.seeded_reference.json'


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

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
        """Seed reference inputs without changing supplied data.

        Baselines are always copied. With ``seed_reference``, the distributed
        reference summaries, tables and figures are also copied so that a single
        analysis can run without its upstream steps; these copies are recorded in
        ``.seeded_reference.json`` so that checks can refuse to treat them as
        regenerated outputs.
        """
        reference = self.data / 'reference'
        if not reference.is_dir():
            raise FileNotFoundError(f'Missing study reference data: {reference}')
        self.output.mkdir(parents=True, exist_ok=True)
        record = self._seed_record()
        for source in reference.rglob('*'):
            baseline = source.relative_to(reference).parts[0] == 'baselines'
            if source.is_file() and (seed_reference or baseline):
                relative = source.relative_to(reference)
                target = self.output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
                    if not baseline:
                        record[relative.as_posix()] = _digest(target)
        (self.output / SEED_RECORD).write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
        for name in ['prediction', 'profiles', 'figures', 'summaries', 'tables', 'logs']:
            (self.output / name).mkdir(exist_ok=True)
        return self

    def _seed_record(self):
        path = self.output / SEED_RECORD
        return json.loads(path.read_text()) if path.is_file() else {}

    def seeded_copies(self):
        """Seeded reference copies that no analysis has since overwritten."""
        return sorted(name for name, digest in self._seed_record().items()
                      if (self.output / name).is_file() and _digest(self.output / name) == digest)

    def discard_seeded_copies(self):
        """Remove seeded reference copies that no analysis has overwritten.

        They are byte-identical to files in the data folder, so nothing is lost;
        removing them stops a later check from reading them as regenerated outputs.
        """
        stale = self.seeded_copies()
        for name in stale:
            (self.output / name).unlink()
        record = {k: v for k, v in self._seed_record().items() if k not in stale and (self.output / k).is_file()}
        (self.output / SEED_RECORD).write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
        return stale

    def require_regenerated(self):
        """Refuse an output folder that still contains unregenerated reference copies."""
        stale = self.seeded_copies()
        if stale:
            raise ValueError(f'{len(stale)} output file(s) are unmodified copies of the distributed reference '
                             f'(e.g. {stale[0]}). Run `etissot reproduce` into a fresh output directory before checking.')
        return self

    def fit_path(self, region):
        """Prefer a new fit in outputs, otherwise use the distributed reference."""
        if region not in {'CONUS', 'Europe', 'Australia', 'TropSAm'}:
            raise ValueError(f'Unknown region: {region}')
        candidate = self.output / 'prediction' / f'fits_{region}.json'
        return candidate if candidate.exists() else self.data / 'prediction' / candidate.name

    def fit_files(self):
        return [self.fit_path(region) for region in ['CONUS', 'Europe', 'Australia', 'TropSAm']]

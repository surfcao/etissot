"""Regenerate the study and compare every weighted quantile with the old rule."""
import argparse
import json
from pathlib import Path
import numpy as np
from etissot.statistics import spatial
from etissot.validation.environment import inspect_environment
from etissot.workspace import Workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manuscript', type=Path, default=Path('manuscript'))
    args = parser.parse_args()
    environment = inspect_environment()
    if not environment['supported_python'] or environment['matches_pins'] is not True:
        raise RuntimeError('Use supported Python with requirements-reproduce.txt installed.')
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(args.output.resolve() / '.matplotlib'))
    stats = dict(calls=0, calls_with_zero_weights=0, changed_calls=0, max_absolute_change=0.)
    corrected = spatial.wquantile

    def audited(x, q, w=None):
        actual = corrected(x, q, w)
        values = np.asarray(x).ravel()
        weights = spatial.W.ravel() if w is None else np.asarray(w).ravel()
        valid = np.isfinite(values)
        values, weights = values[valid], weights[valid]
        order = np.argsort(values)
        values, weights = values[order], weights[order]
        positions = (np.cumsum(weights) - .5 * weights) / weights.sum()
        old = np.interp(q, positions, values)
        delta = float(np.max(np.abs(np.asarray(actual) - old)))
        stats['calls'] += 1
        stats['calls_with_zero_weights'] += int(np.any(weights == 0))
        stats['changed_calls'] += int(delta != 0)
        stats['max_absolute_change'] = max(stats['max_absolute_change'], delta)
        return actual

    # Patch before loading analysis modules so imported aliases are audited too.
    spatial.wquantile = audited
    from etissot.pipeline import run
    from etissot.validation.manuscript import verify
    workspace = Workspace(args.data, args.output)
    try:
        run(workspace)
        report = verify(workspace, args.manuscript, exact_figures=False)
    finally:
        spatial.wquantile = corrected
        if workspace.output.exists():
            (workspace.output / 'quantile_impact.json').write_text(json.dumps(stats, indent=2) + '\n')
            (workspace.output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    print(json.dumps(dict(quantiles=stats, manuscript_mode=report['validation_mode'],
                         figure_pixels_verified=report['figure_pixels_verified']), indent=2))


if __name__ == '__main__':
    main()

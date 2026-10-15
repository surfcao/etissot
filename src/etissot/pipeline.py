"""Named study workflows; heavy modules are loaded only when requested."""
from contextlib import redirect_stdout, redirect_stderr
from importlib import import_module
import json, time
ANALYSES = {'geometry-cases': 'reporting.geometry_cases', 'satclip-correction': 'diagnostics.satclip_correction', 'geographic': 'diagnostics.geographic', 'shape': 'diagnostics.shape', 'downstream': 'diagnostics.downstream', 'summary': 'reporting.summary', 'audit': 'reporting.audit', 'maps': 'reporting.maps', 'estimator': 'diagnostics.estimator', 'historical': 'diagnostics.historical', 'neighbours': 'diagnostics.neighbours', 'harmonics': 'diagnostics.harmonics', 'dense-cached': 'diagnostics.dense_cached', 'replay': 'diagnostics.replay', 'supplement': 'reporting.supplement', 'mixtures': 'diagnostics.mixtures', 'predictions': 'diagnostics.prediction_replay'}
ANALYSES.update({'checkpoint-audits':'reporting.checkpoint_audits', 'neighborhood-controls':'diagnostics.neighborhood_controls', 'manuscript-figures':'reporting.manuscript_figures'})
CHECKS = ['validation.geometry', 'validation.dense', 'validation.downstream']

def run(workspace, steps=None, check_only=False):
    """Reproduce cached analyses and validate outputs; never modify input data.

    ``steps`` is an optional sequence of keys in ANALYSES. With no selection,
    run all generators followed by all numerical checks. Model evaluation and
    GP refitting are separate explicit operations.
    """
    if not __debug__:
        raise RuntimeError('Scientific workflows require Python without optimization (-O); numerical assertions must remain enabled.')
    workspace.prepare(seed_reference=check_only or steps is not None)
    selected = CHECKS if check_only else [ANALYSES[name] for name in (steps if steps is not None else ['checkpoint-audits', 'geometry-cases', 'neighborhood-controls', 'satclip-correction', 'replay', 'geographic', 'shape', 'dense-cached', 'estimator', 'historical', 'neighbours', 'harmonics', 'mixtures', 'predictions', 'summary', 'audit', 'maps', 'downstream', 'supplement', 'manuscript-figures'])]
    if not check_only and steps is None:
        selected += CHECKS
    report = []
    for module in selected:
        print(f'Running {module}', flush=True)
        start = time.monotonic()
        entry = {'module': module, 'status': 'running'}
        report.append(entry)
        try:
            with (workspace.output / 'logs' / f'{module}.log').open('w') as log:
                with redirect_stdout(log), redirect_stderr(log):
                    import_module(f'etissot.{module}').run(workspace)
            entry['status'] = 'passed'
        except Exception as error:
            entry.update(status='failed', error=str(error))
            raise
        finally:
            entry['seconds'] = round(time.monotonic() - start, 2)
            (workspace.output / 'run_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

"""Named study workflows; heavy modules are loaded only when requested."""
from contextlib import redirect_stdout, redirect_stderr
from importlib import import_module
import json, time
ANALYSES = {'geometry-cases': 'reporting.geometry_cases', 'satclip-correction': 'diagnostics.satclip_correction', 'geographic': 'diagnostics.geographic', 'shape': 'diagnostics.shape', 'downstream': 'diagnostics.downstream', 'summary': 'reporting.summary', 'audit': 'reporting.audit', 'maps': 'reporting.maps', 'estimator': 'diagnostics.estimator', 'historical': 'diagnostics.historical', 'neighbours': 'diagnostics.neighbours', 'harmonics': 'diagnostics.harmonics', 'dense-cached': 'diagnostics.dense_cached', 'replay': 'diagnostics.replay', 'supplement': 'reporting.supplement', 'mixtures': 'diagnostics.mixtures', 'predictions': 'diagnostics.prediction_replay'}
ANALYSES.update({'checkpoint-audits':'reporting.checkpoint_audits', 'neighborhood-controls':'diagnostics.neighborhood_controls', 'manuscript-figures':'reporting.manuscript_figures', 'radius-screening':'reporting.radius_screening', 'asymmetry':'reporting.asymmetry', 'reference-accuracy':'reporting.reference_accuracy', 'grid-smoothness':'reporting.grid_smoothness', 'latitude-profiles':'reporting.latitude_profiles'})
CHECKS = ['validation.geometry', 'validation.dense', 'validation.downstream']

def run(workspace, steps=None, check_only=False):
    """Reproduce cached analyses and validate outputs; never modify input data.

    ``steps`` is an optional sequence of keys in ANALYSES. With no selection,
    run all generators followed by all numerical checks. Model evaluation and
    GP refitting are separate explicit operations.
    """
    if not __debug__:
        raise RuntimeError('Scientific workflows require Python without optimization (-O); numerical assertions must remain enabled.')
    # Only standalone analyses may borrow distributed reference prerequisites;
    # checks and full reproduction must read outputs regenerated in this folder.
    workspace.prepare(seed_reference=steps is not None and not check_only)
    selected = CHECKS if check_only else [ANALYSES[name] for name in (steps if steps is not None else ['checkpoint-audits', 'geometry-cases', 'neighborhood-controls', 'satclip-correction', 'replay', 'geographic', 'latitude-profiles', 'shape', 'dense-cached', 'estimator', 'historical', 'neighbours', 'harmonics', 'mixtures', 'predictions', 'summary', 'audit', 'maps', 'downstream', 'supplement', 'radius-screening', 'asymmetry', 'reference-accuracy', 'grid-smoothness', 'manuscript-figures'])]
    full = not check_only and steps is None
    if full:
        selected += CHECKS
    report = []
    # A full reproduction keeps its own record, which `check` and manuscript
    # verification require; single analyses and checks never overwrite it.
    record = workspace.output / 'reproduce_report.json'
    started = time.time()

    def save_record(status):
        if full:
            record.write_text(json.dumps(dict(started=started, status=status, steps=report), indent=2) + '\n')
    save_record('running')
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
        except BaseException as error:
            entry.update(status='failed', error=str(error) or type(error).__name__)
            save_record('failed')
            raise
        finally:
            entry['seconds'] = round(time.monotonic() - start, 2)
            (workspace.output / 'run_report.json').write_text(json.dumps(report, indent=2) + '\n')
    save_record('passed')
    return report

"""Command-line interface for cached analyses and optional fresh evaluation."""
import argparse
import os
from pathlib import Path
from . import __version__
from .workspace import Workspace
from .pipeline import ANALYSES
from .validation.environment import inspect_environment

def main(argv=None):
    parser = argparse.ArgumentParser(prog='etissot', description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    env = sub.add_parser('environment', help='Check Python and pinned reproduction dependencies')
    env.add_argument('--requirements', type=Path, default=Path('requirements-reproduce.txt'))
    models = sub.add_parser('models', help='List, obtain, verify or bundle exact study checkpoints')
    model_commands = models.add_subparsers(dest='model_command', required=True)
    for name in ['list', 'download', 'verify', 'bundle']:
        item = model_commands.add_parser(name)
        item.add_argument('--manifest', type=Path, help='Defaults to the manifest installed with eTissot')
        item.add_argument('--family', nargs='+', dest='families', help='Filter by family, e.g. satclip sinr')
        item.add_argument('--encoder', nargs='+', dest='encoders', help='Filter by checkpoint id')
        if name == 'list':
            item.add_argument('--json', action='store_true', help='Print the complete selected checkpoint records')
        else:
            item.add_argument('--models', type=Path, required=True, help='Directory holding external model files')
        if name == 'download':
            item.add_argument('--dry-run', action='store_true', help='Report planned downloads without writing files')
            item.add_argument('--timeout', type=float, default=60, help='Network operation timeout in seconds')
        if name in ['download', 'bundle']:
            item.add_argument('--replace', action='store_true', help='Explicitly replace an existing differing file or bundle')
        if name == 'bundle':
            item.add_argument('--output', type=Path, required=True, help='Separate checkpoint ZIP to create')
    for name in ['reproduce', 'check', 'analyze', 'fit', 'evaluate-models']:
        item = sub.add_parser(name)
        item.add_argument('--data', type=Path, required=True, help='Extracted study data directory')
        item.add_argument('--output', type=Path, default=Path('outputs'))
        if name == 'reproduce':
            item.add_argument('--manuscript', type=Path, help='Check all generated figures/tables and numerical claim coverage against these TeX sources')
            item.add_argument('--numerical-only', action='store_true', help='Check manuscript numbers and figure presence, reporting pixel differences without failing')
        if name == 'analyze':
            item.add_argument('analysis', choices=sorted(ANALYSES))
        if name == 'fit':
            item.add_argument('--windows', nargs='+', choices=['CONUS', 'Europe', 'Australia', 'TropSAm'])
            item.add_argument('--threads', type=int, default=2)
            item.add_argument('--limit', type=int, help='Stop after this many newly completed fits')
        if name == 'evaluate-models':
            item.add_argument('analysis', choices=['boundaries', 'dense', 'neighbours', 'satclip_correction'])
            item.add_argument('--models', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'models':
        import json
        from .models.checkpoints import load_manifest, select_checkpoints, download_models, verify_models, bundle_models
        selection = dict(manifest=args.manifest, families=args.families, encoders=args.encoders)
        try:
            if args.model_command == 'list':
                entries = select_checkpoints(load_manifest(args.manifest), args.families, args.encoders)
                if args.json:
                    print(json.dumps(entries, indent=2))
                else:
                    print('Checkpoint                         Family      Size MB   Weight license       Bundle')
                    for entry in entries:
                        terms = entry['license']
                        print(f"{entry['id']:<34} {entry['family']:<11} {entry['bytes']/1e6:>7.1f}   "
                              f"{terms.get('id') or 'not established':<20} {'yes' if terms['redistribution'] else 'no'}")
                return
            if args.model_command == 'download':
                report = download_models(args.models, **selection, replace=args.replace,
                                         dry_run=args.dry_run, timeout=args.timeout)
            elif args.model_command == 'verify':
                report = verify_models(args.models, **selection)
            else:
                report = bundle_models(args.models, args.output, **selection, replace=args.replace)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2))
        if report['status'] != 'passed':
            raise SystemExit(1)
        return
    if args.command == 'environment':
        import json
        report = inspect_environment(args.requirements)
        print(json.dumps(report, indent=2))
        if not report['matches_pins'] or not report['supported_python']:
            raise SystemExit(1)
        return
    from importlib.util import find_spec
    missing = [name for name in ['scipy', 'torch', 'matplotlib', 'pyproj', 'PIL'] if find_spec(name) is None]
    if missing:
        parser.error('Study dependencies are missing (' + ', '.join(missing) +
                     '). Install etissot[study], or requirements-reproduce.txt for the pinned study environment.')
    workspace = Workspace(args.data, args.output).prepare(seed_reference=args.command != 'reproduce')
    if args.command == 'reproduce':
        import json
        report = inspect_environment()
        (workspace.output / 'environment.json').write_text(json.dumps(report, indent=2) + '\n')
        if not report['supported_python']:
            parser.error('Python 3.11+ is required; see environment.json.')
        if report['matches_pins'] is not True:
            print('Environment differs from the pinned reproduction setup; see environment.json.', flush=True)
    os.environ.setdefault('MPLCONFIGDIR', str(workspace.output / '.matplotlib'))
    if args.command == 'fit':
        from .prediction.evaluation import run
        run(workspace, windows=args.windows, threads=args.threads, limit=args.limit)
    elif args.command == 'evaluate-models':
        from importlib import import_module
        module = import_module(f'etissot.diagnostics.{args.analysis}')
        (module.collect if args.analysis in ['neighbours', 'satclip_correction'] else module.run)(workspace, args.models)
    else:
        from .pipeline import run
        run(workspace, check_only=args.command == 'check', steps=[args.analysis] if args.command == 'analyze' else None)
    if args.command == 'reproduce' and args.manuscript is not None:
        from .validation.manuscript import verify
        verify(workspace, args.manuscript, exact_figures=not args.numerical_only)
    print(f'Completed. Outputs: {workspace.output}')

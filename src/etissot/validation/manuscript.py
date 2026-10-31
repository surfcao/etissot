"""Fail on missing/stale manuscript figures, generated tables, or tracked numerical claims."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image

from . import prose

BLOCK=re.compile(r'(?m)^% BEGIN GENERATED: ([A-Za-z0-9_]+)\n(.*?)^% END GENERATED: \1$',re.S)
MARKER=re.compile(r'(?im)^[ \t]*%[ \t]*BEGIN[ \t]+GENERATED\b')
END_MARKER=re.compile(r'(?im)^[ \t]*%[ \t]*END[ \t]+GENERATED\b')
BOUNDARY=r'\relax\space% Preserve the original input-file boundary.'


def load_documents(manuscript, contract):
    names = contract.get('documents', ['main.tex', 'supplement.tex'])
    if 'main.tex' not in names:
        raise ValueError('The validation contract must include the main manuscript')
    allowed = {'main.tex', 'supplement.tex', 'additional_results.tex'}
    if len(set(names)) != len(names) or not set(names).issubset(allowed):
        raise ValueError('Invalid validation document list')
    return {name: (Path(manuscript) / name).read_text() for name in names}


TABLE=re.compile(r'\\begin\{(table\*?|sidewaystable\*?|longtable)\}.*?\\end\{\1\}', re.S)
TABULAR=re.compile(r'\\begin\{(tabular\*?|tabularx|tabulary)\}')


def compare_generated_blocks(documents, output, required=None, fresh=None):
    names=[]
    for filename,text in documents.items():
        # Every BEGIN marker must open a well-formed block; none may be skipped.
        found = len(BLOCK.findall(text))
        if len(MARKER.findall(text)) != found or len(END_MARKER.findall(text)) != found:
            raise ValueError(f'Malformed or unmatched generated-block marker in {filename}')
        present=[]
        for name,content in BLOCK.findall(text):
            if name in names:raise ValueError(f'Duplicate generated block: {name}')
            actual=content.replace(BOUNDARY,'').strip()
            path=Path(output)/'tables'/(name+'.tex')
            if not path.is_file() or actual != path.read_text().strip():
                raise ValueError(f'Missing or stale generated block: {filename}: {name}')
            if fresh is not None:
                fresh(f'tables/{name}.tex')
            names.append(name); present.append(name)
        if required is not None and sorted(present) != sorted(required.get(filename, [])):
            missing = sorted(set(required.get(filename, [])) - set(present))
            extra = sorted(set(present) - set(required.get(filename, [])))
            raise ValueError(f'Generated blocks in {filename} differ from the contract: missing {missing}, unexpected {extra}')
    if not names:raise ValueError('No generated blocks found')
    return names


def _normalized(table):
    return ' '.join(table.split())


def check_tables(documents, workspace, contract, fresh=None):
    """Every live table is generated, or hand-written and checked."""
    analytic = contract.get('analytic_tables', {})
    table_count=0; ungenerated=[]
    for name,text in documents.items():
        remaining=prose.live_text(text)  # generated blocks and comments removed
        tables=[m.group(0) for m in TABLE.finditer(remaining)]
        # Count generated and hand-written tables; only the latter are checked here.
        table_count+=len(TABLE.findall(prose.COMMENT.sub(prose._blank, text)))
        outside=TABLE.sub('', remaining)
        if TABULAR.search(outside):
            raise ValueError(f'Tabular material outside a table environment in {name}')
        for table in tables:
            # Identify hand-written tables by label, which survives caption edits.
            labels=re.findall(r'\\label\{([^}]*)\}', table)
            if len(labels)==1 and labels[0] in analytic:
                digest=hashlib.sha256(_normalized(table).encode()).hexdigest()
                if digest!=analytic[labels[0]]:
                    raise ValueError(f'Hand-written table {labels[0]} changed; check it and update analytic_tables in the contract')
                ungenerated.append(dict(file=name,label=labels[0],kind='analytic table; content pinned by hash, numbers in the prose register'))
            elif labels==['tab:cspprecision']:
                expected=(workspace.output/'tables/csp_precision_table.tex').read_text().strip()
                if fresh is not None:
                    fresh('tables/csp_precision_table.tex')
                if table.strip()!=expected:
                    raise ValueError('CSP precision table differs from regenerated site-level statistics')
                ungenerated.append(dict(file=name,label=labels[0],kind='CSP table regenerated from declared site-level measurements'))
            else:raise ValueError(f'Uncovered empirical table in {name}: {labels}')
    return table_count, ungenerated


def freshness(workspace, since):
    """Return a check that an output file was written by the current run."""
    def fresh(relative):
        path=workspace.output/relative
        # Two seconds allow for coarse file-system timestamps.
        if since is not None and path.stat().st_mtime < since - 2:
            raise ValueError(f'Output not written by the current reproduction: {relative}')
    return fresh


def reproduction_start(workspace):
    report=workspace.output/'reproduce_report.json'
    if not report.is_file():
        raise ValueError(f'No full reproduction recorded in {workspace.output}; run `etissot reproduce` first.')
    record=json.loads(report.read_text())
    if record.get('status')!='passed':
        raise ValueError(f'The last full reproduction in {workspace.output} did not complete.')
    return float(record['started'])


def verify(workspace, manuscript, exact_figures=True, since='reproduction'):
    """Check the manuscript against regenerated outputs; the report records failure too."""
    destination=workspace.output/'manuscript_verification.json'
    destination.unlink(missing_ok=True)  # never leave an earlier 'passed' report behind
    try:
        report=_verify(workspace, manuscript, exact_figures, reproduction_start(workspace) if since=='reproduction' else since)
    except Exception as error:
        destination.write_text(json.dumps(dict(status='failed', error=str(error)), indent=2)+'\n')
        raise
    destination.write_text(json.dumps(report,indent=2)+'\n')
    return report


def _verify(workspace, manuscript, exact_figures, since):
    manuscript=Path(manuscript).resolve()
    fresh=freshness(workspace, since)
    # The contract and baselines are read from the data folder, never from outputs.
    contract=json.loads((workspace.data/'reference/manuscript_claims.json').read_text())
    documents=load_documents(manuscript, contract)
    blocks=compare_generated_blocks(documents,workspace.output,contract.get('generated_blocks'),fresh)
    if (manuscript/'numbers.tex').read_bytes() != (workspace.output/'tables/numbers.tex').read_bytes():raise ValueError('Stale numerical macros')
    fresh('tables/numbers.tex')
    figures=sorted(set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', '\n'.join(prose.live_text(d) for d in documents.values()))))
    if len(figures)!=16:raise ValueError(f'Expected all 16 journal and package figures, found {figures}')
    figure_results=[]
    for name in figures:
        generated=workspace.output/'figures'/name;published=manuscript/'figures'/name
        if not generated.is_file():raise ValueError(f'Missing generated figure: {name}')
        fresh(f'figures/{name}')
        with Image.open(generated) as a, Image.open(published) as b:
            x,y=np.asarray(a.convert('RGBA')),np.asarray(b.convert('RGBA'))
        identical = x.shape == y.shape and np.array_equal(x,y)
        if exact_figures and not identical:raise ValueError(f'Figure pixels differ: {name}')
        figure_results.append(dict(file=name,pixels_identical=identical))
    table_count, ungenerated = check_tables(documents, workspace, contract, fresh)
    seeded=set(workspace.seeded_copies())
    cache={}
    def summary(name):
        if f'summaries/{name}' in seeded:raise ValueError(f'Summary was not regenerated: {name}')
        if name not in cache:
            fresh(f'summaries/{name}')
            cache[name]=json.loads((workspace.output/'summaries'/name).read_text())
        return cache[name]
    # Every number in live prose: results recomputed, other kinds pinned.
    if 'prose_numbers' not in contract:raise ValueError('The validation contract has no prose-number register')
    prose_report=prose.check(documents, contract['prose_numbers'], summary)
    claims=[]
    for claim in contract.get('numerical_claims', []):
        # Summary checks not tied to a prose number (values compared with the contract only).
        value=summary(claim['summary'])
        for key in claim['path']:value=value[int(key)] if isinstance(value,list) else value[key]
        actual=float(value)*claim.get('scale',1)
        expected, tolerance = float(claim['expected']), float(claim['atol'])
        if not np.isfinite([actual, expected, tolerance]).all() or tolerance < 0 or abs(actual-expected)>tolerance:
            raise ValueError(f'Numerical claim differs or is nonfinite: {claim["id"]}: {actual}')
        claims.append(dict(id=claim['id'],value=actual,expected=claim['expected'],absolute_tolerance=claim['atol']))
    for item in contract['coverage']:
        for name in item['summaries']:
            if not (workspace.output/'summaries'/name).is_file():raise ValueError(f'Missing coverage evidence: {name}')
            if f'summaries/{name}' in seeded:raise ValueError(f'Coverage evidence was not regenerated: {name}')
            fresh(f'summaries/{name}')
    report=dict(status='passed',validation_mode='exact_artifacts' if exact_figures else 'numerical_only',figure_pixels_verified=all(r['pixels_identical'] for r in figure_results),scope=contract['scope'],generated_blocks=blocks,numerical_macros=True,figures=figure_results,table_count=table_count,analytic_tables=ungenerated,prose_numbers=prose_report,summary_checks=claims,coverage=contract['coverage'],tex_sha256={n:hashlib.sha256((manuscript/n).read_bytes()).hexdigest() for n in ['main.tex','numbers.tex']})
    report['tex_sha256'].update({n:hashlib.sha256((manuscript/n).read_bytes()).hexdigest() for n in documents if n not in report['tex_sha256']})
    report['document_inventory'] = {name: dict(figures=len(re.findall(r'\\includegraphics', prose.live_text(text))), tables=len(TABLE.findall(prose.live_text(text))), generated_blocks=len(BLOCK.findall(text))) for name, text in documents.items()}
    print(f'Verified {len(figures)} figures, {table_count} tables, {len(blocks)} generated blocks, '
          f'{prose_report["results"]} prose results recomputed, {prose_report["pinned"]} other prose numbers pinned '
          f'and {len(claims)} summary checks.')
    return report

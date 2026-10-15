"""Fail on missing/stale manuscript figures, generated tables, or tracked numerical claims."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image

BLOCK=re.compile(r'(?m)^% BEGIN GENERATED: ([a-z_]+)\n(.*?)^% END GENERATED: \1$',re.S)
BOUNDARY=r'\relax\space% Preserve the original input-file boundary.'


def load_documents(manuscript, contract):
    names = contract.get('documents', ['main.tex', 'supplement.tex'])
    if 'main.tex' not in names:
        raise ValueError('The validation contract must include the main manuscript')
    allowed = {'main.tex', 'supplement.tex', 'additional_results.tex'}
    if len(set(names)) != len(names) or not set(names).issubset(allowed):
        raise ValueError('Invalid validation document list')
    return {name: (Path(manuscript) / name).read_text() for name in names}


def compare_generated_blocks(documents, output):
    names=[]
    for filename,text in documents.items():
        for name,content in BLOCK.findall(text):
            if name in names:raise ValueError(f'Duplicate generated block: {name}')
            actual=content.replace(BOUNDARY,'').strip()
            path=Path(output)/'tables'/(name+'.tex')
            if not path.is_file() or actual != path.read_text().strip():
                raise ValueError(f'Missing or stale generated block: {filename}: {name}')
            names.append(name)
    if not names:raise ValueError('No generated blocks found')
    return names


def verify(workspace, manuscript, exact_figures=True):
    manuscript=Path(manuscript).resolve()
    contract=json.loads((workspace.data/'reference/manuscript_claims.json').read_text())
    documents=load_documents(manuscript, contract)
    blocks=compare_generated_blocks(documents,workspace.output)
    if (manuscript/'numbers.tex').read_bytes() != (workspace.output/'tables/numbers.tex').read_bytes():raise ValueError('Stale numerical macros')
    figures=sorted(set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', '\n'.join(documents.values()))))
    if len(figures)!=15:raise ValueError(f'Expected all 15 journal and package figures, found {figures}')
    figure_results=[]
    for name in figures:
        generated=workspace.output/'figures'/name;published=manuscript/'figures'/name
        if not generated.is_file():raise ValueError(f'Missing generated figure: {name}')
        with Image.open(generated) as a, Image.open(published) as b:
            x,y=np.asarray(a.convert('RGBA')),np.asarray(b.convert('RGBA'))
        identical = x.shape == y.shape and np.array_equal(x,y)
        if exact_figures and not identical:raise ValueError(f'Figure pixels differ: {name}')
        figure_results.append(dict(file=name,pixels_identical=identical))
    ungenerated=[];table_count=0
    for name,text in documents.items():
        table_count+=len(re.findall(r'\\begin\{table\}',re.sub(r'(?m)^%.*\n?','',text)))
        remaining=re.sub(r'(?m)^%.*\n?','',BLOCK.sub('',text))  # commented-out tables are not live
        for table in re.findall(r'\\begin\{table\}.*?\\end\{table\}',remaining,re.S):
            if 'Infinitesimal reference anisotropies' in table:
                ungenerated.append(dict(file=name,kind='analytic formula table; checked by analytic geometry and harmonic diagnostics'))
            elif 'CSP anisotropy error against double-precision Jacobians' in table:
                expected=(workspace.output/'tables/csp_precision_table.tex').read_text().strip()
                if table.strip()!=expected:
                    raise ValueError('CSP precision table differs from regenerated site-level statistics')
                ungenerated.append(dict(file=name,kind='CSP table independently regenerated from declared site-level measurements'))
            else:raise ValueError(f'Uncovered empirical table in {name}')
    claims=[]
    for claim in contract['numerical_claims']:
        text=documents[claim['document']]
        if claim.get('literal') and claim['literal'] not in text:raise ValueError(f'Claim text changed without updating its check: {claim["id"]}')
        value=json.loads((workspace.output/'summaries'/claim['summary']).read_text())
        for key in claim['path']:value=value[int(key)] if isinstance(value,list) else value[key]
        actual=float(value)*claim.get('scale',1)
        expected, tolerance = float(claim['expected']), float(claim['atol'])
        if not np.isfinite([actual, expected, tolerance]).all() or tolerance < 0 or abs(actual-expected)>tolerance:
            raise ValueError(f'Numerical claim differs or is nonfinite: {claim["id"]}: {actual}')
        claims.append(dict(id=claim['id'],value=actual,expected=claim['expected'],absolute_tolerance=claim['atol']))
    for item in contract['coverage']:
        for summary in item['summaries']:
            if not (workspace.output/'summaries'/summary).is_file():raise ValueError(f'Missing coverage evidence: {summary}')
    report=dict(status='passed',validation_mode='exact_artifacts' if exact_figures else 'numerical_only',figure_pixels_verified=all(r['pixels_identical'] for r in figure_results),scope=contract['scope'],generated_blocks=blocks,numerical_macros=True,figures=figure_results,table_count=table_count,analytic_tables=ungenerated,numerical_claims=claims,coverage=contract['coverage'],tex_sha256={n:hashlib.sha256((manuscript/n).read_bytes()).hexdigest() for n in ['main.tex','numbers.tex']})
    report['tex_sha256'].update({n:hashlib.sha256((manuscript/n).read_bytes()).hexdigest() for n in documents if n not in report['tex_sha256']})
    report['document_inventory'] = {name: dict(figures=len(re.findall(r'\\includegraphics', text)), tables=len(re.findall(r'\\begin\{table\}', re.sub(r'(?m)^%.*\n?', '', text))), generated_blocks=len(BLOCK.findall(text))) for name, text in documents.items()}
    (workspace.output/'manuscript_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Verified {len(figures)} figures, {table_count} tables, {len(blocks)} generated blocks and {len(claims)} numerical prose checks.')
    return report

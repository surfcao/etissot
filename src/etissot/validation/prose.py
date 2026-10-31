"""Every number in live manuscript prose, and how each one is checked.

Live prose is the document body with comments and generated blocks removed.
Each numeric token in it must match, in order, an entry of the register stored
in the validation contract (``prose_numbers``). An entry records the token, a
short anchor of surrounding text and a kind:

- ``result``: an outcome of the analyses; its ``link`` recomputes the value from
  regenerated summaries, and the written number must equal it at the written
  precision and sign;
- ``design``, ``formula``, ``citation``, ``label``: settings, mathematics,
  references and numbering, which are pinned by the register but not computed;
- ``reading``: an approximate value read from a figure ("flat to about 10 km"),
  pinned with a note naming its source; the report counts these separately.

Any edit that adds, removes or moves a number therefore fails the check until
the register is updated (``scripts/update_prose_register.py``), and any result
whose written value disagrees with the regenerated summaries fails outright.
"""
import json
import re
from pathlib import Path

BLOCK = re.compile(r'(?ms)^[ \t]*%[ \t]*BEGIN[ \t]+GENERATED:[ \t]*([A-Za-z0-9_]+)[ \t]*\n.*?^[ \t]*%[ \t]*END[ \t]+GENERATED:[ \t]*\1[ \t]*$')
COMMENT = re.compile(r'(?<!\\)%[^\n]*')
# Arguments that hold names, keys, paths or layout rather than reported numbers.
SKIP = re.compile(r'\\(?:ref|eqref|autoref|cref|Cref|cite|citep|citet|citealp|label|includegraphics|input|include|url|href|'
                  r'hspace|vspace|setlength|documentclass|usepackage|bibliography|bibliographystyle|newcommand|'
                  r'renewcommand|begin|end)\*?(?:\[[^\]]*\])?\{[^}]*\}')
# Words are consumed whole, so digits inside names (L10, R18, WGS84, cite keys) are not
# numbers, while a control word stops at its letters (\\approx11.5 reads 11.5).
WORD = re.compile(r'\\[A-Za-z]+|[A-Za-z][A-Za-z0-9]*|((?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:e-?\d+)?)')
KINDS = ('result', 'design', 'formula', 'citation', 'label', 'reading')
ANCHOR = 24
OPS = ('value', 'diff', 'ratio', 'sum', 'abs', 'len', 'min', 'max', 'mean', 'max_abs_diff')


def _blank(match):
    return re.sub(r'[^\n]', ' ', match.group(0))


def live_text(text):
    """Text with generated blocks and comments blanked, offsets preserved."""
    return COMMENT.sub(_blank, BLOCK.sub(_blank, text))


def tokens(text):
    """Numeric tokens of the document body in order: line, token, left and right context."""
    live = live_text(text)
    scan = SKIP.sub(_blank, live)
    begin = text.find('\\begin{document}')
    found = []
    for match in WORD.finditer(scan):
        if match.group(1) is None or match.start() < begin:
            continue
        left = ' '.join(scan[max(0, match.start() - 4 * ANCHOR):match.start()].split())
        right = ' '.join(scan[match.end():match.end() + 4 * ANCHOR].split())
        found.append(dict(line=scan.count('\n', 0, match.start()) + 1, token=match.group(0),
                          left=left, right=right, anchor=left[-ANCHOR:] + '|' + right[:ANCHOR]))
    return found


def text_value(token, left):
    """Value as written, with sign from a preceding minus (not a range dash)."""
    clean = token.replace(',', '')
    decimals = len(clean.split('.')[1]) if '.' in clean and 'e' not in clean else 0
    tail = left.rstrip('{} $')
    negative = bool(re.search(r'(?:(?<!-)-|−|\\textminus|\\minus)$', tail))
    return (-1 if negative else 1) * float(clean), decimals


def evaluate(link, summary):
    """Recompute a linked result; ``summary(name)`` returns a parsed summary."""
    if link.get('op', 'value') not in OPS or not link.get('refs'):
        raise ValueError(f'Invalid result link: {link}')
    values = []
    for ref in link['refs']:
        value = summary(ref['summary'])
        for key in ref['path']:
            value = value[int(key)] if isinstance(value, list) else value[key]
        values.append(value)
    op = link.get('op', 'value')
    if op == 'len':
        result = float(len(values[0]))
    else:
        v = [float(x) for x in values]
        if op == 'max_abs_diff' and (len(v) < 2 or len(v) % 2):
            raise ValueError(f'max_abs_diff needs pairs of references: {link}')
        result = {'max_abs_diff': lambda: max(abs(a - b) for a, b in zip(v[0::2], v[1::2])),
                  'value': lambda: v[0], 'diff': lambda: v[0] - v[1], 'ratio': lambda: v[0] / v[1],
                  'sum': lambda: sum(v), 'abs': lambda: abs(v[0]), 'min': lambda: min(v),
                  'max': lambda: max(v), 'mean': lambda: sum(v) / len(v)}[op]()
    return result * float(link.get('scale', 1))


def agrees(shown, decimals, computed, compare='rounded', atol=None):
    """The written number states the computed value as the prose says.

    ``rounded``: equal at the written precision; ``upper``/``lower``: a bound
    that holds ("fewer than 1%", "above 11");
    ``approx``: within an explicit absolute tolerance ("approximately 1,720 km");
    ``below_power_of_ten``: the token is the exponent of a bound ("less than $10^{-6}$").
    """
    unit = 10 ** -decimals
    slack = 1e-9 * unit + 1e-12 * abs(shown)
    if compare == 'rounded':
        return abs(computed - shown) <= 0.5 * unit + slack
    if compare == 'upper':
        return computed <= shown + slack
    if compare == 'lower':
        return computed >= shown - slack
    if compare == 'below_power_of_ten':  # "less than $10^{-6}$": the token is the exponent
        return computed < 10.0 ** -shown
    if compare == 'approx' and atol is not None and atol >= 0:
        return abs(computed - shown) <= float(atol) + slack
    raise ValueError(f'Invalid comparison {compare!r} (approx needs atol)')


def check(documents, register, summary, fresh=None):
    """Check every live-prose number against the register; return a report."""
    report = dict(results=0, pinned=0, by_kind={})
    for name, text in documents.items():
        found = tokens(text)
        entries = register.get(name)
        if entries is None:
            raise ValueError(f'No prose-number register for {name}')
        for index, item in enumerate(found):
            if index >= len(entries) or (entries[index]['token'], entries[index]['anchor']) != (item['token'], item['anchor']):
                expected = entries[index] if index < len(entries) else None
                raise ValueError(
                    f'Unregistered or changed number in {name} line {item["line"]}: "...{item["left"][-40:]} [{item["token"]}] '
                    f'{item["right"][:30]}..."' + (f' (register expects {expected["token"]} at "{expected["anchor"]}")' if expected else '')
                    + '; check it and run scripts/update_prose_register.py')
        if len(entries) != len(found):
            missing = entries[len(found)]
            raise ValueError(f'Registered number no longer in {name}: {missing["token"]} at "{missing["anchor"]}"; '
                             'run scripts/update_prose_register.py')
        for item, entry in zip(found, entries):
            kind = entry.get('kind')
            if kind not in KINDS:
                raise ValueError(f'Unclassified number in {name} line {item["line"]}: {item["token"]} at "{entry["anchor"]}"')
            report['by_kind'][kind] = report['by_kind'].get(kind, 0) + 1
            if kind == 'reading' and not entry.get('note'):
                raise ValueError(f'Figure reading without a source note in {name} line {item["line"]}: {item["token"]}')
            if kind != 'result':
                report['pinned'] += 1
                continue
            if not entry.get('link'):
                raise ValueError(f'Result without a link in {name} line {item["line"]}: {item["token"]} at "{entry["anchor"]}"')
            for ref in entry['link']['refs']:
                if fresh is not None:
                    fresh(f"summaries/{ref['summary']}")
            computed = evaluate(entry['link'], summary)
            shown, decimals = text_value(item['token'], item['left'])
            if not agrees(shown, decimals, computed, entry.get('compare', 'rounded'), entry.get('atol')):
                raise ValueError(f'Number in {name} line {item["line"]} differs from regenerated results: '
                                 f'written {item["token"]}, computed {computed:.6g} ("...{item["left"][-40:]} [{item["token"]}]")')
            report['results'] += 1
    return report


def load_register(path):
    return json.loads(Path(path).read_text())['prose_numbers']

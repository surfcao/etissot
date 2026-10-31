"""Bring the prose-number register in line with an edited manuscript.

Every number in the live prose of the validated documents must have a register
entry (see etissot.validation.prose). After editing the manuscript, run

    python scripts/update_prose_register.py --data ./data --manuscript ./manuscript

to see which numbers were added, removed or moved. Entries are carried over
where the surrounding text still identifies them, including numbers whose value
changed (their links are then checked against the regenerated results). New
numbers are marked "unclassified" and fail verification until a kind (and, for
results, a link) is recorded. Add --write to save the updated register into
data/reference/manuscript_claims.json.
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from etissot.validation.prose import tokens  # noqa: E402


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def align(found, entries):
    """Carry register entries over to the current tokens; return entries and a change log."""
    keys_new = [(x['token'], x['anchor']) for x in found]
    keys_old = [(e['token'], e['anchor']) for e in entries]
    result, log = [None] * len(found), []
    matcher = difflib.SequenceMatcher(None, keys_old, keys_new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                result[j1 + k] = dict(entries[i1 + k])
            continue
        pool = list(range(i1, i2))
        for j in range(j1, j2):
            item = found[j]
            best, score = None, 0.0
            for i in pool:
                old = entries[i]
                s = similarity(old['anchor'], item['anchor']) + (0.2 if old['token'] == item['token'] else 0)
                if s > score:
                    best, score = i, s
            if best is not None and score >= 0.75:
                pool.remove(best)
                entry = dict(entries[best], token=item['token'], anchor=item['anchor'])
                if entries[best]['token'] != item['token']:
                    log.append(f'changed  line {item["line"]}: {entries[best]["token"]} -> {item["token"]} ({entry.get("kind")})')
                result[j] = entry
            else:
                result[j] = dict(token=item['token'], anchor=item['anchor'], kind='unclassified')
                log.append(f'new      line {item["line"]}: ...{item["left"][-40:]} [{item["token"]}] {item["right"][:30]}...')
        for i in pool:
            log.append(f'removed  {entries[i]["token"]} at "{entries[i]["anchor"]}" ({entries[i].get("kind")})')
    return result, log


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--manuscript', type=Path, default=Path('manuscript'))
    parser.add_argument('--write', action='store_true', help='Save the updated register')
    args = parser.parse_args()
    path = args.data / 'reference/manuscript_claims.json'
    contract = json.loads(path.read_text())
    register = contract.setdefault('prose_numbers', {})
    unclassified = 0
    for name in contract['documents']:
        found = tokens((args.manuscript / name).read_text())
        updated, log = align(found, register.get(name, []))
        register[name] = updated
        unclassified += sum(e['kind'] == 'unclassified' for e in updated)
        print(f'{name}: {len(found)} numbers; {len(log)} changes')
        for line in log:
            print('  ' + line)
    print(f'{unclassified} unclassified numbers need a kind' + (' and, for results, a link.' if unclassified else '.'))
    if args.write:
        path.write_text(json.dumps(contract, indent=1) + '\n')
        print(f'Wrote {path}')


if __name__ == '__main__':
    main()

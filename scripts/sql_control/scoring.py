"""Text agreement, strict SELECT parsing, and bounded execution are distinct."""

from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

import sqlglot
from sqlglot import exp

from scripts.sql_control.fixtures import SEEDS, population


def normalize(sql):
    # Reference-compatible text metric; this also folds string literal case.
    # Keep its known limitation visible and measure execution separately.
    return re.sub(r'\s+', ' ', sql.strip()).removesuffix(';').strip().lower()


def parsed_select(raw):
    if not raw.strip() or len(raw.encode()) > 16384:
        return None
    try:
        statements = sqlglot.parse(raw, read='sqlite', error_level=sqlglot.ErrorLevel.RAISE)
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            return None
        tree = statements[0]
        forbidden = (exp.DML, exp.DDL, exp.Command, exp.Into)
        if any(isinstance(node, forbidden) for node in tree.walk()):
            return None
        return tree
    except (sqlglot.errors.SqlglotError, RecursionError):
        return None


def same_results(left, right, ordered):
    if left['columns'] != right['columns']:
        return False
    if ordered:
        return left['rows'] == right['rows']
    # Bag comparison preserves duplicates; NULL and scalar types survive JSON.
    return Counter(map(tuple, left['rows'])) == Counter(map(tuple, right['rows']))


def execution(case, prediction):
    if parsed_select(prediction) is None:
        return {'execution_match': False, 'execution_category': 'non_select_or_parse_error'}
    for seed in SEEDS:
        payload = {'schema': case['context'], 'population': population(seed),
                   'gold': case['answer'], 'prediction': prediction}
        try:
            completed = subprocess.run(
                [sys.executable, '-I', '-S', str(Path(__file__).with_name('worker.py'))],
                input=json.dumps(payload), text=True, capture_output=True, timeout=4,
                cwd='/', env={'PYTHONHASHSEED': '0'})
            if completed.returncode or len(completed.stdout) > 131072:
                raise ValueError('worker failure')
            result = json.loads(completed.stdout)
            if 'error' in result:
                return {'execution_match': False, 'execution_category': 'execution_error'}
            if not same_results(*result['results'], case['ordered']):
                return {'execution_match': False, 'execution_category': 'result_mismatch'}
        except (subprocess.TimeoutExpired, ValueError):
            return {'execution_match': False, 'execution_category': 'resource_limit'}
    return {'execution_match': True, 'execution_category': 'match'}


def outcome(record, raw, fixture=False):
    valid = parsed_select(raw) is not None
    exact = normalize(raw) == normalize(record['answer'])
    row = {'id': record['id'], 'raw': raw, 'exact_match': exact, 'parse_valid': valid,
           'category': ('empty' if not raw.strip() else 'parse_or_non_select' if not valid
                        else 'exact' if exact else 'text_mismatch')}
    if fixture:
        row.update(execution(record, raw))
    return row


def aggregate(rows):
    if not rows:
        raise ValueError('Cannot score an empty cohort')
    keys = ['exact_match', 'parse_valid'] + (['execution_match'] if 'execution_match' in rows[0] else [])
    return {'metrics': {key: {'correct': sum(r[key] for r in rows), 'total': len(rows),
                              'fraction': sum(r[key] for r in rows) / len(rows)} for key in keys},
            'categories': dict(Counter(r['category'] for r in rows)),
            'execution_categories': dict(Counter(r['execution_category'] for r in rows
                                                 if 'execution_category' in r))}


def paired(base, adapted):
    if not base or [r['id'] for r in base] != [r['id'] for r in adapted]:
        raise ValueError('Unpaired rows')
    b, a = aggregate(base), aggregate(adapted)
    return {key: {'base': b['metrics'][key], 'adapted': a['metrics'][key],
                  'delta_pp': 100 * (a['metrics'][key]['fraction'] - b['metrics'][key]['fraction']),
                  'fail_to_pass': sum(not x[key] and y[key] for x, y in zip(base, adapted, strict=True)),
                  'pass_to_fail': sum(x[key] and not y[key] for x, y in zip(base, adapted, strict=True))}
            for key in b['metrics']}

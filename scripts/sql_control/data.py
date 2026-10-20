"""Frozen selection and prompts; no model outputs influence the cohorts."""

from collections import defaultdict
import random
import re
import unicodedata

from scripts.freight import digest


def words(value):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', value).casefold())


def schema_key(record):
    return tuple(words(record['context']))


def near(left, right, threshold=0.85):
    # Same tokenized schema, and bag-of-word question similarity. This catches
    # case, spacing, punctuation and word-order variants, conservatively.
    a, b = set(words(left['question'])), set(words(right['question']))
    return schema_key(left) == schema_key(right) and len(a & b) / max(1, len(a | b)) >= threshold


def identity(record):
    return digest({k: record[k] for k in ('context', 'question', 'answer')})


def prompt(record):
    # Independently authored: the reference repository has no explicit license.
    return ('Write one SQLite SELECT query answering the question using the table definitions. '
            'Return only SQL, without Markdown or explanation.\n\nTable definitions:\n'
            + record['context'].strip() + '\n\nQuestion:\n' + record['question'].strip())


def messages(record, completion=False):
    result = [{'role': 'user', 'content': prompt(record)}]
    if completion:
        result.append({'role': 'assistant', 'content': record['answer'].strip()})
    return result


def select(rows, sizes, seed=42, threshold=0.85, reserved=()):
    """Greedy seeded representatives, test then validation then training.

    Reject all same-schema near matches to any previously selected record,
    including within a split. No SQL correctness/length filtering is used.
    """
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    seen = defaultdict(list)
    for row in reserved:
        seen[schema_key(row)].append(row)
    splits = {name: [] for name in sizes}
    rejected = {'blank_or_invalid': 0, 'duplicate_or_near': 0}
    names = iter(sizes)
    name = next(names)
    for index in indices:
        row = rows[index]
        if not all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ('context', 'question', 'answer')):
            rejected['blank_or_invalid'] += 1
            continue
        group = seen[schema_key(row)]
        if any(near(row, previous, threshold) for previous in group):
            rejected['duplicate_or_near'] += 1
            continue
        record = {k: row[k] for k in ('context', 'question', 'answer')}
        record.update(id=identity(row), source_index=index)
        splits[name].append(record)
        group.append(row)
        if len(splits[name]) == sizes[name]:
            name = next(names, None)
            if name is None:
                return splits, rejected
    raise ValueError('Insufficient unique examples for fixed split sizes')


def validate_splits(splits, sizes, threshold=0.85, reserved=()):
    if set(splits) != set(sizes):
        raise ValueError('Unexpected splits')
    seen, ids = defaultdict(list), set()
    for row in reserved:
        seen[schema_key(row)].append(row)
    for name, size in sizes.items():
        if len(splits[name]) != size:
            raise ValueError('Wrong split size')
        for row in splits[name]:
            if row['id'] != identity(row) or row['id'] in ids:
                raise ValueError('Invalid or duplicate content ID')
            group = seen[schema_key(row)]
            if any(near(row, previous, threshold) for previous in group):
                raise ValueError('Duplicate/near-duplicate leakage')
            ids.add(row['id'])
            group.append(row)

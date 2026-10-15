"""Auditable KodCode ingestion and controlled data studies, reusing experiment.py."""

import argparse
import ast
import builtins
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re
import statistics
import sys
import urllib.request
import warnings

from scripts import experiment as exp

KIND = 'kodcode-study-v1'
PROMPT = 'python-task-hidden-tests-v1'
PINS = exp.ROOT / 'configs/data-study-sources.json'
CONFIG = exp.ROOT / 'configs/data-study.json'
# Explicitly bounded to simple, deterministic, stdlib function tasks.
MODULES = {'math', 'collections', 'itertools', 'functools', 'typing', 're',
           'string', 'heapq', 'bisect', 'operator', 'statistics', 'fractions',
           'decimal', 'copy', 'datetime', 'calendar', 'random', 'json'}
RULES = {'version': KIND, 'text_shingles': 5, 'code_shingles': 7,
         'text_jaccard': 0.8, 'code_jaccard': 0.85,
         'quality': 'at least 5 asserts; 80..2400 question chars; <=2400 code chars; no truncation claim',
         'balance': 'round robin by source subset, then difficulty; seeded hash order within each bucket'}


def local_artifact(path):
    """Keep all corpora and generated experiment state inside ignored local roots."""
    path = Path(path).resolve()
    roots = [exp.ROOT / name for name in ('data', 'datasets', 'runs', 'outputs', 'adapters')]
    if not any(path.is_relative_to(root.resolve()) and path != root.resolve() for root in roots):
        raise ValueError('Study artifacts must be under data/, datasets/, runs/, outputs/, or adapters/.')
    return path


def verified_sources(directory, download=False):
    directory = local_artifact(directory)
    pins = exp.read_json(PINS)
    for pin in pins.values():
        if not re.fullmatch('[0-9a-f]{40}', pin['revision']) or pin['revision'] not in pin['url']:
            raise ValueError('Source URL must include its immutable revision.')
        path = directory / pin['filename']
        if download and not path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.partial')
            with urllib.request.urlopen(pin['url'], timeout=60) as response, temporary.open('wb') as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if exp.file_hash(temporary) != pin['sha256']:
                temporary.unlink()
                raise ValueError('Source checksum mismatch: ' + pin['filename'])
            temporary.replace(path)
        if exp.file_hash(path) != pin['sha256']:
            raise ValueError('Source checksum mismatch: ' + pin['filename'])
    return pins


def parse(code):
    # Parsing never executes imported code, solutions, or tests.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        return ast.parse(code)


def dependencies(tree, solution_import=False):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(n.name.split('.')[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError('relative_import')
            names.add((node.module or '').split('.')[0])
    if names - MODULES - ({'solution'} if solution_import else set()):
        raise ValueError('unsupported_dependency')
    return sorted(names - {'solution'})


def convert(row):
    """Conservatively adapt plain zero-argument assertion tests to the existing scorer."""
    question, code, tests = (row.get(k) for k in ('question', 'solution', 'test'))
    if not all(isinstance(v, str) and v.strip() for v in (question, code, tests)):
        raise ValueError('missing_fields')
    if not 40 <= len(question) <= 12000 or not 30 <= len(code) <= 12000 or len(tests) > 30000:
        raise ValueError('length_bounds')
    try:
        solution, suite = parse(code), parse(tests)
    except (SyntaxError, ValueError):
        raise ValueError('invalid_python') from None
    imports = dependencies(solution) + dependencies(suite, solution_import=True)
    functions = [n for n in solution.body if isinstance(n, ast.FunctionDef)]
    if not functions or any(isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef)) for n in ast.walk(solution)):
        raise ValueError('unsupported_solution')
    if any(isinstance(n, ast.Pass) or
           isinstance(n, ast.Constant) and n.value is Ellipsis or
           isinstance(n, ast.Name) and n.id == 'NotImplementedError' for n in ast.walk(solution)):
        raise ValueError('placeholder_solution')
    retained, calls = [], []
    for node in suite.body:
        if isinstance(node, ast.Import) and any(n.name == 'solution' for n in node.names):
            raise ValueError('unsupported_test_import')
        if isinstance(node, ast.ImportFrom) and node.module == 'solution':
            # Renamed imports require actual module semantics; do not silently rewrite them.
            if any(n.asname or n.name == '*' for n in node.names):
                raise ValueError('unsupported_test_import')
            if not {n.name for n in node.names} <= {n.name for n in functions}:
                raise ValueError('test_solution_mismatch')
            continue
        if isinstance(node, ast.FunctionDef):
            if (not node.name.startswith('test_') or node.decorator_list or
                    node.args.args or node.args.posonlyargs or node.args.kwonlyargs or
                    node.args.vararg or node.args.kwarg):
                raise ValueError('unsupported_test_function')
            calls.append(node.name + '()')
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            raise ValueError('unsupported_test_structure')
        retained.append(node)
    assertions = sum(isinstance(n, ast.Assert) for n in ast.walk(suite))
    if not calls or assertions < 2:
        raise ValueError('insufficient_tests')
    test_code = ast.unparse(ast.Module(body=retained, type_ignores=[])) + '\n' + '\n'.join(calls)
    # Exclude obvious host/process/file operations even though execution is Docker-only.
    forbidden = {'eval', 'exec', 'compile', 'open', '__import__', 'input', 'breakpoint'}
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in forbidden
           for tree in (solution, suite) for n in ast.walk(tree)):
        raise ValueError('unsupported_operations')
    signatures = '\n'.join('def ' + f.name + '(' + ast.unparse(f.args) + '):' for f in functions)
    sample_id = 'kodcode/' + str(row['question_id'])
    return {'task_id': sample_id, 'text': question.strip() + '\n\nRequired signatures:\n' + signatures,
            'prompt_style': PROMPT, 'code': code.strip() + '\n', 'test_list': [test_code], 'test_setup_code': '',
            'provenance': {'source_id': row['question_id'], 'source_row_sha256': exp.digest(row),
                           'sample_sha256': exp.digest([question, code, tests]),
                           'subset': row.get('subset', 'unknown'),
                           'difficulty': row.get('gpt_difficulty', 'unknown'),
                           'question_chars': len(question), 'code_chars': len(code),
                           'assertions': assertions, 'dependencies': sorted(set(imports)),
                           'quality': assertions >= 5 and 80 <= len(question) <= 2400 and len(code) <= 2400}}


class CanonicalCode(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id not in dir(builtins) and node.id not in MODULES:
            node.id = 'identifier'
        return node

    def visit_arg(self, node):
        node.arg, node.annotation = 'argument', None
        return node

    def visit_FunctionDef(self, node):
        node.name, node.returns = 'function', None
        node.body = [n for n in node.body if not (isinstance(n, ast.Expr) and
                     isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        return self.generic_visit(node)


def signatures(text, code):
    words = re.findall(r'\w+', text.lower())
    try:
        tree = CanonicalCode().visit(parse(code))
        tokens = re.findall(r'\w+|[^\w\s]', ast.dump(tree, include_attributes=False))
    except (SyntaxError, ValueError):
        tokens = re.findall(r'\w+', code)
    def shingles(values, width):
        return frozenset(tuple(values[i:i + width]) for i in range(max(1, len(values) - width + 1)))
    return (shingles(words, RULES['text_shingles']), shingles(tokens, RULES['code_shingles']))


class OverlapIndex:
    """Exact Jaccard via inverted indexes; no probabilistic missed candidates."""
    def __init__(self):
        self.values = []
        self.postings = [defaultdict(list), defaultdict(list)]

    def match(self, signature):
        for axis, (values, threshold) in enumerate(zip(signature, (0.8, 0.85))):
            prefix = sorted(values, key=lambda v: (len(self.postings[axis].get(v, ())), v))[:
                len(values) - math.ceil(threshold * len(values)) + 1]
            candidates = {i for value in prefix for i in self.postings[axis].get(value, ())}
            for i in sorted(candidates):
                previous, label = self.values[i]
                if min(len(values), len(previous[axis])) < threshold * max(len(values), len(previous[axis])):
                    continue
                intersection = len(values & previous[axis])
                if intersection / len(values | previous[axis]) >= threshold:
                    return label
        return None

    def add(self, signature, label):
        i = len(self.values)
        self.values.append((signature, label))
        for axis, values in enumerate(signature):
            for value in values:
                self.postings[axis][value].append(i)


def historical(directory):
    directory = Path(directory)
    rows = exp.read_jsonl(directory / 'mbpp.jsonl')
    with gzip.open(directory / 'humaneval.jsonl.gz', 'rt') as stream:
        human = [json.loads(line) for line in stream if line.strip()]
    return ([(f'mbpp/{r["task_id"]}', r['text'], r['code']) for r in rows] +
            [(r['task_id'], r['prompt'], r['prompt'] + r['canonical_solution']) for r in human])


def eligible_rows(rows, history):
    index = OverlapIndex()
    for label, text, code in history:
        index.add(signatures(text, code), label)
    eligible, rejected, seen = [], [], set()
    for row in sorted(rows, key=lambda r: r['question_id']):
        label = str(row['question_id'])
        try:
            if label in seen:
                raise ValueError('duplicate_source_id')
            seen.add(label)
            sample = convert(row)
            signature = signatures(row['question'], row['solution'])
            overlap = index.match(signature)
            if overlap:
                rejected.append({'source_id': label, 'reason': 'overlap', 'with': overlap})
                continue
            index.add(signature, sample['task_id'])
            eligible.append(sample)
        except ValueError as exc:
            rejected.append({'source_id': label, 'reason': str(exc)})
    return eligible, rejected


def ordered(rows, seed):
    return sorted(rows, key=lambda r: exp.digest([seed, r['task_id']]))


def select(rows, seed, sizes, validation_size, eval_size):
    if (not sizes or sizes != sorted(set(sizes)) or any(type(n) is not int or n <= 0 for n in sizes)
            or validation_size <= 0 or eval_size <= 0):
        raise ValueError('Sizes must be distinct increasing positive integers; holdouts must be positive.')
    rows = ordered(rows, seed)
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate eligible sample IDs.')
    reserve = validation_size + 2 * eval_size
    if len(rows) < reserve + max(sizes):
        raise ValueError(f'Only {len(rows)} eligible examples; need {reserve + max(sizes)}. No duplication or undersized sets.')
    valid, development, confirmation, pool = (rows[:validation_size], rows[validation_size:validation_size + eval_size],
        rows[validation_size + eval_size:reserve], rows[reserve:])
    buckets = defaultdict(list)
    for r in pool:
        if r['provenance']['quality']:
            buckets[(r['provenance']['subset'], r['provenance']['difficulty'])].append(r)
    balanced = []
    while buckets:
        for key in sorted(list(buckets)):
            balanced.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
    if len(balanced) < min(sizes):
        raise ValueError(f'Only {len(balanced)} quality candidates; cannot make same-size comparison.')
    return {'valid': valid, 'development': development, 'confirmation': confirmation,
            **{f'size-{n}': pool[:n] for n in sizes}, f'quality-{min(sizes)}': balanced[:min(sizes)]}


def diagnostics(rows):
    return {'count': len(rows), 'quality_candidates': sum(r['provenance']['quality'] for r in rows),
            'subsets': dict(Counter(r['provenance']['subset'] for r in rows)),
            'difficulty': dict(Counter(r['provenance']['difficulty'] for r in rows)),
            'dependencies': dict(Counter(m for r in rows for m in r['provenance']['dependencies'])),
            **{key: {'min': min((r['provenance'][key] for r in rows), default=None),
                     'max': max((r['provenance'][key] for r in rows), default=None),
                     'mean': sum(r['provenance'][key] for r in rows) / len(rows) if rows else None}
               for key in ('question_chars', 'code_chars', 'assertions')}}


def evaluation_row(row):
    return {k: row[k] for k in ('task_id', 'text', 'prompt_style', 'test_list', 'test_setup_code')}


def prepare(args, config):
    validate_study_config(config)
    path = local_artifact(args.output)
    if path.exists():
        raise ValueError('Study output already exists; choose a fresh path.')
    pins = verified_sources(args.sources, download=args.download)
    import pyarrow.parquet as pq
    rows = pq.read_table(Path(args.sources) / pins['kodcode']['filename']).to_pylist()
    history = historical(args.sources)
    eligible, rejected = eligible_rows(rows, history)
    audit = {'source_rows': len(rows), 'historical_rows': len(history), 'eligible': len(eligible),
             'reasons': dict(Counter(r['reason'] for r in rejected)), 'diagnostics': diagnostics(eligible)}
    # On insufficient coverage save the audit, never partial/mislabelled training sets.
    exp.fresh_directory(path)
    exp.write_json(path / 'audit.json', audit)
    exp.write_jsonl(path / 'rejections.jsonl', rejected)
    groups = select(eligible, config['data_seed'], args.sizes, config['validation_size'], config['eval_size'])
    exp.write_jsonl(path / 'samples.jsonl', [{'task_id': r['task_id'], **r['provenance']} for r in eligible])
    files = {}
    for name in ('development', 'confirmation'):
        exp.write_jsonl(path / f'{name}.jsonl', [evaluation_row(r) for r in groups[name]])
        files[f'{name}.jsonl'] = exp.file_hash(path / f'{name}.jsonl')
    for name, group in groups.items():
        if not name.startswith(('size-', 'quality-')):
            continue
        directory = exp.fresh_directory(path / name)
        for split, samples in (('train', group), ('valid', groups['valid'])):
            exp.write_jsonl(directory / f'{split}.jsonl', [
                {'messages': exp.messages(r) + [{'role': 'assistant', 'content': r['code']}]} for r in samples])
        manifest = {'kind': KIND, 'selection': name, 'study': str(path), 'seed': config['data_seed'],
                    'prompt_version': PROMPT, 'ids': {'train': [r['task_id'] for r in group],
                        'valid': [r['task_id'] for r in groups['valid']]},
                    'diagnostics': diagnostics(group),
                    'files': {n: exp.file_hash(directory / n) for n in ('train.jsonl', 'valid.jsonl')}}
        exp.write_json(directory / 'manifest.json', manifest)
        for n in ('train.jsonl', 'valid.jsonl', 'manifest.json'):
            files[f'{name}/{n}'] = exp.file_hash(directory / n)
    for name in ('audit.json', 'rejections.jsonl', 'samples.jsonl'):
        files[name] = exp.file_hash(path / name)
    manifest = {'kind': KIND, 'sources': pins, 'rules': RULES, 'configuration': config,
                'sizes': args.sizes, 'ids': {k: [r['task_id'] for r in v] for k, v in groups.items()},
                'files': files, 'code': exp.code_state(),
                'limitations': 'Static validity, not proved correctness. Approximate lexical/AST overlap excludes '
                'known matches, not semantic paraphrases or public pretraining contamination. Holdouts are new '
                'to this workflow, not independently authored benchmarks.'}
    manifest['study_sha256'] = exp.digest(manifest)
    exp.write_json(path / 'study.json', manifest)
    study_info(path)
    print(json.dumps(audit, indent=2))
    print('Frozen study: ' + str(path))


def study_info(path):
    path = local_artifact(path)
    info = exp.read_json(path / 'study.json')
    payload = {k: v for k, v in info.items() if k != 'study_sha256'}
    if info['kind'] != KIND or exp.digest(payload) != info['study_sha256']:
        raise ValueError('Study manifest integrity failure.')
    if info['sources'] != exp.read_json(PINS) or info['rules'] != RULES:
        raise ValueError('Study sources/rules differ from the inspected pins.')
    for name, checksum in info['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or exp.file_hash(path / name) != checksum:
            raise ValueError('Study file changed: ' + name)
    ids, sizes = info['ids'], info['sizes']
    if not sizes or sizes != sorted(set(sizes)) or any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError('Invalid nested subset sizes.')
    selections = {f'size-{n}' for n in sizes} | {f'quality-{min(sizes)}'}
    required_files = {'audit.json', 'rejections.jsonl', 'samples.jsonl', 'development.jsonl', 'confirmation.jsonl'}
    required_files.update(f'{name}/{file}' for name in selections
                          for file in ('manifest.json', 'train.jsonl', 'valid.jsonl'))
    if set(info['files']) != required_files or set(ids) != selections | {'valid', 'development', 'confirmation'}:
        raise ValueError('Incomplete study manifest inventory.')
    if (len(ids['valid']) != info['configuration']['validation_size'] or
            any(len(ids[name]) != info['configuration']['eval_size'] for name in ('development', 'confirmation'))):
        raise ValueError('Holdout counts differ from the preregistered configuration.')
    holdouts = ids['valid'] + ids['development'] + ids['confirmation']
    if len(set(holdouts)) != len(holdouts):
        raise ValueError('Holdout splits overlap.')
    previous = []
    for n in sizes:
        current = ids[f'size-{n}']
        if len(current) != n or current[:len(previous)] != previous:
            raise ValueError('Invalid nested subsets.')
        previous = current
    for key, values in ids.items():
        if len(values) != len(set(values)) or (key.startswith(('size-', 'quality-')) and set(values) & set(holdouts)):
            raise ValueError('Split leakage or duplicate IDs.')
    if len(ids[f'quality-{min(sizes)}']) != min(sizes):
        raise ValueError('Quality comparison size differs.')
    samples = exp.read_jsonl(path / 'samples.jsonl')
    lookup = {r['task_id']: r for r in samples}
    if len(lookup) != len(samples) or any(i not in lookup for group in ids.values() for i in group):
        raise ValueError('Missing or duplicate sample provenance.')
    if any(not lookup[i]['quality'] for i in ids[f'quality-{min(sizes)}']):
        raise ValueError('Quality selection contains ineligible samples.')
    for name in ('development', 'confirmation'):
        rows = exp.read_jsonl(path / f'{name}.jsonl')
        if [r['task_id'] for r in rows] != ids[name] or any(set(r) != set(evaluation_row(r)) for r in rows):
            raise ValueError('Evaluation IDs differ or reference fields leaked.')
    return info


def dataset_info(path):
    path = local_artifact(path)
    data = exp.read_json(path / 'manifest.json')
    study = study_info(data['study'])
    name = data['selection']
    if (not path.samefile(Path(data['study']) / name) or data['ids'] != {
            'train': study['ids'][name], 'valid': study['ids']['valid']}):
        raise ValueError('Training manifest differs from frozen study.')
    for split in ('train', 'valid'):
        rows = exp.read_jsonl(path / f'{split}.jsonl')
        if len(rows) != len(data['ids'][split]) or exp.file_hash(path / f'{split}.jsonl') != data['files'][f'{split}.jsonl']:
            raise ValueError('Training split changed.')
    return {**data, 'study_sha256': study['study_sha256']}


def training_settings(config, data, args):
    validate_study_config(config)
    if getattr(args, 'mask_variant', 'corrected') != 'corrected' or args.resume_adapter is not None:
        raise ValueError('Data studies require fresh corrected-mask adapters.')
    study = study_info(data['study'])
    expected = study['configuration']
    # Training seed may vary for replicas; each study fixes its exposure protocol.
    for key in ('model_id', 'model_revision', 'dtype', 'decoding', 'scoring', 'data_seed', 'validation_size', 'eval_size'):
        if config[key] != expected[key]:
            raise ValueError('Study configuration mismatch: ' + key)
    t = config['training']
    reference = exp.read_json(CONFIG)['training']
    for key in reference:
        if key not in ('seed', 'iters', 'steps_per_report', 'steps_per_eval', 'save_every', 'val_batches') and t[key] != reference[key]:
            raise ValueError('Controlled training setting differs: ' + key)
    if t.get('grad_accumulation_steps', 1) != 1 or t.get('lr_schedule') or t.get('optimizer_config'):
        raise ValueError('Undeclared optimizer or accumulation changes.')
    policy = config['study']
    if policy != expected['study']:
        raise ValueError('Exposure policy differs from the preregistered study.')
    n = len(data['ids']['train'])
    if policy['policy'] == 'fixed-epochs' and type(policy.get('epochs')) is int and policy['epochs'] > 0:
        steps = n * policy['epochs']
    elif policy['policy'] == 'fixed-compute' and type(policy.get('steps')) is int and policy['steps'] > 0:
        steps = policy['steps']
    else:
        raise ValueError('Declare positive fixed-epochs/epochs or fixed-compute/steps.')
    return {**t, 'iters': steps, 'study_exposure': {**policy, 'sample_ids': data['ids']['train'],
            'study_sha256': data['study_sha256']}}


def validate_study_config(config):
    reference = exp.read_json(CONFIG)
    for key in ('model_id', 'model_revision', 'dtype', 'decoding', 'scoring'):
        if config[key] != reference[key]:
            raise ValueError('Controlled study setting differs: ' + key)


def validate_exposure(runtime, configuration):
    policy = configuration['study_exposure']
    observed = runtime['exposure']
    counts = observed['sample_counts']
    ids = policy['sample_ids']
    if (any(type(v) is not int or v <= 0 for v in counts.values()) or
            len(ids) != len(set(ids)) or set(counts) - set(ids) or sum(counts.values()) != configuration['iters'] or
            observed['samples_seen'] != configuration['iters'] or
            observed['unique_samples_seen'] != len(counts) or observed['target_tokens_seen'] <= 0 or
            observed['retained_tokens_seen'] <= observed['target_tokens_seen'] or
            not observed.get('order_sha256') or runtime['train_batches'] != configuration['iters']):
        raise ValueError('Invalid effective exposure accounting.')
    if policy['policy'] == 'fixed-epochs' and (set(counts) != set(ids) or
            any(counts[i] != policy['epochs'] for i in ids)):
        raise ValueError('Fixed-epoch study did not expose every sample exactly as declared.')


def freeze(args, config):
    study = study_info(args.study)
    destination = Path(args.study) / 'choice.json'
    if destination.exists() or (Path(args.study) / 'confirmation-access').exists():
        raise ValueError('Choice is already frozen or confirmation was attempted.')
    baseline = exp.read_json(args.baseline)
    adapters = []
    for path in args.adapted:
        result = exp.read_json(path)
        exp.compare_results(baseline, result)
        if result['protocol'].get('study_sha256') != study['study_sha256'] or result['protocol']['split'] != 'development':
            raise ValueError('Freeze requires development results from this study.')
        adapters.append(result['adapter'])
    choice = {**exp.provenance(), 'study_sha256': study['study_sha256'],
        'configuration': config, 'decision': args.decision, 'adapters': adapters,
        'development_inputs': [{'path': str(Path(p).resolve()), 'sha256': exp.file_hash(p)}
                               for p in [args.baseline, *args.adapted]],
        'protocol': baseline['protocol']}
    with destination.open('x') as stream:
        json.dump(choice, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('Frozen choice: ' + str(destination))


def report(args, config):
    """Keep size contrasts and same-size selection contrasts explicitly separate."""
    baseline = exp.read_json(args.baseline)
    comparisons = []
    controlled = None
    for path in args.adapted:
        result = exp.read_json(path)
        comparison = exp.compare_results(baseline, result)
        record = result['adapter']['training']
        if not result['protocol'].get('study_sha256'):
            raise ValueError('Study report requires external-study evaluations.')
        configuration = dict(record['configuration'])
        policy = dict(configuration.pop('study_exposure'))
        policy.pop('sample_ids')
        for key in ('data', 'adapter_path', 'seed', 'iters'):
            configuration.pop(key, None)
        value = (configuration, policy, record['versions'], record['code'], record['loss_mask'])
        if controlled is not None and value != controlled:
            raise ValueError('Training controls differ beyond subset/seed/declared exposure.')
        controlled = value
        comparison.update(input_sha256=exp.file_hash(path),
                          policy={k: v for k, v in policy.items() if k != 'study_sha256'},
                          elapsed_seconds=record['runtime']['elapsed_seconds'])
        comparisons.append(comparison)
    by_selection = defaultdict(list)
    seen = set()
    for c in comparisons:
        key = (c['selection'], c['training_seed'])
        if key in seen:
            raise ValueError('Duplicate selection/seed; cannot count repeated scoring as seed replication.')
        seen.add(key)
        by_selection[c['selection']].append(c)
    summary = {name: {'seeds': [c['training_seed'] for c in group],
                     'mean_pass_at_1_percent': statistics.mean(c['adapted']['pass_at_1_percent'] for c in group),
                     'seed_stdev_percentage_points': statistics.stdev(c['adapted']['pass_at_1_percent'] for c in group)
                        if len(group) > 1 else None} for name, group in by_selection.items()}
    size, quality, incomplete = [], [], []
    for x in comparisons:
        for y in comparisons:
            if x['training_seed'] != y['training_seed']:
                continue
            nx, ny = int(x['selection'].split('-')[1]), int(y['selection'].split('-')[1])
            contrast = {'from': x['selection'], 'to': y['selection'], 'seed': x['training_seed'],
                        'delta_percentage_points': y['adapted']['pass_at_1_percent'] - x['adapted']['pass_at_1_percent']}
            if x['selection'].startswith('size-') and y['selection'].startswith('size-') and nx < ny:
                if x['exposure']['unique_samples_seen'] < nx or y['exposure']['unique_samples_seen'] < ny:
                    incomplete.append(dict(contrast, reason='Incomplete pool exposure; cannot interpret as a size effect.'))
                else:
                    size.append(contrast)
            if x['selection'].startswith('size-') and y['selection'].startswith('quality-') and nx == ny:
                quality.append(contrast)
    output = local_artifact(args.output)
    if output.exists():
        raise ValueError('Report output exists; use a fresh path.')
    exp.write_json(output, {**exp.provenance(), 'baseline_sha256': exp.file_hash(args.baseline),
        'comparisons': comparisons, 'summary': summary, 'size_contrasts': size, 'quality_contrasts': quality,
        'exposure_limited_contrasts': incomplete,
        'limitations': 'Intervals are descriptive paired normal approximations, unadjusted for multiple '
        'comparisons. Seed SD is unavailable for one run. Fixed epochs differ in steps/tokens/compute; '
        'fixed compute differs in coverage/reuse. Inspect exposure before interpreting any size contrast. '
        'Quality selection changes length/test-count/subset composition together; it does not prove '
        'causation by any one quality dimension. Public synthetic holdouts may share semantic/pretraining '
        'content. Only the locked one-time confirmation can confirm a frozen development choice.'})
    print('Report: ' + str(output))


def confirmation_access(path, study, config, adapter, model):
    choice_path = Path(path) / 'choice.json'
    choice = exp.read_json(choice_path)
    if (choice['study_sha256'] != study['study_sha256'] or config != choice['configuration'] or
            model != choice['protocol']['model'] or adapter is not None and adapter not in choice['adapters']):
        raise ValueError('Confirmation choice/model/configuration differs from the frozen decision.')
    directory = Path(path) / 'confirmation-access'
    directory.mkdir(exist_ok=True)
    # A failed attempt also consumes access: no fresh decoding retries after seeing outcomes.
    key = exp.digest(None if adapter is None else adapter['files'])
    with (directory / f'{key}.json').open('x') as stream:
        json.dump({'choice_sha256': exp.file_hash(choice_path), **exp.provenance()}, stream)
    return exp.file_hash(choice_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(CONFIG))
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--sources', default='datasets/issue13-sources')
    p.add_argument('--download', action='store_true')
    p.add_argument('--sizes', type=int, nargs='+', default=[300, 1000, 3000])
    p.add_argument('--output', required=True)
    p = commands.add_parser('verify')
    p.add_argument('--study', required=True)
    p = commands.add_parser('freeze')
    p.add_argument('--study', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--adapted', nargs='+', required=True)
    p.add_argument('--decision', required=True, help='Development-based choice and planned comparison; immutable before confirmation')
    p = commands.add_parser('report')
    p.add_argument('--baseline', required=True)
    p.add_argument('--adapted', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        config = exp.read_json(args.config)
        exp.validate_config(config)
        if args.command == 'verify':
            info = study_info(args.study)
            print('Verified study ' + info['study_sha256'])
        else:
            globals()[args.command](args, config)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

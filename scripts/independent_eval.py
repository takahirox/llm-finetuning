"""Pinned HumanEval full-module chat evaluation using the existing Docker sandbox.

This is a chat adaptation, not upstream prompt-plus-body completion sampling.
Hidden tests and canonical solutions are never supplied to generation.
"""
import argparse
import ast
import copy
import gzip
import json
import keyword
from pathlib import Path
import re
import statistics

from scripts import experiment as ex
from scripts.evaluation_sandbox import score

REVISION = '6d43fb980f9fee3c892a914eda09951f772ad10d'
URL = f'https://raw.githubusercontent.com/openai/human-eval/{REVISION}/data/HumanEval.jsonl.gz'
ARCHIVE_SHA256 = 'b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef'
PROMPT_VERSION = 'humaneval-full-module-chat-v1'
TEST_VERSION = 'humaneval-hidden-check-v1'
SEEDS = (43, 44, 45)


def messages(problem):
    return [{'role': 'user', 'content': (
        'Write a complete Python module solving the function specification below. '
        'Define the requested function with the given name and signature, and include '
        'any required imports or helpers. Return only Python code, without Markdown '
        'fences or explanation. Do not include tests.\n\n' + problem['prompt'])}]


def test_case(problem):
    entry = problem['entry_point']
    if not entry.isidentifier() or keyword.iskeyword(entry):
        raise ValueError('Invalid entry point.')
    return {'test_setup_code': '', 'test_list': [problem['test'] + f'\ncheck({entry})\n']}


def assembled_solution(problem, solution):
    # Preserve the supplied prefix imports/declarations, then override its stub
    # with the requested complete module. Never append hidden tests here.
    return problem['prompt'] + '\n' + solution


def prepare(archive, directory):
    import hashlib
    raw = Path(archive).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('HumanEval archive checksum differs from the immutable pin.')
    rows = [json.loads(line) for line in gzip.decompress(raw).splitlines() if line.strip()]
    rows.sort(key=lambda r: int(r['task_id'].split('/')[1]))
    if [r['task_id'] for r in rows] != [f'HumanEval/{i}' for i in range(164)]:
        raise ValueError('Expected all 164 official task IDs exactly once.')
    for row in rows:
        test_case(row)
        ast.parse(row['prompt'] + row['canonical_solution'])
        ast.parse(row['test'])
    path = ex.fresh_directory(directory)
    problems = [{k: r[k] for k in ('task_id', 'prompt', 'entry_point', 'test')} for r in rows]
    references = [{'task_id': r['task_id'], 'program': r['prompt'] + r['canonical_solution']} for r in rows]
    ex.write_jsonl(path / 'problems.jsonl', problems)
    ex.write_jsonl(path / 'references.jsonl', references)
    ex.write_json(path / 'manifest.json', {'source': URL, 'revision': REVISION,
        'archive_sha256': ARCHIVE_SHA256, 'task_ids': [r['task_id'] for r in rows],
        'files': {name: ex.file_hash(path / name) for name in ('problems.jsonl', 'references.jsonl')}})


def dataset_info(directory):
    path = Path(directory)
    manifest = ex.read_json(path / 'manifest.json')
    if (manifest['source'], manifest['revision'], manifest['archive_sha256']) != (URL, REVISION, ARCHIVE_SHA256):
        raise ValueError('Dataset provenance differs from pinned HumanEval.')
    if set(manifest['files']) != {'problems.jsonl', 'references.jsonl'}:
        raise ValueError('Unexpected data file manifest.')
    for name, expected in manifest['files'].items():
        if ex.file_hash(path / name) != expected:
            raise ValueError(f'Dataset changed: {name}.')
    rows = ex.read_jsonl(path / 'problems.jsonl')
    if [r['task_id'] for r in rows] != manifest['task_ids'] or manifest['task_ids'] != [f'HumanEval/{i}' for i in range(164)]:
        raise ValueError('Dataset task IDs/order changed.')
    if any(set(r) != {'task_id', 'prompt', 'entry_point', 'test'} for r in rows):
        raise ValueError('Unexpected generation record fields.')
    return manifest, rows


def function_fingerprints(source):
    """Conservative exact alpha-normalized main-function-body fingerprints."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    fingerprints = set()
    for function in (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        function = copy.deepcopy(function)
        locals_ = {a.arg for a in ast.walk(function) if isinstance(a, ast.arg)}
        if function.args.vararg: locals_.add(function.args.vararg.arg)
        if function.args.kwarg: locals_.add(function.args.kwarg.arg)
        locals_.update(n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
        locals_.update(n.name for n in ast.walk(function) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        locals_.add(function.name)
        names = {}

        class Normalize(ast.NodeTransformer):
            def renamed(self, name):
                if name not in names: names[name] = f'local_{len(names)}'
                return names[name]

            def visit_FunctionDef(self, node):
                node.name = self.renamed(node.name)
                node.returns = None
                node.decorator_list = []
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body = node.body[1:]
                return self.generic_visit(node)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_arg(self, node):
                node.arg = self.renamed(node.arg)
                node.annotation = None
                return node

            def visit_Name(self, node):
                if node.id in locals_: node.id = self.renamed(node.id)
                return node

        normalized = Normalize().visit(function)
        fingerprints.add(ast.dump(normalized, include_attributes=False))
    return fingerprints


def words(text):
    return set(re.findall(r'[a-z]+', text.lower()))


def overlap_audit(problems, references, mbpp_rows):
    """Fixed pre-output rules; compare against all previously used MBPP data."""
    reference_map = {r['task_id']: r['program'] for r in references}
    training = [(r['task_id'], words(r['text']), function_fingerprints(r['code'])) for r in mbpp_rows]
    excluded = []
    details = []
    for problem in problems:
        prompt_words = words(ast.get_docstring(next(n for n in ast.parse(problem['prompt']).body
            if isinstance(n, ast.FunctionDef) and n.name == problem['entry_point'])) or problem['prompt'])
        functions = function_fingerprints(reference_map[problem['task_id']])
        matches = []
        for task_id, tokens, candidate_functions in training:
            shared = len(prompt_words & tokens)
            jaccard = shared / len(prompt_words | tokens) if prompt_words | tokens else 0.0
            same_function = bool(functions & candidate_functions)
            if same_function or (shared >= 8 and jaccard >= 0.8):
                matches.append({'mbpp_task_id': task_id, 'exact_normalized_function': same_function,
                                'prompt_jaccard': jaccard, 'shared_words': shared})
        if matches:
            excluded.append(problem['task_id'])
            details.append({'task_id': problem['task_id'], 'matches': matches})
    return {'excluded_ids': excluded,
            'primary_ids': [r['task_id'] for r in problems if r['task_id'] not in excluded],
            'matches': details, 'mbpp_compared_count': len(training),
            'limit': 'Conservative syntactic/textual screen; semantic overlap and public pretraining contamination cannot be ruled out.'}


def audit_references(directory, config, output, image=ex.DEFAULT_IMAGE):
    if Path(output).exists(): raise ValueError('Output exists.')
    manifest, problems = dataset_info(directory)
    refs = ex.read_jsonl(Path(directory) / 'references.jsonl')
    if [r['task_id'] for r in refs] != manifest['task_ids']: raise ValueError('Reference order differs.')
    protocol = ex.scoring_protocol(config, image)
    result = {'status': 'running', 'dataset': manifest, 'scoring': protocol, 'problems': []}
    for problem, reference in zip(problems, refs):
        outcome = score(reference['program'], test_case(problem), protocol['image'], protocol['scoring'])
        result['problems'].append({'task_id': problem['task_id'], **outcome})
        ex.write_json(output, result)
        print(problem['task_id'], 'reference', outcome['status'], flush=True)
        if not outcome['passed']: raise ValueError('Canonical reference failed; abort before model evaluation.')
    result['status'] = 'complete'
    ex.write_json(output, result)


def evaluate(args, config):
    if Path(args.output).exists(): raise ValueError('Output exists.')
    ex.require_mlx()
    manifest, problems = dataset_info(args.data)
    plan = ex.read_json(args.plan)
    if plan['dataset'] != manifest: raise ValueError('Frozen dataset differs.')
    if config['decoding'] != plan['decoding'] or config['scoring'] != plan['scoring']:
        raise ValueError('Frozen decoding/scoring differs.')
    model_info = ex.model_info(args.model, config)
    adapter = ex.adapter_info(args.adapter, model_info)
    if adapter:
        training = adapter['training']
        if (training['configuration']['seed'] not in SEEDS or training['configuration']['iters'] != 50 or
                training['configuration']['learning_rate'] != 1e-6 or training['resume'] is not None or
                training['loss_mask']['variant'] != 'corrected' or training['runtime']['train_batches'] != 50 or
                training['code'] != ex.code_state() or ex.stable_dataset(training['dataset']) != plan['training_dataset']):
            raise ValueError('Training differs from the frozen independent-seed experiment.')
        if any(training['configuration'].get(k) != v for k, v in plan['training_hyperparameters'].items()):
            raise ValueError('Training hyperparameters differ beyond the seed.')
        runtime = training['runtime']
        if (runtime['status'] != 'complete' or runtime['trainer_calls'] != 1 or runtime['validation_batches'] != 12 or
                runtime['configuration_sha256'] != ex.digest(training['configuration']) or
                training['versions'] != ex.versions()):
            raise ValueError('Invalid training runtime evidence.')
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    model, tokenizer = load(args.model, adapter_path=args.adapter, tokenizer_config={'trust_remote_code': False})
    sampler = make_sampler(temp=config['decoding']['temperature'], top_p=config['decoding']['top_p'])
    protocol = {**ex.scoring_protocol(config, ex.DEFAULT_IMAGE), 'test_version': TEST_VERSION,
        'dataset': manifest, 'examples': problems, 'model': model_info, 'versions': ex.versions(),
        'decoding': config['decoding'], 'prompt_version': PROMPT_VERSION,
        'generation_harness': ex.file_hash(__file__), 'code': ex.code_state(),
        'plan_sha256': ex.file_hash(args.plan)}
    result = {**ex.provenance(), 'status': 'running', 'generation_status': 'running',
              'adapter': adapter, 'protocol': protocol, 'problems': []}
    ex.write_json(args.output, result)
    for problem in problems:
        mx.random.seed(config['decoding']['seed'])
        prompt = tokenizer.apply_chat_template(messages(problem), tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=config['decoding']['max_tokens'], sampler=sampler, verbose=False)
        try: solution = ex.extract_code(response)
        except ValueError: solution = ''
        result['problems'].append({'task_id': problem['task_id'], 'response': response, 'solution': solution})
        ex.write_json(args.output, result)
        print(problem['task_id'], 'generated', flush=True)
    result['generations_sha256'] = ex.validate_generations(result)
    result['generation_status'] = 'complete'
    ex.write_json(args.output, result)
    for i, (problem, row) in enumerate(zip(problems, list(result['problems']))):
        outcome = score(assembled_solution(problem, row['solution']), test_case(problem), protocol['image'], protocol['scoring']) if row['solution'] else {
            'passed': False, 'status': 'format_error', 'exit_code': None}
        result['problems'][i] = {**row, **outcome}
        ex.write_json(args.output, result)
        print(problem['task_id'], outcome['status'], flush=True)
    result.update(status='complete', summary=ex.aggregate(result['problems']))
    ex.write_json(args.output, result)


def validate_result(result):
    if result['status'] != 'complete' or result['generation_status'] != 'complete':
        raise ValueError('Incomplete evaluation.')
    ex.validate_generations(result)
    expected = [f'HumanEval/{i}' for i in range(164)]
    if [r['task_id'] for r in result['protocol']['examples']] != expected:
        raise ValueError('Expected the complete fixed 164-task protocol.')
    if result['protocol']['prompt_version'] != PROMPT_VERSION or result['protocol']['test_version'] != TEST_VERSION:
        raise ValueError('Wrong evaluation protocol.')
    for row in result['problems']:
        if (row['status'] not in ('passed', 'failed', 'format_error', 'timeout') or
                type(row['passed']) is not bool or row['passed'] != (row['status'] == 'passed') or
                (row['status'] == 'passed' and row['exit_code'] != 0) or
                (row['status'] == 'failed' and (type(row['exit_code']) is not int or row['exit_code'] == 0)) or
                (row['status'] in ('format_error', 'timeout') and row['exit_code'] is not None) or
                (row['status'] == 'format_error') != (not row['solution'])):
            raise ValueError('Invalid score outcome.')
    if ex.aggregate(result['problems']) != result['summary']: raise ValueError('Summary differs from rows.')


def summarize(results, primary_ids):
    if len(results) != 4 or results[0]['adapter'] is not None: raise ValueError('Expected base and three seeds.')
    for result in results:
        validate_result(result)
        if result['protocol'] != results[0]['protocol']: raise ValueError('Evaluation protocols differ.')
    seeds = [r['adapter']['training']['configuration']['seed'] for r in results[1:]]
    if sorted(seeds) != list(SEEDS): raise ValueError('Expected distinct seeds 43,44,45.')
    if not primary_ids or len(set(primary_ids)) != len(primary_ids): raise ValueError('Invalid primary cohort.')
    known = {r['task_id'] for r in results[0]['problems']}
    if not set(primary_ids) <= known: raise ValueError('Unknown cohort task.')
    reports = {}
    for cohort, ids in (('primary', set(primary_ids)), ('full', known)):
        rows = [[r for r in result['problems'] if r['task_id'] in ids] for result in results]
        scores = [ex.aggregate(r) for r in rows]
        pairs = []
        for seed, after in zip(seeds, rows[1:]):
            changes = {k: [] for k in ('fail_to_pass', 'pass_to_fail', 'unchanged_pass', 'unchanged_fail')}
            for x, y in zip(rows[0], after):
                if x['task_id'] != y['task_id']: raise ValueError('Task pairing differs.')
                key = ('unchanged_pass' if x['passed'] else 'unchanged_fail') if x['passed'] == y['passed'] else ('fail_to_pass' if y['passed'] else 'pass_to_fail')
                changes[key].append(x['task_id'])
            pairs.append({'seed': seed, 'transitions': {k: len(v) for k, v in changes.items()}, 'task_ids': changes})
        rates = [s['pass_at_1_percent'] for s in scores[1:]]
        reports[cohort] = {'scores': scores, 'seeds': seeds, 'mean_percent': statistics.mean(rates),
                          'sample_sd_percentage_points': statistics.stdev(rates),
                          'mean_delta_from_base_pp': statistics.mean(rates) - scores[0]['pass_at_1_percent'], 'pairs': pairs}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('prepare'); sub.add_argument('--archive', required=True); sub.add_argument('--data', required=True)
    sub = commands.add_parser('audit-references'); sub.add_argument('--data', required=True); sub.add_argument('--output', required=True)
    sub = commands.add_parser('evaluate')
    for name in ('model', 'data', 'plan', 'output'): sub.add_argument('--' + name, required=True)
    sub.add_argument('--adapter')
    args = parser.parse_args()
    config = ex.read_json(args.config); ex.validate_config(config)
    if args.command == 'prepare': prepare(args.archive, args.data)
    elif args.command == 'audit-references': audit_references(args.data, config, args.output)
    else: evaluate(args, config)


if __name__ == '__main__': main()

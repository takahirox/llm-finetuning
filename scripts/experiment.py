"""Reproducible Qwen/MLX LoRA experiment; pure data/comparison code needs only stdlib."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import random
import re
import subprocess
import sys
import urllib.request

from scripts.evaluation_sandbox import resolve_image, score

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiment.json'
DATA_REVISION = 'f82046ba5aabbbb427dbfd38a254d26bff08b533'
DATA_URL = f'https://raw.githubusercontent.com/google-research/google-research/{DATA_REVISION}/mbpp/mbpp.jsonl'
DATA_SHA256 = 'ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f'
PROMPT_VERSION = 'mbpp-tests-chat-v1'
SCORER_VERSION = 'docker-mbpp-ready-v2'
TEST_VERSION = 'mbpp-standard-test-list-v1'
DEFAULT_IMAGE = 'python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def versions():
    result = {'python': platform.python_version()}
    for name in ('mlx', 'mlx-lm', 'transformers', 'huggingface-hub', 'numpy'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not installed'
    return result


def provenance():
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'versions': versions()}


def code_state():
    return {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'files': {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / 'scripts').glob('*.py'))},
            'lock_sha256': file_hash(ROOT / 'uv.lock'),
            'project_sha256': file_hash(ROOT / 'pyproject.toml')}


def require_mlx():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or sys.version_info[:2] != (3, 12):
        raise ValueError('MLX stages require native Apple Silicon macOS and Python 3.12; run scripts/check_environment.py.')
    mac_version = platform.mac_ver()[0]
    if not mac_version or int(mac_version.split('.')[0]) < 14:
        raise ValueError('MLX stages require macOS 14 or newer; run scripts/check_environment.py.')
    try:
        import mlx.core as mx
        if not mx.metal.is_available():
            raise ValueError('Metal unavailable; run scripts/check_environment.py.')
        if versions()['mlx-lm'] != '0.32.0' or versions()['mlx'] != '0.32.3':
            raise ValueError('Run uv sync --locked to install the supported MLX versions.')
    except ImportError as exc:
        raise ValueError('MLX missing; run uv sync --locked and scripts/check_environment.py.') from exc


def fresh_directory(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f'{path} already exists; use a new output directory to preserve provenance.')
    path.mkdir(parents=True)
    return path


def messages(problem):
    return [{'role': 'user', 'content': (
        'Write a complete Python solution to the task below. Return only Python code, '
        'without Markdown fences or explanation. Use the function names in these assertions.\n\n'
        + problem['text'].strip() + '\n\nAssertions:\n' + '\n'.join(problem['test_list'])
    )}]


def extract_code(response):
    blocks = re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```', response, flags=re.DOTALL)
    if len(blocks) > 1:
        raise ValueError('Multiple code blocks; ambiguous solution.')
    if blocks:
        return blocks[0].strip() + '\n'
    if '```' in response or not response.strip():
        raise ValueError('Empty solution or incomplete Markdown fence.')
    return response.strip() + '\n'


def select_data(rows, seed, eval_size, validation_size):
    ids = [row['task_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate MBPP task IDs.')
    train = sorted((r for r in rows if 601 <= r['task_id'] <= 974), key=lambda r: r['task_id'])
    heldout = sorted((r for r in rows if 11 <= r['task_id'] <= 510), key=lambda r: r['task_id'])
    if eval_size == 500 and {r['task_id'] for r in heldout} != set(range(11, 511)):
        raise ValueError('Full evaluation requires exactly official MBPP test IDs 11–510.')
    if not 0 < validation_size < len(train) or not 0 < eval_size <= len(heldout):
        raise ValueError('Evaluation/validation sample sizes exceed available split or are empty.')
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(heldout)
    return train[validation_size:], train[:validation_size], heldout[:eval_size]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare_data(args, config):
    # Official immutable source; no datasets dependency or mutable Hub split resolver.
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError('MBPP source checksum mismatch.')
    groups = select_data([json.loads(line) for line in raw.splitlines()], config['data_seed'],
                         config['eval_size'], config['validation_size'])
    path = fresh_directory(args.data)
    for name, rows in zip(('train', 'valid'), groups[:2]):
        write_jsonl(path / f'{name}.jsonl', [
            {'messages': messages(r) + [{'role': 'assistant', 'content': r['code']}]}
            for r in rows
        ])
    # Reference solutions and challenge tests are deliberately absent from evaluation files.
    write_jsonl(path / 'eval.jsonl', [{k: r[k] for k in ('task_id', 'text', 'test_list', 'test_setup_code')}
                                    for r in groups[2]])
    write_jsonl(path / 'validation_eval.jsonl', [
        {k: r[k] for k in ('task_id', 'text', 'test_list', 'test_setup_code')} for r in groups[1]])
    write_json(path / 'manifest.json', {
        **provenance(), 'source': DATA_URL, 'source_sha256': DATA_SHA256,
        'source_revision': DATA_REVISION, 'eval_size': config['eval_size'],
        'seed': config['data_seed'], 'prompt_version': PROMPT_VERSION, 'test_version': TEST_VERSION,
        'ids': {name: [r['task_id'] for r in rows]
                for name, rows in zip(('train', 'valid', 'eval'), groups)},
        'split_sha256': digest([[r['task_id'] for r in rows] for rows in groups]),
        'files': {name: file_hash(path / name) for name in (
            'train.jsonl', 'valid.jsonl', 'eval.jsonl', 'validation_eval.jsonl')},
    })
    print(f'Prepared train/validation/evaluation counts: {[len(rows) for rows in groups]} in {path}')


def dataset_info(path):
    path = Path(path)
    manifest = read_json(path / 'manifest.json')
    if manifest['prompt_version'] != PROMPT_VERSION:
        raise ValueError('Dataset prompt version differs; prepare a fresh dataset.')
    if manifest['source'] != DATA_URL or manifest['source_sha256'] != DATA_SHA256:
        raise ValueError('Dataset source/checksum differs from pinned MBPP.')
    if manifest.get('source_revision', DATA_REVISION) != DATA_REVISION or manifest.get('test_version', TEST_VERSION) != TEST_VERSION:
        raise ValueError('Dataset source revision or test format differs.')
    ids = manifest['ids']
    if manifest.get('eval_size', len(ids['eval'])) != len(ids['eval']) or not ids['eval']:
        raise ValueError('Dataset evaluation size differs from manifest or is empty.')
    if len(ids['eval']) == 500 and set(ids['eval']) != set(range(11, 511)):
        raise ValueError('Full evaluation requires exactly official MBPP test IDs 11–510.')
    all_ids = sum((ids[k] for k in ('train', 'valid', 'eval')), [])
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Dataset splits overlap or contain duplicate IDs.')
    if any(not 601 <= i <= 974 for k in ('train', 'valid') for i in ids[k]):
        raise ValueError('Training/validation contains non-training task IDs.')
    if any(not 11 <= i <= 510 for i in ids['eval']):
        raise ValueError('Evaluation contains non-held-out task IDs.')
    evaluation = read_jsonl(path / 'eval.jsonl')
    if [r['task_id'] for r in evaluation] != ids['eval'] or any(not r['test_list'] for r in evaluation):
        raise ValueError('Evaluation examples differ from the manifest or have no tests.')
    for name in ('train.jsonl', 'valid.jsonl'):
        if len(read_jsonl(path / name)) != len(ids[name.split('.')[0]]):
            raise ValueError('Training/validation row count differs from the manifest.')
    for name in ('train.jsonl', 'valid.jsonl', 'eval.jsonl'):
        if file_hash(path / name) != manifest['files'][name]:
            raise ValueError(f'Dataset changed: {name}; prepare a fresh dataset.')
    if 'validation_eval.jsonl' in manifest['files']:
        validation = read_jsonl(path / 'validation_eval.jsonl')
        if (file_hash(path / 'validation_eval.jsonl') != manifest['files']['validation_eval.jsonl'] or
                [r['task_id'] for r in validation] != ids['valid'] or
                any(set(r) != {'task_id', 'text', 'test_list', 'test_setup_code'} or not r['test_list'] for r in validation)):
            raise ValueError('Validation prompts changed or contain unexpected reference fields.')
        valid_chat = read_jsonl(path / 'valid.jsonl')
        if any(chat['messages'][:-1] != messages(problem) for chat, problem in zip(valid_chat, validation)):
            raise ValueError('Validation scoring prompts differ from the loss-validation prompts.')
    if 'split_sha256' in manifest and manifest['split_sha256'] != digest([ids[k] for k in ('train', 'valid', 'eval')]):
        raise ValueError('Split ID checksum mismatch.')
    return manifest


def prepare_model(args, config):
    require_mlx()
    if Path(args.model).exists():
        raise ValueError('Model output already exists; select a fresh --model path.')
    from huggingface_hub import snapshot_download
    from mlx_lm.convert import convert
    source = snapshot_download(repo_id=config['model_id'], revision=config['model_revision'],
                               cache_dir=str(ROOT / '.cache/huggingface'))
    convert(hf_path=source, mlx_path=args.model, dtype=config['dtype'])
    path = Path(args.model)
    files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*')) if p.is_file()}
    write_json(path / 'experiment_model.json', {
        **provenance(), 'model_id': config['model_id'], 'revision': config['model_revision'],
        'dtype': config['dtype'], 'quantized': False, 'files': files,
    })
    print(f'Converted non-quantized base model to {path}')


def model_info(path, config):
    path = Path(path)
    if not (path / 'experiment_model.json').is_file():
        raise ValueError(f'Model missing at {path}; run prepare-model first.')
    info = read_json(path / 'experiment_model.json')
    if (info['model_id'], info['revision'], info['dtype']) != (
        config['model_id'], config['model_revision'], config['dtype']
    ):
        raise ValueError('Model configuration differs from the prepared model.')
    if not info['files'] or not any(n.endswith('.safetensors') for n in info['files']):
        raise ValueError('Prepared model has no weights.')
    actual_files = {str(p.relative_to(path)) for p in path.rglob('*')
                    if p.is_file() and p.name != 'experiment_model.json'}
    if actual_files != set(info['files']):
        raise ValueError('Model directory contents changed; prepare a fresh model.')
    for name, expected in info['files'].items():
        if file_hash(path / name) != expected:
            raise ValueError(f'Model file changed: {name}.')
    return {k: info[k] for k in ('model_id', 'revision', 'dtype', 'quantized', 'files')}


def adapter_info(path, model):
    if path is None:
        return None
    path = Path(path)
    for name in ('adapters.safetensors', 'adapter_config.json', 'experiment_training.json'):
        if not (path / name).is_file():
            raise ValueError(f'Adapter file missing: {path / name}; select a completed training directory.')
    info = read_json(path / 'experiment_training.json')
    if info['status'] != 'complete' or info['model'] != model:
        raise ValueError('Adapter training incomplete or base model does not match.')
    for name, expected in info['adapter_files'].items():
        if file_hash(path / name) != expected:
            raise ValueError(f'Adapter changed: {name}.')
    return {'path': str(path.resolve()), 'files': info['adapter_files'],
            'training': info}


def load_generator(args, config, model):
    require_mlx()
    adapter = adapter_info(args.adapter, model)
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    loaded, tokenizer = load(args.model, adapter_path=None if adapter is None else args.adapter,
                             tokenizer_config={'trust_remote_code': False})
    settings = config['decoding']
    sampler = make_sampler(temp=settings['temperature'], top_p=settings['top_p'])

    def generate_one(problem):
        mx.random.seed(settings['seed'])
        prompt = tokenizer.apply_chat_template(messages(problem), tokenize=False, add_generation_prompt=True)
        return generate(loaded, tokenizer, prompt=prompt, max_tokens=settings['max_tokens'],
                        sampler=sampler, verbose=False)
    return generate_one, adapter


def infer(args, config):
    model = model_info(args.model, config)
    generate_one, adapter = load_generator(args, config, model)
    problem = {'text': args.task, 'test_list': args.test or []}
    response = generate_one(problem)
    write_json(args.output, {**provenance(), 'model': model, 'adapter': adapter,
                            'decoding': config['decoding'], 'messages': messages(problem), 'response': response})
    print(response)


def train(args, config):
    from scripts.loss_masking import integration_info
    require_mlx()
    variant = getattr(args, 'mask_variant', 'corrected')
    integration = integration_info(variant)
    model = model_info(args.model, config)
    data = dataset_info(args.data)
    if data['seed'] != config['data_seed'] or len(data['ids']['valid']) != config['validation_size']:
        raise ValueError('Training split size/seed differs from configuration; prepare fresh data.')
    resume = adapter_info(args.resume_adapter, model)
    path = Path(args.adapter_output)
    training = dict(config['training'], model=str(Path(args.model).resolve()), train=True,
                    test=False, data=str(Path(args.data).resolve()), adapter_path=str(path.resolve()),
                    resume_adapter_file=None if resume is None else str(Path(args.resume_adapter).resolve() / 'adapters.safetensors'))
    if training['fine_tune_type'] != 'lora':
        raise ValueError('Only LoRA is supported by this experiment.')
    if resume is not None and any(resume['training']['configuration'][key] != training[key] for key in ('lora_parameters', 'num_layers')):
        raise ValueError('Resume LoRA parameters differ from the saved adapter.')
    fresh_directory(path)
    # JSON is a YAML subset accepted by MLX-LM; config captures every experiment parameter.
    write_json(path / 'training_config.json', training)
    metadata = {**provenance(), 'model': model, 'dataset': data, 'configuration': training,
                'loss_mask': integration, 'code': code_state(), 'resume': resume, 'status': 'running'}
    write_json(path / 'experiment_training.json', metadata)
    with (path / 'training.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'scripts.masked_lora', '--mask-variant', variant, '--config',
                                 str(path / 'training_config.json')], stdout=log, stderr=subprocess.STDOUT)
    metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
    metadata['status'] = 'complete' if result.returncode == 0 else 'failed'
    if result.returncode == 0:
        try:
            runtime = read_json(path / 'trainer_runtime.json')
            if (runtime['status'] != 'complete' or runtime['integration'] != integration or
                    runtime['versions'] != metadata['versions'] or runtime['configuration_sha256'] != digest(training) or
                    runtime['trainer_calls'] != 1 or not runtime['train_batches'] or not runtime['validation_batches']):
                raise ValueError('Inconsistent trainer runtime evidence.')
            metadata['runtime'] = runtime
            metadata['adapter_files'] = {n: file_hash(path / n) for n in (
                'adapters.safetensors', 'adapter_config.json', 'training_config.json', 'trainer_runtime.json')}
        except (ValueError, KeyError, OSError) as exc:
            metadata['status'] = 'failed'
            write_json(path / 'experiment_training.json', metadata)
            raise ValueError('Missing or inconsistent trainer runtime evidence/adapter files; training rejected.') from exc
    write_json(path / 'experiment_training.json', metadata)
    if result.returncode:
        raise ValueError(f'Training failed; inspect {path / "training.log"}.')
    print(f'Adapter and training log saved to {path}')


def aggregate(rows):
    if not rows or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Results must be nonempty with unique task IDs.')
    if any(type(r['passed']) is not bool for r in rows):
        raise ValueError('Each problem must have a boolean passed result.')
    passed = sum(r['passed'] for r in rows)
    return {'total': len(rows), 'passed': passed, 'pass_at_1_percent': 100 * passed / len(rows)}


def scoring_protocol(config, image_reference):
    return {'scorer_version': SCORER_VERSION, 'test_version': TEST_VERSION,
            'scoring': {'startup_seconds': 30, **config['scoring']},
            'image': resolve_image(image_reference), 'image_reference': image_reference,
            'scoring_python': platform.python_version(),
            'scoring_harness': {p.name: file_hash(p) for p in (
                Path(__file__), Path(__file__).with_name('evaluation_sandbox.py'))}}


def generation_rows(result):
    return [{k: r[k] for k in ('task_id', 'response', 'solution')} for r in result['problems']]


def validate_generations(result):
    expected = [r['task_id'] for r in result['protocol']['examples']]
    if not expected or len(expected) != len(set(expected)) or [r['task_id'] for r in result['problems']] != expected:
        raise ValueError('Generated task IDs/order differ from the complete protocol.')
    for row in result['problems']:
        try:
            solution = extract_code(row['response'])
        except ValueError:
            solution = ''
        if row['solution'] != solution:
            raise ValueError('Saved solution differs from the extracted response.')
    fingerprint = digest(generation_rows(result))
    if 'generation_status' in result and result['generation_status'] == 'complete' and 'generations_sha256' not in result:
        raise ValueError('Complete saved generations lack a checksum.')
    if result.get('generations_sha256', fingerprint) != fingerprint:
        raise ValueError('Saved generations checksum mismatch.')
    return fingerprint


def score_saved(result, output):
    # All generations are persisted first, so even a startup failure can be rescored.
    result['status'] = 'running'
    result.pop('summary', None)
    result.pop('finished_at', None)
    protocol = result['protocol']
    problems = protocol['examples']
    generations = generation_rows(result)
    # Keep the complete immutable generation set in the file throughout scoring.
    result['problems'] = generations
    write_json(output, result)
    for index, (row, problem) in enumerate(zip(generations, problems)):
        if not row['solution']:
            outcome = {'passed': False, 'status': 'format_error', 'exit_code': None}
        else:
            outcome = score(row['solution'], problem, protocol['image'], protocol['scoring'])
        result['problems'][index] = {**row, **outcome}
        write_json(output, result)
        print(f"{problem['task_id']}: {outcome['status']}", flush=True)
    result.update(status='complete', summary=aggregate(result['problems']),
                  finished_at=datetime.now(timezone.utc).isoformat())
    write_json(output, result)
    print(json.dumps(result['summary'], indent=2))


def evaluate(args, config):
    model = model_info(args.model, config)
    data = dataset_info(args.data)
    split = getattr(args, 'split', 'test')
    filename = 'validation_eval.jsonl' if split == 'validation' else 'eval.jsonl'
    if split == 'validation' and filename not in data['files']:
        raise ValueError('Validation scoring prompts missing; prepare fresh data in a new directory.')
    problems = read_jsonl(Path(args.data) / filename)
    expected_size = config['validation_size'] if split == 'validation' else config['eval_size']
    if len(problems) != expected_size or data['seed'] != config['data_seed']:
        raise ValueError('Evaluation size/seed differs from configuration; prepare fresh data.')
    scoring = scoring_protocol(config, args.image)
    generate_one, adapter = load_generator(args, config, model)
    protocol = {'model': model, 'dataset': data, 'examples': problems,
                'split': split,
                'prompt_version': PROMPT_VERSION, 'decoding': config['decoding'],
                'versions': versions(), 'generation_harness': file_hash(Path(__file__)), **scoring}
    result = {**provenance(), 'protocol': protocol, 'adapter': adapter,
              'status': 'running', 'generation_status': 'running', 'problems': []}
    write_json(args.output, result)
    for problem in problems:
        response = generate_one(problem)
        try:
            solution = extract_code(response)
        except ValueError:
            solution = ''
        result['problems'].append({'task_id': problem['task_id'], 'response': response, 'solution': solution})
        write_json(args.output, result)
        print(f"{problem['task_id']}: generated", flush=True)
    result['generations_sha256'] = validate_generations(result)
    result['generation_status'] = 'complete'
    write_json(args.output, result)
    score_saved(result, args.output)


def rescore(args, config):
    result = read_json(args.input)
    # Legacy completed runs can also be rescored; partial generation is never accepted.
    if result.get('generation_status', result['status']) != 'complete':
        raise ValueError('Rescoring requires a complete saved generation set.')
    fingerprint = validate_generations(result)
    result.update(provenance(), generation_status='complete', generations_sha256=fingerprint,
                  rescored_from={'path': str(Path(args.input).resolve()), 'sha256': file_hash(args.input),
                                 'scoring_protocol': {k: result['protocol'].get(k) for k in (
                                     'scorer_version', 'scoring', 'image', 'image_reference', 'scoring_harness')}})
    result['protocol'].update(scoring_protocol(config, args.image))
    score_saved(result, args.output)


def validate_scored(result):
    if result['status'] != 'complete':
        raise ValueError('Only completed evaluations can be compared.')
    if result.get('generation_status', 'complete') != 'complete':
        raise ValueError('Incomplete generation set.')
    validate_generations(result)
    protocol = result['protocol']
    required = ('model', 'dataset', 'examples', 'prompt_version', 'decoding', 'versions',
                'scorer_version', 'test_version', 'scoring', 'image', 'image_reference',
                'scoring_python', 'scoring_harness')
    if any(key not in protocol for key in required) or protocol['scorer_version'] != SCORER_VERSION:
        raise ValueError('Incomplete or obsolete scoring protocol; rescore saved generations.')
    if (protocol['test_version'] != TEST_VERSION or not protocol['scoring_harness'] or
            not (protocol.get('generation_harness') or protocol.get('harness'))):
        raise ValueError('Incomplete scoring provenance.')
    for key in ('startup_seconds', 'wall_seconds', 'cpu_seconds', 'memory_mb'):
        if type(protocol['scoring'].get(key)) is not int or protocol['scoring'][key] <= 0:
            raise ValueError('Invalid scoring limits.')
    for row in result['problems']:
        if type(row['passed']) is not bool or row['status'] not in ('passed', 'failed', 'timeout', 'format_error'):
            raise ValueError('Incomplete or invalid scoring outcome.')
        if row['passed'] != (row['status'] == 'passed'):
            raise ValueError('Scoring status and pass flag disagree.')
        if (row['status'] == 'passed' and row['exit_code'] != 0 or
                row['status'] == 'failed' and (type(row['exit_code']) is not int or row['exit_code'] == 0) or
                row['status'] in ('timeout', 'format_error') and row['exit_code'] is not None or
                (row['status'] == 'format_error') != (row['solution'] == '')):
            raise ValueError('Invalid scoring exit code or format outcome.')
    summary = aggregate(result['problems'])
    if result.get('summary') != summary:
        raise ValueError('Scoring summary differs from per-task outcomes.')
    return summary


def stable_dataset(dataset):
    return {k: v for k, v in dataset.items() if k not in ('created_at', 'versions')}


def validate_mask_pair(legacy, corrected):
    from scripts.loss_masking import UPSTREAM_HASHES
    records = []
    for result, variant in ((legacy, 'legacy'), (corrected, 'corrected')):
        adapter = result.get('adapter')
        if not adapter or 'training' not in adapter:
            raise ValueError('Mask comparison requires two freshly trained adapters with provenance.')
        training = adapter['training']
        runtime = training.get('runtime', {})
        integration = training.get('loss_mask', {})
        protocol = result['protocol']
        dataset = training['dataset']
        if (training['status'] != 'complete' or training.get('resume') is not None or
                integration.get('variant') != variant or integration.get('upstream_files') != UPSTREAM_HASHES or
                integration.get('implementation') != 'exclusive-target-bound-v1' or
                runtime.get('status') != 'complete' or runtime.get('integration') != integration or
                runtime.get('versions') != training['versions'] or runtime.get('trainer_calls') != 1 or
                not runtime.get('train_batches') or not runtime.get('validation_batches') or
                runtime.get('configuration_sha256') != digest(training['configuration'])):
            raise ValueError('Missing or inconsistent mask variant/trainer runtime provenance.')
        if (protocol.get('split') != 'validation' or len(dataset['ids']['valid']) < 80 or
                [r['task_id'] for r in protocol['examples']] != dataset['ids']['valid'] or
                stable_dataset(dataset) != stable_dataset(protocol['dataset']) or
                training['model'] != protocol['model'] or training['versions'] != protocol['versions']):
            raise ValueError('Mask comparison requires the same enlarged internal validation data/model/versions used in training.')
        ids = dataset['ids']
        if (set(ids['train']) & set(ids['valid']) or
                set(ids['train'] + ids['valid']) != set(range(601, 975)) or
                any(not 11 <= i <= 510 for i in ids['eval']) or
                dataset.get('split_sha256') != digest([ids[k] for k in ('train', 'valid', 'eval')])):
            raise ValueError('Invalid controlled split IDs/checksum.')
        config = dict(training['configuration'])
        if config['iters'] != 100 or config.get('mask_prompt') is not True:
            raise ValueError('Controlled mask comparison requires 100-step prompt-masked training.')
        for key in ('model', 'data', 'adapter_path', 'resume_adapter_file'):
            config.pop(key, None)
        code = training['code']
        if (integration['hook_sha256'] != code['files']['scripts/loss_masking.py'] or
                integration['runner_sha256'] != code['files']['scripts/masked_lora.py']):
            raise ValueError('Mask integration differs from recorded training code.')
        effective = dict(runtime['effective_configuration'])
        for key in ('model', 'data', 'adapter_path', 'resume_adapter_file'):
            effective.pop(key, None)
        records.append({'configuration': config, 'effective_configuration': effective,
                        'model': training['model'], 'dataset': stable_dataset(dataset),
                        'versions': training['versions'], 'code': code,
                        'integration': {k: v for k, v in integration.items() if k != 'variant'},
                        'tokenization': {k: runtime[k] for k in ('train', 'valid', 'tokenizer')}})
    if records[0] != records[1]:
        raise ValueError('A/B training provenance differs beyond the loss-mask variant/output paths.')


def compare_results(baseline, adapted, *, mask_ab=False):
    b, a = validate_scored(baseline), validate_scored(adapted)
    if mask_ab:
        validate_mask_pair(baseline, adapted)
    elif baseline['adapter'] is not None or adapted['adapter'] is None:
        raise ValueError('Comparison requires an adapter-free baseline and an adapted evaluation.')
    # Timestamps in data preparation are irrelevant; the actual file hashes/IDs must match.
    def protocol(result):
        value = dict(result['protocol'])
        value['dataset'] = {k: v for k, v in value['dataset'].items() if k not in ('created_at', 'versions')}
        return value
    if protocol(baseline) != protocol(adapted):
        raise ValueError('Evaluation protocols differ (model, examples, prompts, decoding, scorer, image, or libraries).')
    before, after = baseline['problems'], adapted['problems']
    transitions = {name: [] for name in ('fail_to_pass', 'pass_to_fail', 'unchanged_pass', 'unchanged_fail')}
    for x, y in zip(before, after):
        name = ('unchanged_pass' if x['passed'] else 'unchanged_fail') if x['passed'] == y['passed'] else (
            'pass_to_fail' if x['passed'] else 'fail_to_pass')
        transitions[name].append(x['task_id'])
    return {'baseline': b, 'adapted': a,
            'comparison': 'legacy-to-corrected-mask' if mask_ab else 'base-to-adapter',
            'failure_categories': {name: {status: sum(r['status'] == status for r in result['problems'])
                                         for status in ('failed', 'timeout', 'format_error')}
                                   for name, result in (('baseline', baseline), ('adapted', adapted))},
            'transitions': {k: len(v) for k, v in transitions.items()},
            'changed_task_ids': {k: transitions[k] for k in ('fail_to_pass', 'pass_to_fail')},
            'net_correct_change': a['passed'] - b['passed'],
            'interpretation': ('This mask-only comparison measures the effect on this internal validation split; '
                               'it does not establish generalization or explain the historical test regression.' if mask_ab else
                              'This single paired, test-conditioned MBPP evaluation does not prove general improvement. '
                               'The earlier 50-task sample is a different sample; public MBPP may be present in pretraining.'),
            'difference_percentage_points': 100 * (a['passed'] - b['passed']) / b['total'],
            'outcomes': [{'task_id': x['task_id'], 'baseline_passed': x['passed'],
                          'adapted_passed': y['passed'], 'changed': x['passed'] != y['passed']}
                         for x, y in zip(before, after)]}


def print_comparison(report):
    labels = (('baseline', 'Legacy mask'), ('adapted', 'Corrected mask')) if report.get('comparison') == 'legacy-to-corrected-mask' else (
        ('baseline', 'Base'), ('adapted', 'Adapted'))
    for key, label in labels:
        value = report[key]
        print(f"{label}: {value['passed']}/{value['total']} ({value['pass_at_1_percent']:.2f}%)")
    print(f"Difference: {report['difference_percentage_points']:+.2f} percentage points; "
          f"net correct: {report['net_correct_change']:+d}")
    print(json.dumps({k: report[k] for k in ('transitions', 'changed_task_ids')}, indent=2))
    print(json.dumps(report['failure_categories'], indent=2))
    print(report['interpretation'])


def validate_config(config):
    if not re.fullmatch(r'[0-9a-f]{40}', config['model_revision']):
        raise ValueError('Model revision must be an immutable 40-character commit hash.')
    if config['dtype'] != 'bfloat16':
        raise ValueError('This workflow uses non-quantized bfloat16 weights.')
    for key in ('eval_size', 'validation_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    if config['eval_size'] > 500:
        raise ValueError('eval_size exceeds the official 500-problem test split.')
    decoding, scoring, training = (config[k] for k in ('decoding', 'scoring', 'training'))
    if type(scoring.get('startup_seconds', 30)) is not int or scoring.get('startup_seconds', 30) <= 0:
        raise ValueError('startup_seconds must be a positive integer.')
    if any(type(seed) is not int for seed in (config['data_seed'], decoding['seed'], training['seed'])):
        raise ValueError('All seeds must be integers.')
    lora = training['lora_parameters']
    if type(lora['rank']) is not int or lora['rank'] <= 0 or not math.isfinite(lora['scale']) or lora['scale'] <= 0 or not 0 <= lora['dropout'] < 1:
        raise ValueError('Invalid LoRA rank, scale, or dropout.')
    for section, keys in ((decoding, ('max_tokens',)),
                          (scoring, ('wall_seconds', 'cpu_seconds', 'memory_mb')),
                          (training, ('iters', 'batch_size', 'num_layers', 'max_seq_length',
                                      'steps_per_report', 'steps_per_eval', 'save_every', 'val_batches'))):
        for key in keys:
            if type(section[key]) is not int or section[key] <= 0:
                raise ValueError(f'{key} must be a positive integer.')
    if not math.isfinite(decoding['temperature']) or decoding['temperature'] < 0 or not 0 < decoding['top_p'] <= 1:
        raise ValueError('Invalid temperature or top_p.')
    if training['fine_tune_type'] != 'lora' or training.get('trust_remote_code', False):
        raise ValueError('Only LoRA without remote code is supported.')
    if training.get('mask_prompt') is not True:
        raise ValueError('Completion-only training requires mask_prompt=true.')
    if not math.isfinite(training['learning_rate']) or training['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive and finite.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(CONFIG))
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare-model', 'prepare-data', 'infer', 'train', 'evaluate', 'rescore', 'compare', 'compare-masks'):
        sub = commands.add_parser(name)
        if name in ('prepare-model', 'infer', 'train', 'evaluate'):
            sub.add_argument('--model', default='models/qwen3-4b-2507')
        if name in ('prepare-data', 'train', 'evaluate'):
            sub.add_argument('--data', default='data/mbpp')
        if name in ('infer', 'evaluate'):
            sub.add_argument('--adapter', help='Completed adapter directory; omitted means base model')
        if name in ('infer', 'evaluate', 'rescore', 'compare', 'compare-masks'):
            sub.add_argument('--output', required=True)
        if name == 'infer':
            sub.add_argument('--task', required=True)
            sub.add_argument('--test', action='append', help='Optional task assertion; repeatable')
        if name in ('evaluate', 'rescore'):
            sub.add_argument('--image', default=DEFAULT_IMAGE)
        if name == 'evaluate':
            sub.add_argument('--split', choices=('test', 'validation'), default='test')
        if name == 'rescore':
            sub.add_argument('--input', required=True, help='Saved evaluation with all generations')
        if name == 'train':
            sub.add_argument('--mask-variant', choices=('corrected', 'legacy'), default='corrected',
                             help='legacy is diagnostic only; normal training uses corrected masking')
            sub.add_argument('--adapter-output', default='adapters/mbpp-lora')
            sub.add_argument('--resume-adapter', help='Resume weights from a completed adapter directory')
        if name == 'compare':
            sub.add_argument('--baseline', required=True)
            sub.add_argument('--adapted', required=True)
        if name == 'compare-masks':
            sub.add_argument('--legacy', required=True)
            sub.add_argument('--corrected', required=True)
    args = parser.parse_args(argv)
    try:
        if getattr(args, 'output', None) and Path(args.output).exists():
            raise ValueError(f'{args.output} exists; select a new --output path.')
        config = read_json(args.config)
        validate_config(config)
        if args.command in ('compare', 'compare-masks'):
            if args.command == 'compare-masks':
                args.baseline, args.adapted = args.legacy, args.corrected
            report = compare_results(read_json(args.baseline), read_json(args.adapted),
                                     mask_ab=args.command == 'compare-masks')
            report.update(provenance(), inputs={k: {'path': str(Path(v).resolve()), 'sha256': file_hash(v)}
                                               for k, v in (('baseline', args.baseline), ('adapted', args.adapted))})
            write_json(args.output, report)
            print_comparison(report)
        else:
            globals()[args.command.replace('-', '_')](args, config)
    except (ValueError, TypeError, OSError, KeyError, ImportError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

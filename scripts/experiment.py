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
SCORER_VERSION = 'docker-mbpp-v1'
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
    write_json(path / 'manifest.json', {
        **provenance(), 'source': DATA_URL, 'source_sha256': DATA_SHA256,
        'seed': config['data_seed'], 'prompt_version': PROMPT_VERSION,
        'ids': {name: [r['task_id'] for r in rows]
                for name, rows in zip(('train', 'valid', 'eval'), groups)},
        'files': {name: file_hash(path / name) for name in ('train.jsonl', 'valid.jsonl', 'eval.jsonl')},
    })
    print(f'Prepared train/validation/evaluation counts: {[len(rows) for rows in groups]} in {path}')


def dataset_info(path):
    path = Path(path)
    manifest = read_json(path / 'manifest.json')
    if manifest['prompt_version'] != PROMPT_VERSION:
        raise ValueError('Dataset prompt version differs; prepare a fresh dataset.')
    ids = manifest['ids']
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
    require_mlx()
    model = model_info(args.model, config)
    data = dataset_info(args.data)
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
                'resume': resume, 'status': 'running'}
    write_json(path / 'experiment_training.json', metadata)
    with (path / 'training.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'mlx_lm', 'lora', '--config',
                                 str(path / 'training_config.json')], stdout=log, stderr=subprocess.STDOUT)
    metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
    metadata['status'] = 'complete' if result.returncode == 0 else 'failed'
    if result.returncode == 0:
        metadata['adapter_files'] = {n: file_hash(path / n) for n in ('adapters.safetensors', 'adapter_config.json')}
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


def evaluate(args, config):
    model = model_info(args.model, config)
    data = dataset_info(args.data)
    problems = read_jsonl(Path(args.data) / 'eval.jsonl')
    image = resolve_image(args.image)
    generate_one, adapter = load_generator(args, config, model)
    protocol = {'model': model, 'dataset': data, 'examples': problems,
                'prompt_version': PROMPT_VERSION, 'scorer_version': SCORER_VERSION,
                'decoding': config['decoding'], 'scoring': config['scoring'], 'image': image,
                'image_reference': args.image, 'versions': versions(),
                'harness': {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name('evaluation_sandbox.py'))}}
    result = {**provenance(), 'protocol': protocol, 'adapter': adapter,
              'status': 'running', 'problems': []}
    write_json(args.output, result)
    for problem in problems:
        response = generate_one(problem)
        try:
            solution = extract_code(response)
        except ValueError as exc:
            solution = ''
            outcome = {'passed': False, 'status': 'format_error', 'detail': str(exc), 'exit_code': None}
        else:
            outcome = score(solution, problem, image, config['scoring'])
        result['problems'].append({'task_id': problem['task_id'], 'response': response,
                                   'solution': solution, **outcome})
        write_json(args.output, result)
        print(f"{problem['task_id']}: {outcome['status']}", flush=True)
    result.update(status='complete', summary=aggregate(result['problems']),
                  finished_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output, result)
    print(json.dumps(result['summary'], indent=2))


def compare_results(baseline, adapted):
    if baseline['status'] != 'complete' or adapted['status'] != 'complete':
        raise ValueError('Only completed evaluations can be compared.')
    if baseline['adapter'] is not None or adapted['adapter'] is None:
        raise ValueError('Comparison requires an adapter-free baseline and an adapted evaluation.')
    # Timestamps in data preparation are irrelevant; the actual file hashes/IDs must match.
    def protocol(result):
        value = dict(result['protocol'])
        value['dataset'] = {k: v for k, v in value['dataset'].items() if k not in ('created_at', 'versions')}
        return value
    if protocol(baseline) != protocol(adapted):
        raise ValueError('Evaluation protocols differ (model, examples, prompts, decoding, scorer, image, or libraries).')
    before, after = baseline['problems'], adapted['problems']
    expected = [r['task_id'] for r in baseline['protocol']['examples']]
    if [r['task_id'] for r in before] != expected or [r['task_id'] for r in after] != expected:
        raise ValueError('Evaluated task IDs/order differ from the protocol.')
    b, a = aggregate(before), aggregate(after)
    return {'baseline': b, 'adapted': a,
            'difference_percentage_points': a['pass_at_1_percent'] - b['pass_at_1_percent'],
            'outcomes': [{'task_id': x['task_id'], 'baseline_passed': x['passed'],
                          'adapted_passed': y['passed'], 'changed': x['passed'] != y['passed']}
                         for x, y in zip(before, after)]}


def validate_config(config):
    if not re.fullmatch(r'[0-9a-f]{40}', config['model_revision']):
        raise ValueError('Model revision must be an immutable 40-character commit hash.')
    if config['dtype'] != 'bfloat16':
        raise ValueError('This workflow uses non-quantized bfloat16 weights.')
    for key in ('eval_size', 'validation_size'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    decoding, scoring, training = (config[k] for k in ('decoding', 'scoring', 'training'))
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
    if not math.isfinite(training['learning_rate']) or training['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive and finite.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(CONFIG))
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare-model', 'prepare-data', 'infer', 'train', 'evaluate', 'compare'):
        sub = commands.add_parser(name)
        if name in ('prepare-model', 'infer', 'train', 'evaluate'):
            sub.add_argument('--model', default='models/qwen3-4b-2507')
        if name in ('prepare-data', 'train', 'evaluate'):
            sub.add_argument('--data', default='data/mbpp')
        if name in ('infer', 'evaluate'):
            sub.add_argument('--adapter', help='Completed adapter directory; omitted means base model')
        if name in ('infer', 'evaluate', 'compare'):
            sub.add_argument('--output', required=True)
        if name == 'infer':
            sub.add_argument('--task', required=True)
            sub.add_argument('--test', action='append', help='Optional task assertion; repeatable')
        if name == 'evaluate':
            sub.add_argument('--image', default=DEFAULT_IMAGE)
        if name == 'train':
            sub.add_argument('--adapter-output', default='adapters/mbpp-lora')
            sub.add_argument('--resume-adapter', help='Resume weights from a completed adapter directory')
        if name == 'compare':
            sub.add_argument('--baseline', required=True)
            sub.add_argument('--adapted', required=True)
    args = parser.parse_args(argv)
    try:
        if getattr(args, 'output', None) and Path(args.output).exists():
            raise ValueError(f'{args.output} exists; select a new --output path.')
        config = read_json(args.config)
        validate_config(config)
        if args.command == 'compare':
            report = compare_results(read_json(args.baseline), read_json(args.adapted))
            report.update(provenance(), inputs={k: {'path': str(Path(v).resolve()), 'sha256': file_hash(v)}
                                               for k, v in (('baseline', args.baseline), ('adapted', args.adapted))})
            write_json(args.output, report)
            print(json.dumps(report, indent=2))
        else:
            globals()[args.command.replace('-', '_')](args, config)
    except (ValueError, TypeError, OSError, KeyError, ImportError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()

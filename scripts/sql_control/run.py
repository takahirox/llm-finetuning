"""Copy-and-run MLX SQL experiment with fail-closed local receipts."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from scripts.freight import ROOT, digest, new_directory, now, read, sha, write, require_environment
from scripts.sql_control import data as sql_data
from scripts.sql_control import fixtures
from scripts.sql_control.scoring import aggregate, outcome, paired

CONFIG = ROOT / 'configs/sql-control.json'


def config():
    return read(CONFIG)


def sources():
    paths = list((ROOT / 'scripts/sql_control').glob('*.py'))
    paths += [CONFIG, ROOT / 'configs/sql-reference-audit.json', ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    paths += [ROOT / 'scripts' / name for name in
              ('freight.py', 'experiment.py', 'loss_masking.py', 'masked_lora.py', 'check_environment.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def environment():
    from scripts.freight import environment as freight_environment
    return freight_environment() | {'sqlglot': importlib.metadata.version('sqlglot'),
                                   'sqlite': __import__('sqlite3').sqlite_version}


def require_sql_environment():
    require_environment()
    if importlib.metadata.version('sqlglot') != '26.33.0':
        raise ValueError('Run uv sync --locked --extra sql')


def jsonl(path, records):
    with Path(path).open('x') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=True) + '\n')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def split_source(path):
    cfg = config()
    if sha(path) != cfg['dataset_sha256']:
        raise ValueError('Dataset source checksum mismatch')
    rows = read(path)
    if len(rows) != cfg['dataset_rows']:
        raise ValueError('Wrong source row count')
    splits, excluded = sql_data.select(rows, cfg['split_sizes'], cfg['split_seed'],
                                       cfg['near_duplicate_threshold'], fixtures.cases())
    sql_data.validate_splits(splits, cfg['split_sizes'], cfg['near_duplicate_threshold'], fixtures.cases())
    return splits, excluded


def data_files(path):
    return {p.name: sha(p) for p in sorted(Path(path).glob('*.jsonl'))}


def data_manifest(path, splits, excluded):
    cfg = config()
    return {'format_version': 1, 'dataset_id': cfg['dataset_id'],
            'revision': cfg['dataset_revision'], 'source_sha256': cfg['dataset_sha256'],
            'license': 'CC-BY-4.0', 'attribution': 'b-mc2 (2023), sql-create-context; derived from WikiSQL and Spider',
            'split_seed': cfg['split_seed'], 'split_sizes': cfg['split_sizes'],
            'near_duplicate_threshold': cfg['near_duplicate_threshold'], 'exclusions': excluded,
            'fixture_sha256': fixtures.fingerprint(), 'files': data_files(path),
            'ids': {name: [r['id'] for r in records] for name, records in splits.items()}}


def prepare_data(data, source=None):
    out = new_directory(data)
    target = out / 'source.json'
    cfg = config()
    if source:
        shutil.copyfile(source, target)
    else:
        url = (f"https://huggingface.co/datasets/{cfg['dataset_id']}/resolve/"
               f"{cfg['dataset_revision']}/{cfg['dataset_file']}")
        with urllib.request.urlopen(url, timeout=120) as response, target.open('xb') as stream:
            shutil.copyfileobj(response, stream)
    splits, excluded = split_source(target)
    for name, records in splits.items():
        jsonl(out / f'{name}-records.jsonl', records)
        if name in ('train', 'valid'):
            jsonl(out / f'{name}.jsonl', [{'messages': sql_data.messages(r, True)} for r in records])
    write(out / 'manifest.json', data_manifest(out, splits, excluded))
    print(f'Frozen 5000 train / 100 validation / 500 test, exclusions: {excluded}; {out}')


def data_fingerprint(path):
    path = Path(path)
    splits, excluded = split_source(path / 'source.json')
    for name, records in splits.items():
        if read_jsonl(path / f'{name}-records.jsonl') != records:
            raise ValueError('Saved split differs from deterministic source selection')
        if name in ('train', 'valid') and read_jsonl(path / f'{name}.jsonl') != [
                {'messages': sql_data.messages(r, True)} for r in records]:
            raise ValueError('Training prompt/target mismatch')
    expected = data_manifest(path, splits, excluded)
    if read(path / 'manifest.json') != expected:
        raise ValueError('Data manifest mismatch')
    return expected


def files(path):
    root = Path(path)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '.cache' not in p.relative_to(root).parts and p.name != 'model_manifest.json'}


def prepare_model(model):
    require_sql_environment()
    from huggingface_hub import snapshot_download
    from mlx_lm.convert import convert
    out = Path(model).resolve()
    if out.exists():
        raise ValueError('Model output already exists')
    cfg = config()
    snapshot = snapshot_download(cfg['model_id'], revision=cfg['model_revision'],
                                 cache_dir=ROOT / '.cache/huggingface',
                                 allow_patterns=['*.json', '*.txt', '*.safetensors', 'LICENSE', 'README.md'])
    upstream = files(snapshot)
    convert(hf_path=snapshot, mlx_path=str(out), quantize=True, q_bits=4,
            q_group_size=64, q_mode='affine', dtype='bfloat16', trust_remote_code=False)
    shutil.copyfile(Path(snapshot) / 'LICENSE', out / 'LICENSE')
    shutil.copyfile(Path(snapshot) / 'README.md', out / 'UPSTREAM_README.md')
    write(out / 'model_manifest.json', {'format_version': 1, 'model_id': cfg['model_id'],
          'revision': cfg['model_revision'], 'quantization': cfg['quantization'],
          'conversion_dtype': 'bfloat16',
          'upstream_files': upstream, 'environment': environment(), 'files': files(out)})
    model_fingerprint(out)
    print(f'Pinned model converted once to MLX affine 4-bit: {out}')


def model_fingerprint(path):
    manifest = read(Path(path) / 'model_manifest.json')
    cfg = config()
    quantization = read(Path(path) / 'config.json')['quantization']
    if (manifest['format_version'] != 1 or manifest['model_id'] != cfg['model_id']
            or manifest['revision'] != cfg['model_revision']
            or manifest['quantization'] != cfg['quantization']
            or manifest.get('conversion_dtype') != 'bfloat16'
            or any(quantization.get(k) != v for k, v in cfg['quantization'].items())
            or manifest['environment'] != environment()
            or not manifest['upstream_files'] or manifest['files'] != files(path)
            or not any(name.endswith('.safetensors') for name in manifest['files'])):
        raise ValueError('Model fingerprint/quantization mismatch')
    return manifest


def protocol(model, data):
    return {'model': model_fingerprint(model), 'data': data_fingerprint(data),
            'config': config(), 'sources': sources(), 'environment': environment(),
            'fixtures': fixtures.fingerprint()}


def recipe(model, data, adapter):
    return config()['training'] | {'model': str(Path(model).resolve()),
                                   'data': str(Path(data).resolve()),
                                   'adapter_path': str(Path(adapter).resolve())}


def generation_prompt(tokenizer, record):
    return tokenizer.apply_chat_template(sql_data.messages(record),
                                         add_generation_prompt=True, tokenize=False)


def check_training(training, expected_protocol):
    if (not training or training.get('format_version') != 1 or training.get('status') != 'complete'
            or training['protocol'] != expected_protocol
            or training['recipe'] != config()['training']
            or training['runtime']['status'] != 'complete'
            or training['runtime']['train_batches'] != 5000
            or training['runtime']['train']['count'] != 5000
            or training['runtime']['valid']['count'] != 100
            or training['runtime']['integration']['variant'] != 'corrected'
            or training['runtime']['effective_configuration'] != training['effective_configuration']
            or any(training['effective_configuration'].get(k) != v for k, v in config()['training'].items())
            or set(training['adapter_files']) != {'adapter_config.json', 'adapters.safetensors', 'trainer_runtime.json'}):
        raise ValueError('Incomplete or incompatible fresh training receipt')


def adapter_fingerprint(path, expected_protocol):
    path = Path(path)
    training = read(path / 'training_manifest.json')
    check_training(training, expected_protocol)
    if training['adapter_files'] != {name: sha(path / name) for name in training['adapter_files']}:
        raise ValueError('Adapter files changed')
    if read(path / 'adapter_config.json') != training['effective_configuration']:
        raise ValueError('Effective adapter configuration changed')
    if read(path / 'trainer_runtime.json') != training['runtime']:
        raise ValueError('Trainer runtime differs from training receipt')
    return training


def train(model, data, adapter):
    require_sql_environment()
    frozen = protocol(model, data)
    out = new_directory(adapter)
    training_data = out / 'training-data'
    training_data.mkdir()
    for name in ('train.jsonl', 'valid.jsonl'):
        shutil.copyfile(Path(data) / name, training_data / name)
    effective = recipe(model, training_data, out)
    write(out / 'recipe.json', effective)
    manifest = {'format_version': 1, 'status': 'running', 'started_at': now(),
                'protocol': frozen, 'recipe': config()['training'],
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(out / 'training_manifest.json', manifest)
    started = time.monotonic()
    try:
        # Reuse the source-hash-verified exclusive-bound completion loss hook.
        command = [sys.executable, '-m', 'scripts.masked_lora', '--config', str(out / 'recipe.json')]
        with (out / 'training.log').open('x') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if completed.returncode:
            raise ValueError(f'Training failed; inspect {out / "training.log"}')
        if protocol(model, data) != frozen:
            raise ValueError('Training inputs changed')
        for name in ('train.jsonl', 'valid.jsonl'):
            if sha(training_data / name) != frozen['data']['files'][name]:
                raise ValueError('Trainer input changed')
        effective_saved = read(out / 'adapter_config.json')
        if any(effective_saved.get(k) != v for k, v in effective.items()):
            raise ValueError('Effective training configuration differs')
        manifest.update(status='complete', seconds=time.monotonic() - started,
                        finished_at=now(), effective_configuration=effective_saved,
                        runtime=read(out / 'trainer_runtime.json'),
                        adapter_files={name: sha(out / name) for name in
                                       ('adapter_config.json', 'adapters.safetensors', 'trainer_runtime.json')})
        check_training(manifest, frozen)
    except BaseException:
        manifest['status'] = 'failed'
        raise
    finally:
        write(out / 'training_manifest.json', manifest)
    print(f'Fresh rank-8 training complete: {out}; {manifest["seconds"]:.1f}s')


def evaluate(model, data, output, adapter=None):
    require_sql_environment()
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    frozen = protocol(model, data)
    training = adapter_fingerprint(adapter, frozen) if adapter else None
    out = new_directory(output)
    result = {'format_version': 1, 'status': 'running', 'kind': 'adapted' if adapter else 'base',
              'started_at': now(), 'protocol': frozen, 'training': training, 'cohorts': {}}
    write(out / 'result.json', result)
    started = time.monotonic()
    try:
        net, tokenizer = load(str(Path(model).resolve()),
                              adapter_path=str(Path(adapter).resolve()) if adapter else None)
        for name, records in [('test', read_jsonl(Path(data) / 'test-records.jsonl')),
                              ('sqlite_sanity', fixtures.cases())]:
            rows = []
            with (out / f'{name}-predictions.jsonl').open('x') as stream:
                for record in records:
                    prompt = generation_prompt(tokenizer, record)
                    mx.random.seed(config()['decoding']['seed'])
                    raw = generate(net, tokenizer, prompt=prompt,
                                   max_tokens=config()['decoding']['max_tokens'],
                                   sampler=make_sampler(temp=0.0, top_p=1.0), verbose=False)
                    row = outcome(record, raw, name == 'sqlite_sanity')
                    row['prompt_sha256'] = digest(prompt)
                    rows.append(row)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    if len(rows) % 25 == 0:
                        print(f'{name}: {len(rows)}/{len(records)}', flush=True)
            result['cohorts'][name] = {'records': records, 'rows': rows, 'summary': aggregate(rows)}
        if protocol(model, data) != frozen:
            raise ValueError('Evaluation inputs changed')
        if adapter and adapter_fingerprint(adapter, frozen) != training:
            raise ValueError('Adapter changed during evaluation')
        result.update(status='complete', finished_at=now(), seconds=time.monotonic() - started,
                      peak_memory_bytes=mx.get_peak_memory())
    except BaseException:
        result['status'] = 'failed'
        raise
    finally:
        write(out / 'result.json', result)
    print(json.dumps({name: cohort['summary'] for name, cohort in result['cohorts'].items()}, indent=2))


def validate_result(result, data, expected_protocol, tokenizer):
    cfg = config()
    if result.get('format_version') != 1 or result.get('status') != 'complete':
        raise ValueError('Incomplete or unsupported saved result')
    saved = result['protocol']
    if (saved != expected_protocol or saved['data'] != data_fingerprint(data) or saved['config'] != cfg
            or saved['sources'] != sources() or saved['environment'] != environment()
            or saved['fixtures'] != fixtures.fingerprint()
            or saved['model']['model_id'] != cfg['model_id']
            or saved['model']['revision'] != cfg['model_revision']
            or saved['model']['quantization'] != cfg['quantization']
            or saved['model']['environment'] != saved['environment']):
        raise ValueError('Saved protocol is incompatible with frozen experiment')
    if result['kind'] == 'base':
        if result['training'] is not None:
            raise ValueError('Base receipt cannot have an adapter')
    elif result['kind'] == 'adapted':
        check_training(result['training'], saved)
    else:
        raise ValueError('Unknown result kind')
    cohorts = {'test': read_jsonl(Path(data) / 'test-records.jsonl'), 'sqlite_sanity': fixtures.cases()}
    if set(result['cohorts']) != set(cohorts):
        raise ValueError('Wrong cohorts')
    for name, records in cohorts.items():
        cohort = result['cohorts'][name]
        if cohort['records'] != records or len(cohort['rows']) != len(records):
            raise ValueError('Changed/reordered/incomplete evaluation cohort')
        expected = [outcome(record, row['raw'], name == 'sqlite_sanity') |
                    {'prompt_sha256': digest(generation_prompt(tokenizer, record))}
                    for record, row in zip(records, cohort['rows'], strict=True)]
        if cohort['rows'] != expected or cohort['summary'] != aggregate(expected):
            raise ValueError('Saved scores differ from recomputed scores')


def compare(base, adapted, data, model, output, notes):
    from transformers import AutoTokenizer
    expected_protocol = protocol(model, data)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(model).resolve()), trust_remote_code=False)
    b, a = read(base), read(adapted)
    for result in (b, a):
        validate_result(result, data, expected_protocol, tokenizer)
    if b['kind'] != 'base' or a['kind'] != 'adapted' or b['protocol'] != a['protocol']:
        raise ValueError('Comparison requires identical base/adapted protocols')
    for name in b['cohorts']:
        if [r['prompt_sha256'] for r in b['cohorts'][name]['rows']] != [
                r['prompt_sha256'] for r in a['cohorts'][name]['rows']]:
            raise ValueError('Generation prompts differ')
    out = new_directory(output)
    metrics = {name: paired(b['cohorts'][name]['rows'], a['cohorts'][name]['rows']) for name in b['cohorts']}
    lines = ['# Local SQL base / fresh rank-8 LoRA comparison', '',
             '| Cohort / metric | Base | LoRA | Delta (pp) | Fail→pass | Pass→fail |',
             '| --- | --- | --- | --- | --- | --- |']
    for cohort, scores in metrics.items():
        for metric, values in scores.items():
            left, right = values['base'], values['adapted']
            lines.append(f"| {cohort} / {metric} | {left['correct']}/{left['total']} ({100*left['fraction']:.2f}%) "
                         f"| {right['correct']}/{right['total']} ({100*right['fraction']:.2f}%) "
                         f"| {values['delta_pp']:+.2f} | {values['fail_to_pass']} | {values['pass_to_fail']} |")
    for label, result in [('Base', b), ('LoRA', a)]:
        lines += ['', f"{label} failure categories: " + json.dumps(
            {name: {'text': c['summary']['categories'], 'execution': c['summary']['execution_categories']}
             for name, c in result['cohorts'].items()}, sort_keys=True)]
    lines += ['', 'Third-party single-seed report: exact match 22/500 (4.4%) → 342/500 (68.4%), +64.0 pp; '
              'parse validity 490/500 (98.0%) → 496/500 (99.2%), +1.2 pp. These are not our measurements.',
              '', 'Local differences: MLX affine 4-bit for BOTH runs; CUDA NF4 upstream only for adapter; '
              'independent prompt + Qwen chat template; deduplicated test-first cohort; separate validation; '
              'accumulation 8; pinned parser and strict one-SELECT check. See docs/sql-control.md.',
              '', 'Text normalization folds literal case and whitespace, so can accept unequal values and reject '
              'equivalent SQL. Parse validity does not establish correctness. SQLite scores cover only 12 '
              'original fixtures × three fixed populations, not the 500-case benchmark or general code ability.',
              '', 'Interpretation / root-cause notes: ' + notes, '']
    report = '\n'.join(lines)
    (out / 'comparison.md').write_text(report)
    write(out / 'comparison.json', {'format_version': 1, 'base_sha256': sha(base), 'adapted_sha256': sha(adapted),
          'metrics': metrics, 'notes': notes,
          'paired': {name: [{'id': x['id'], 'base': x, 'adapted': y}
                            for x, y in zip(b['cohorts'][name]['rows'], a['cohorts'][name]['rows'], strict=True)]
                     for name in b['cohorts']}})
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-data')
    prepare.add_argument('--data', type=Path, required=True)
    prepare.add_argument('--source', type=Path, help='Optional already-downloaded pinned JSON; checksum verified')
    commands.add_parser('prepare-model').add_argument('--model', type=Path, required=True)
    training, evaluation = commands.add_parser('train'), commands.add_parser('evaluate')
    for command in (training, evaluation):
        command.add_argument('--model', type=Path, required=True)
        command.add_argument('--data', type=Path, required=True)
    training.add_argument('--adapter', type=Path, required=True)
    evaluation.add_argument('--adapter', type=Path)
    evaluation.add_argument('--output', type=Path, required=True)
    comparison = commands.add_parser('compare')
    for name in ('base', 'adapted', 'data', 'model', 'output'):
        comparison.add_argument('--' + name, type=Path, required=True)
    comparison.add_argument('--notes', required=True, help='Evidence-based interpretation, including possible causes if improvement fails')
    args = vars(parser.parse_args())
    command = args.pop('command')
    try:
        globals()[command.replace('-', '_')](**args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()

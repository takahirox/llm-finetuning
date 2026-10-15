"""Run the pinned LoRA CLI with an explicit, verified trainer loss hook."""

import argparse
from copy import deepcopy
import importlib
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.experiment import digest, read_json, versions, write_json
from scripts.loss_masking import integration_info, make_loss, validate_variant


def audit_dataset(dataset, max_length):
    """Fail before optimization if tokenization/truncation removes all targets."""
    records = []
    for i in range(len(dataset)):
        row = dataset[i]
        messages = row['messages']
        if [m['role'] for m in messages] != ['user', 'assistant']:
            raise ValueError('This integration requires one user prompt and one assistant completion.')
        tokens, offset = dataset.process(row)
        prefix = dataset.tokenizer.apply_chat_template(
            messages[:-1], tools=row.get('tools'), add_generation_prompt=True, return_dict=False)
        if tokens[:offset] != prefix or not 0 < offset < min(len(tokens), max_length):
            raise ValueError(f'Example {i}: prompt boundary mismatch or no completion after truncation; increase max_seq_length.')
        records.append({'tokens_sha256': digest(tokens), 'offset': offset,
                        'length': len(tokens), 'retained_length': min(len(tokens), max_length)})
    return {'count': len(records), 'tokens_and_boundaries_sha256': digest(records),
            'truncated': sum(r['length'] > max_length for r in records),
            'intended_targets': sum(r['retained_length'] - r['offset'] for r in records)}


def run(configuration, variant):
    """Bind the public trainer loss argument at the LoRA dispatch boundary.

    lora.train_model has no loss parameter. Its module-local train reference is
    replaced only for this dedicated process and restored even when training fails.
    The trainer itself passes this same loss to its validation evaluator.
    """
    validate_variant(variant)
    integration = integration_info(variant)
    lora = importlib.import_module('mlx_lm.lora')
    trainer = importlib.import_module('mlx_lm.tuner.trainer')
    if lora.train is not trainer.train:
        raise ValueError('Unexpected LoRA trainer binding; refusing to replace it.')
    args = SimpleNamespace(**(deepcopy(lora.CONFIG_DEFAULTS) | configuration))
    if not args.train or args.test or not args.mask_prompt:
        raise ValueError('Masked runner requires prompt-masked training without test evaluation.')
    runtime_path = Path(args.adapter_path) / 'trainer_runtime.json'
    runtime = {'status': 'running', 'integration': integration, 'versions': versions(),
               'effective_configuration': vars(args),
               'configuration_sha256': digest(configuration), 'trainer_calls': 0,
               'train_batches': 0, 'validation_batches': 0}
    selected_loss = make_loss(variant)
    started = time.monotonic()

    def batches(*positional, **kwargs):
        for batch, lengths in trainer.iterate_batches(*positional, **kwargs):
            # Python-side preflight stays outside the compiled gradient step.
            if any(max(1, offset) >= min(length, batch.shape[1]) for offset, length in lengths.tolist()):
                raise ValueError('Batch contains no intended completion targets after truncation.')
            key = 'train_batches' if kwargs.get('loop', False) else 'validation_batches'
            runtime[key] += 1
            yield batch, lengths

    def train_with_mask(**kwargs):
        runtime['trainer_calls'] += 1
        for name, key in (('train_dataset', 'train'), ('val_dataset', 'valid')):
            # CacheDataset wraps the exact ChatDataset used by this trainer.
            dataset = kwargs[name]._data
            runtime[key] = audit_dataset(dataset, args.max_seq_length)
            tokenizer = dataset.tokenizer
            runtime['tokenizer'] = {'pad_token_id': tokenizer.pad_token_id,
                                    'eos_token_id': tokenizer.eos_token_id,
                                    'collator_padding_id': 0}
        runtime['loss_callable'] = selected_loss.__qualname__
        write_json(runtime_path, runtime)
        return trainer.train(**kwargs, loss=selected_loss, iterate_batches=batches)

    lora.train = train_with_mask
    try:
        lora.run(args)
        if runtime['trainer_calls'] != 1 or not runtime['train_batches'] or not runtime['validation_batches']:
            raise ValueError('Expected trainer hook was not executed for training and validation.')
        runtime['status'] = 'complete'
    except BaseException:
        runtime['status'] = 'failed'
        raise
    finally:
        lora.train = trainer.train
        runtime['elapsed_seconds'] = time.monotonic() - started
        write_json(runtime_path, runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mask-variant', choices=('corrected', 'legacy'), default='corrected')
    args = parser.parse_args()
    run(read_json(args.config), args.mask_variant)


if __name__ == '__main__':
    main()

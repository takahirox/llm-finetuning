"""Controlled split, validation scoring and A/B provenance regression tests."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import experiment as exp
from scripts import loss_masking as masking


def problem(task_id):
    return {'task_id': task_id, 'text': f'task {task_id}', 'test_list': ['assert True'],
            'test_setup_code': '', 'code': f'reference_marker_{task_id}'}


def paired_results():
    config = exp.read_json(exp.ROOT / 'configs/padding-ab.json')
    groups = exp.select_data([problem(i) for i in range(1, 975)], 42, 50, 80)
    ids = {key: [r['task_id'] for r in group] for key, group in zip(('train', 'valid', 'eval'), groups)}
    dataset = {'ids': ids, 'split_sha256': exp.digest([ids[k] for k in ('train', 'valid', 'eval')])}
    examples = [{k: r[k] for k in ('task_id', 'text', 'test_list', 'test_setup_code')} for r in groups[1]]
    model = {'revision': config['model_revision'], 'files': {'model.safetensors': 'fixture'}}
    versions = {'python': 'fixture', 'mlx-lm': '0.32.0'}
    results = []
    for variant in ('legacy', 'corrected'):
        configuration = dict(config['training'], model='/model', data='/data', adapter_path=f'/adapters/{variant}',
                             resume_adapter_file=None, train=True, test=False)
        integration = {'variant': variant, 'implementation': 'exclusive-target-bound-v1',
                       'upstream_files': masking.UPSTREAM_HASHES, 'hook_sha256': 'hook', 'runner_sha256': 'runner'}
        training = {'status': 'complete', 'resume': None, 'configuration': configuration,
                    'model': model, 'dataset': dataset, 'versions': versions, 'loss_mask': integration,
                    'code': {'revision': 'fixture', 'files': {'scripts/loss_masking.py': 'hook', 'scripts/masked_lora.py': 'runner'}},
                    'runtime': {'status': 'complete', 'integration': integration, 'versions': versions,
                                'configuration_sha256': exp.digest(configuration), 'effective_configuration': configuration,
                                'trainer_calls': 1, 'train_batches': 100, 'validation_batches': 20,
                                'train': {'tokens_and_boundaries_sha256': 'train'},
                                'valid': {'tokens_and_boundaries_sha256': 'valid'}, 'tokenizer': {'collator_padding_id': 0}}}
        protocol = {'model': model, 'dataset': dataset, 'examples': examples, 'split': 'validation',
                    'decoding': config['decoding'], 'prompt_version': exp.PROMPT_VERSION,
                    'versions': versions, 'scorer_version': exp.SCORER_VERSION, 'test_version': exp.TEST_VERSION,
                    'image': 'fixture-image', 'image_reference': 'fixture-image', 'scoring_python': 'fixture',
                    'scoring_harness': {'fixture': 'hash'}, 'generation_harness': 'fixture', 'scoring': config['scoring']}
        rows = [{'task_id': r['task_id'], 'response': 'pass', 'solution': 'pass\n',
                 'passed': i % 4 in ((0, 1) if variant == 'legacy' else (0, 2)),
                 'status': 'passed' if i % 4 in ((0, 1) if variant == 'legacy' else (0, 2)) else 'failed',
                 'exit_code': 0 if i % 4 in ((0, 1) if variant == 'legacy' else (0, 2)) else 1} for i, r in enumerate(examples)]
        result = {'status': 'complete', 'generation_status': 'complete', 'adapter': {'training': training},
                  'protocol': protocol, 'problems': rows, 'summary': exp.aggregate(rows)}
        result['generations_sha256'] = exp.digest(exp.generation_rows(result))
        results.append(copy.deepcopy(result))
    return results


class PaddingABTests(unittest.TestCase):
    def test_training_rejects_missing_or_wrong_runtime_even_after_successful_exit(self):
        config = exp.read_json(exp.CONFIG)
        for variant in ('legacy', 'corrected'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / variant
                args = SimpleNamespace(model='model', data='data', adapter_output=output,
                                       resume_adapter=None, mask_variant=variant)

                def pretend_training(command, **kwargs):
                    self.assertEqual(command[4], variant)
                    if variant == 'legacy':
                        exp.write_json(output / 'trainer_runtime.json', {'status': 'complete', 'integration': {'variant': 'corrected'}})
                    return SimpleNamespace(returncode=0)

                with patch.object(exp, 'require_mlx'), patch.object(exp, 'model_info', return_value={}), \
                     patch.object(exp, 'dataset_info', return_value={'seed': 42, 'ids': {'valid': list(range(20))}}), \
                     patch.object(exp, 'code_state', return_value={}), \
                     patch.object(masking, 'integration_info', side_effect=lambda v: {'variant': v}), \
                     patch.object(exp.subprocess, 'run', side_effect=pretend_training):
                    with self.assertRaisesRegex(ValueError, 'training rejected'):
                        exp.train(args, config)
                self.assertEqual(exp.read_json(output / 'experiment_training.json')['status'], 'failed')

    def test_80_split_and_reference_free_validation_prompts(self):
        rows = [problem(i) for i in range(1, 975)]
        config = exp.read_json(exp.ROOT / 'configs/padding-ab.json')
        groups = exp.select_data(rows, 42, 50, 80)
        self.assertEqual([len(g) for g in groups], [294, 80, 50])
        self.assertEqual(groups, exp.select_data(rows[::-1], 42, 50, 80))
        raw = '\n'.join(json.dumps(r) for r in rows).encode()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data'
            with patch.object(exp.urllib.request, 'urlopen', return_value=io.BytesIO(raw)), \
                 patch.object(exp, 'DATA_SHA256', exp.hashlib.sha256(raw).hexdigest()):
                exp.prepare_data(SimpleNamespace(data=path), config)
                manifest = exp.dataset_info(path)
                valid = exp.read_jsonl(path / 'validation_eval.jsonl')
                self.assertEqual([r['task_id'] for r in valid], manifest['ids']['valid'])
                self.assertNotIn('reference_marker', (path / 'validation_eval.jsonl').read_text())
                train = (path / 'train.jsonl').read_text()
                for r in valid:
                    self.assertNotIn(f"reference_marker_{r['task_id']}", train)
                    self.assertNotIn(f"task {r['task_id']}\"", train)
                bad = copy.deepcopy(manifest)
                bad['split_sha256'] = 'incorrect'
                exp.write_json(path / 'manifest.json', bad)
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    exp.dataset_info(path)
                exp.write_json(path / 'manifest.json', manifest)
                valid[0]['code'] = 'leaked reference'
                exp.write_jsonl(path / 'validation_eval.jsonl', valid)
                manifest['files']['validation_eval.jsonl'] = exp.file_hash(path / 'validation_eval.jsonl')
                exp.write_json(path / 'manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'reference fields'):
                    exp.dataset_info(path)

    def test_evaluate_validation_uses_validation_file_and_ids(self):
        legacy, _ = paired_results()
        config = exp.read_json(exp.ROOT / 'configs/padding-ab.json')
        dataset = legacy['protocol']['dataset'] | {'seed': 42, 'files': {'validation_eval.jsonl': 'hash'}}
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(model='model', data='data', image='image', adapter='adapter',
                                   split='validation', output=Path(temporary) / 'validation.json')
            with patch.object(exp, 'model_info', return_value=legacy['protocol']['model']), \
                 patch.object(exp, 'dataset_info', return_value=dataset), \
                 patch.object(exp, 'read_jsonl', return_value=legacy['protocol']['examples']) as read, \
                 patch.object(exp, 'scoring_protocol', return_value={}), \
                 patch.object(exp, 'load_generator', return_value=(lambda _: 'pass', legacy['adapter'])), \
                 patch.object(exp, 'score_saved'):
                exp.evaluate(args, config)
            read.assert_called_once_with(Path('data/validation_eval.jsonl'))
            result = exp.read_json(args.output)
            self.assertEqual(result['protocol']['split'], 'validation')
            self.assertEqual([r['task_id'] for r in result['problems']], dataset['ids']['valid'])

    def test_mask_pair_reports_all_four_transitions_and_failure_categories(self):
        legacy, corrected = paired_results()
        report = exp.compare_results(legacy, corrected, mask_ab=True)
        self.assertEqual(report['transitions'], {'fail_to_pass': 20, 'pass_to_fail': 20,
                                                 'unchanged_pass': 20, 'unchanged_fail': 20})
        self.assertEqual(report['baseline']['passed'], 40)
        self.assertEqual(report['adapted']['passed'], 40)
        self.assertEqual(report['net_correct_change'], 0)
        self.assertEqual(report['difference_percentage_points'], 0)
        self.assertEqual(report['failure_categories']['baseline']['failed'], 40)
        with self.assertRaisesRegex(ValueError, 'adapter-free'):
            exp.compare_results(legacy, corrected)
        for passed in (True, False):
            legacy, corrected = paired_results()
            index = 3 if passed else 0
            corrected['problems'][index].update(passed=passed, status='passed' if passed else 'failed',
                                                 exit_code=0 if passed else 1)
            corrected['summary'] = exp.aggregate(corrected['problems'])
            difference = exp.compare_results(legacy, corrected, mask_ab=True)['difference_percentage_points']
            self.assertEqual(difference, 1.25 if passed else -1.25)

    def test_rejects_confounders_and_inconsistent_runtime_provenance(self):
        for change in ('seed', 'batch_size', 'learning_rate', 'iters', 'resume', 'mask', 'code', 'hook',
                       'runtime', 'runtime-config', 'tokenization', 'effective-config', 'dataset', 'model',
                       'versions', 'split', 'example-order', 'decoding', 'scoring', 'small-validation'):
            legacy, corrected = paired_results()
            training = corrected['adapter']['training']
            runtime = training['runtime']
            if change in ('seed', 'batch_size', 'learning_rate', 'iters'):
                training['configuration'][change] += 1
                runtime['configuration_sha256'] = exp.digest(training['configuration'])
            elif change == 'resume':
                training['resume'] = {'path': 'old'}
            elif change == 'mask':
                training['loss_mask']['variant'] = 'legacy'
            elif change == 'code':
                training['code']['revision'] = 'other'
            elif change == 'hook':
                training['loss_mask']['hook_sha256'] = 'other'
            elif change == 'runtime':
                runtime['train_batches'] = 0
            elif change == 'runtime-config':
                runtime['configuration_sha256'] = 'other'
            elif change == 'tokenization':
                runtime['train']['tokens_and_boundaries_sha256'] = 'other'
            elif change == 'effective-config':
                runtime['effective_configuration']['mask_prompt'] = False
            elif change == 'dataset':
                training['dataset']['ids']['train'][0] = 11
            elif change == 'model':
                training['model']['revision'] = 'other'
            elif change == 'versions':
                training['versions']['mlx-lm'] = 'other'
            elif change == 'split':
                corrected['protocol']['split'] = 'test'
            elif change == 'example-order':
                corrected['protocol']['examples'].reverse()
            elif change == 'small-validation':
                training['dataset']['ids']['valid'] = training['dataset']['ids']['valid'][:20]
            else:
                corrected['protocol'][change]['seed' if change == 'decoding' else 'wall_seconds'] += 1
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                exp.compare_results(legacy, corrected, mask_ab=True)

    def test_variant_and_prompt_masking_selection_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'Unknown loss-mask'):
            masking.make_loss('typo')
        config = exp.read_json(exp.CONFIG)
        config['training']['mask_prompt'] = False
        with self.assertRaisesRegex(ValueError, 'mask_prompt=true'):
            exp.validate_config(config)

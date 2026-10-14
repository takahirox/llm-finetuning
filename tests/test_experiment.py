"""Hardware-independent workflow tests. Generated code is never run on the host."""

import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import experiment as exp
from scripts import evaluation_sandbox as sandbox


def row(task_id):
    return {'task_id': task_id, 'text': f'task marker {task_id}',
            'code': f'def f{task_id}(): return {task_id}',
            'test_list': [f'assert f{task_id}() == {task_id}'], 'test_setup_code': ''}


class DataTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(i) for i in list(range(601, 609)) + list(range(11, 16)) + [1, 511]]

    def test_selection_is_disjoint_reproducible_and_uses_official_ranges(self):
        groups = exp.select_data(self.rows, 42, 3, 2)
        self.assertEqual(groups, exp.select_data(list(reversed(self.rows)), 42, 3, 2))
        sets = [set(r['task_id'] for r in g) for g in groups]
        self.assertEqual([len(g) for g in groups], [6, 2, 3])
        self.assertTrue(sets[0].isdisjoint(sets[1] | sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))
        self.assertTrue(all(i >= 601 for i in sets[0] | sets[1]))
        for rows, count in ((self.rows + [self.rows[0]], 3), (self.rows, 0), (self.rows, 99)):
            with self.assertRaises(ValueError):
                exp.select_data(rows, 42, count, 2)

    def prepare(self, path):
        raw = '\n'.join(json.dumps(r) for r in self.rows).encode()
        with patch.object(exp.urllib.request, 'urlopen', return_value=io.BytesIO(raw)), \
             patch.object(exp, 'DATA_SHA256', exp.hashlib.sha256(raw).hexdigest()):
            exp.prepare_data(SimpleNamespace(data=str(path)),
                             {'data_seed': 42, 'eval_size': 3, 'validation_size': 2})
        return exp.dataset_info(path)

    def test_prepared_files_exclude_heldout_material_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data'
            manifest = self.prepare(path)
            training = (path / 'train.jsonl').read_text() + (path / 'valid.jsonl').read_text()
            for task_id in manifest['ids']['eval']:
                self.assertNotIn(f'task marker {task_id}', training)
                self.assertNotIn(f'def f{task_id}()', training)
                self.assertNotIn(f'assert f{task_id}()', training)
            self.assertTrue(all('code' not in r for r in exp.read_jsonl(path / 'eval.jsonl')))
            self.assertEqual([r['task_id'] for r in exp.read_jsonl(path / 'eval.jsonl')], manifest['ids']['eval'])
            (path / 'train.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                exp.dataset_info(path)

    def test_manifest_rejects_split_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data'
            manifest = self.prepare(path)
            manifest['ids']['eval'][0] = manifest['ids']['train'][0]
            exp.write_json(path / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                exp.dataset_info(path)

    def test_prompt_has_assertions_but_no_reference_solution(self):
        prompt = exp.messages(row(12))
        self.assertEqual(prompt[0]['role'], 'user')
        self.assertIn('assert f12()', prompt[0]['content'])
        self.assertNotIn('def f12()', prompt[0]['content'])

    def test_format_handling(self):
        self.assertEqual(exp.extract_code('x = 1'), 'x = 1\n')
        self.assertEqual(exp.extract_code('Here:\n```python\nx = 1\n```'), 'x = 1\n')
        for value in ('', '```python\nx = 1', '```py\nx=1\n```\n```py\nx=2\n```'):
            with self.assertRaises(ValueError):
                exp.extract_code(value)


class AdapterTests(unittest.TestCase):
    def save_adapter(self, path, model):
        (path / 'adapters.safetensors').write_bytes(b'fixture only')
        exp.write_json(path / 'adapter_config.json', {'fine_tune_type': 'lora'})
        exp.write_json(path / 'experiment_training.json', {
            'status': 'complete', 'model': model,
            'adapter_files': {n: exp.file_hash(path / n) for n in ('adapters.safetensors', 'adapter_config.json')},
        })

    def test_explicit_adapter_or_base_and_integrity(self):
        self.assertIsNone(exp.adapter_info(None, {'revision': 'base'}))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            self.save_adapter(path, {'revision': 'base'})
            self.assertIsNotNone(exp.adapter_info(path, {'revision': 'base'}))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                exp.adapter_info(path, {'revision': 'other'})
            (path / 'adapters.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                exp.adapter_info(path, {'revision': 'base'})
        with self.assertRaisesRegex(ValueError, 'missing'):
            exp.adapter_info('/missing/adapter', {})

    def test_loader_passes_none_for_baseline_and_selected_adapter_for_adapted(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = 'rendered prompt'
        fake_lm = SimpleNamespace(load=Mock(return_value=(object(), tokenizer)),
                                  generate=Mock(return_value='def f(): pass'))
        core = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        sampler = Mock(return_value='sampler')
        modules = {'mlx_lm': fake_lm, 'mlx_lm.sample_utils': SimpleNamespace(make_sampler=sampler),
                   'mlx': SimpleNamespace(core=core), 'mlx.core': core}
        config = exp.read_json(exp.CONFIG)
        with patch.dict('sys.modules', modules), patch.object(exp, 'require_mlx'), \
             patch.object(exp, 'adapter_info', side_effect=[None, {'path': 'adapter'}]):
            for adapter in (None, 'adapters/test'):
                generate, selected = exp.load_generator(SimpleNamespace(model='models/test', adapter=adapter), config, {})
                generate(row(11))
                self.assertEqual(fake_lm.load.call_args.kwargs['adapter_path'], adapter)
                self.assertEqual(selected is None, adapter is None)
                self.assertEqual(fake_lm.generate.call_args.kwargs['max_tokens'], 512)
                self.assertEqual(core.random.seed.call_args.args, (42,))
        tokenizer.apply_chat_template.assert_called_with(exp.messages(row(11)), tokenize=False, add_generation_prompt=True)

    def test_model_missing_and_content_integrity(self):
        config = exp.read_json(exp.CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'prepare-model'):
                exp.model_info(path, config)
            (path / 'model.safetensors').write_bytes(b'fixture')
            exp.write_json(path / 'experiment_model.json', {
                'model_id': config['model_id'], 'revision': config['model_revision'], 'dtype': config['dtype'],
                'quantized': False, 'files': {'model.safetensors': exp.file_hash(path / 'model.safetensors')},
            })
            exp.model_info(path, config)
            (path / 'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                exp.model_info(path, config)

    def test_unsupported_environment_fails_before_native_import(self):
        for system, machine, mac_version in (('Linux', 'arm64', ''), ('Darwin', 'x86_64', '14.0'),
                                             ('Darwin', 'arm64', '13.0')):
            with patch.object(exp.platform, 'system', return_value=system), \
                 patch.object(exp.platform, 'machine', return_value=machine), \
                 patch.object(exp.platform, 'mac_ver', return_value=(mac_version, (), '')), \
                 patch.object(exp.sys, 'version_info', (3, 12, 13)):
                with self.assertRaisesRegex(ValueError, 'check_environment'):
                    exp.require_mlx()

    def test_training_invocation_and_metadata(self):
        config = exp.read_json(exp.CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'adapter'
            args = SimpleNamespace(model='models/base', data='data/mbpp', resume_adapter=None,
                                   adapter_output=str(output))

            def train(command, **kwargs):
                self.assertEqual(command[1:4], ['-m', 'mlx_lm', 'lora'])
                actual = exp.read_json(command[-1])
                self.assertTrue(actual['train'])
                self.assertFalse(actual['test'])
                self.assertEqual(actual['fine_tune_type'], 'lora')
                self.assertEqual(actual['data'], str(Path('data/mbpp').resolve()))
                (output / 'adapters.safetensors').write_bytes(b'fixture')
                exp.write_json(output / 'adapter_config.json', actual)
                return SimpleNamespace(returncode=0)

            with patch.object(exp, 'require_mlx'), patch.object(exp, 'model_info', return_value={'revision': 'base'}), \
                 patch.object(exp, 'dataset_info', return_value={'ids': {'train': [601]}}), \
                 patch.object(exp.subprocess, 'run', side_effect=train):
                exp.train(args, config)
            info = exp.adapter_info(output, {'revision': 'base'})
            self.assertEqual(info['training']['status'], 'complete')


class SandboxTests(unittest.TestCase):
    settings = {'wall_seconds': 5, 'cpu_seconds': 2, 'memory_mb': 128}

    def test_scoring_pass_failure_and_docker_error(self):
        for exit_code in (0, 1, 137, 125, 126, 127):
            with self.subTest(exit_code=exit_code), patch.object(sandbox.subprocess, 'run') as run:
                run.side_effect = [SimpleNamespace(returncode=exit_code), SimpleNamespace(returncode=0)]
                if exit_code in (125, 126, 127):
                    with self.assertRaisesRegex(ValueError, 'Docker execution failed'):
                        sandbox.score('solution', row(11), 'image', self.settings)
                else:
                    outcome = sandbox.score('solution', row(11), 'image', self.settings)
                    self.assertEqual(outcome['passed'], exit_code == 0)
                command = run.call_args_list[0].args[0]
                self.assertIn('--read-only', command)
                self.assertIn('--network', command)
                self.assertIn('none', command)
                self.assertIn('--memory', command)
                self.assertIn('--pids-limit', command)
                self.assertNotIn('-v', command)
                payload = json.loads(run.call_args_list[0].kwargs['input'])
                self.assertEqual(payload['tests'], row(11)['test_list'])
                self.assertEqual(run.call_args_list[-1].args[0][:3], ['docker', 'rm', '-f'])

    def test_timeout_kills_container_and_counts_failure(self):
        with patch.object(sandbox.subprocess, 'run') as run:
            run.side_effect = [subprocess.TimeoutExpired('docker', 5), SimpleNamespace(returncode=0)]
            self.assertEqual(sandbox.score('while True: pass', row(11), 'image', self.settings)['status'], 'timeout')
            self.assertEqual(run.call_count, 2)

    def test_missing_docker_and_cleanup_error_abort(self):
        with patch.object(sandbox.subprocess, 'run', side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, 'Docker/image unavailable'):
                sandbox.resolve_image('image')
        with patch.object(sandbox.subprocess, 'run') as run:
            run.side_effect = [SimpleNamespace(returncode=0), subprocess.TimeoutExpired('docker', 15)]
            with self.assertRaisesRegex(ValueError, 'clean up'):
                sandbox.score('pass', row(11), 'image', self.settings)


class EvaluationTests(unittest.TestCase):
    def test_persists_each_solution_and_compares_same_examples(self):
        config = exp.read_json(exp.CONFIG)
        problems = [row(11), row(12)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evaluation.json'
            args = SimpleNamespace(model='model', data='data', image='image', adapter=None, output=output)
            with patch.object(exp, 'model_info', return_value={'revision': 'base'}), \
                 patch.object(exp, 'dataset_info', return_value={'ids': {'eval': [11, 12]}}), \
                 patch.object(exp, 'read_jsonl', return_value=problems), \
                 patch.object(exp, 'resolve_image', return_value='sha256:fixture'), \
                 patch.object(exp, 'load_generator', return_value=(Mock(side_effect=['def f11(): return 11', '```python']), None)), \
                 patch.object(exp, 'score', return_value={'passed': True, 'status': 'passed', 'exit_code': 0}) as score:
                exp.evaluate(args, config)
            result = exp.read_json(output)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['summary']['pass_at_1_percent'], 50)
            self.assertIsNone(result['adapter'])
            self.assertEqual([r['task_id'] for r in result['problems']], [11, 12])
            self.assertEqual(result['problems'][1]['status'], 'format_error')
            self.assertEqual(score.call_count, 1)
            self.assertEqual(score.call_args.args[1], problems[0])
            self.assertEqual(result['protocol']['decoding'], config['decoding'])
            self.assertEqual(result['protocol']['examples'], problems)

    def test_infrastructure_failure_leaves_incomplete_evaluation(self):
        config = exp.read_json(exp.CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evaluation.json'
            args = SimpleNamespace(model='model', data='data', image='image', adapter=None, output=output)
            with patch.object(exp, 'model_info', return_value={}), \
                 patch.object(exp, 'dataset_info', return_value={}), \
                 patch.object(exp, 'read_jsonl', return_value=[row(11)]), \
                 patch.object(exp, 'resolve_image', return_value='image'), \
                 patch.object(exp, 'load_generator', return_value=(Mock(return_value='pass'), None)), \
                 patch.object(exp, 'score', side_effect=ValueError('Docker execution failed')):
                with self.assertRaises(ValueError):
                    exp.evaluate(args, config)
            self.assertEqual(exp.read_json(output)['status'], 'running')


class ComparisonTests(unittest.TestCase):
    def results(self):
        baseline = {'status': 'complete', 'adapter': None,
                    'protocol': {'dataset': {'ids': [11, 12], 'created_at': 'yesterday'},
                                 'examples': [{'task_id': 11}, {'task_id': 12}], 'decoding': {'seed': 42}},
                    'problems': [{'task_id': 11, 'passed': True}, {'task_id': 12, 'passed': False}]}
        adapted = copy.deepcopy(baseline)
        adapted['adapter'] = {'path': 'adapter'}
        adapted['problems'][1]['passed'] = True
        return baseline, adapted

    def test_scores_changes_and_negative_difference(self):
        baseline, adapted = self.results()
        adapted['protocol']['dataset']['created_at'] = 'today'
        report = exp.compare_results(baseline, adapted)
        self.assertEqual(report['baseline']['pass_at_1_percent'], 50)
        self.assertEqual(report['adapted']['pass_at_1_percent'], 100)
        self.assertEqual(report['difference_percentage_points'], 50)
        self.assertEqual([r['changed'] for r in report['outcomes']], [False, True])
        adapted['problems'][0]['passed'] = False
        adapted['problems'][1]['passed'] = False
        self.assertEqual(exp.compare_results(baseline, adapted)['difference_percentage_points'], -50)

    def test_mismatched_or_incomplete_evaluations_are_rejected(self):
        for key, value in (('decoding', {'seed': 1}), ('dataset', {'ids': [13]})):
            baseline, adapted = self.results()
            adapted['protocol'][key] = value
            with self.assertRaisesRegex(ValueError, 'protocols differ'):
                exp.compare_results(baseline, adapted)
        baseline, adapted = self.results()
        adapted['problems'].reverse()
        with self.assertRaisesRegex(ValueError, 'IDs/order'):
            exp.compare_results(baseline, adapted)
        adapted['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'completed'):
            exp.compare_results(baseline, adapted)
        for rows in ([], [{'task_id': 1, 'passed': True}] * 2, [{'task_id': 1, 'passed': 'false'}]):
            with self.assertRaises(ValueError):
                exp.aggregate(rows)

    def test_requires_real_baseline_and_adapter_selection(self):
        baseline, adapted = self.results()
        baseline['adapter'] = {'path': 'oops'}
        with self.assertRaisesRegex(ValueError, 'adapter-free'):
            exp.compare_results(baseline, adapted)

    def test_default_config_and_rejects_unbounded_or_full_training(self):
        config = exp.read_json(exp.CONFIG)
        exp.validate_config(config)
        for section, key, value in (('decoding', 'max_tokens', -1), ('scoring', 'wall_seconds', 0),
                                    ('training', 'fine_tune_type', 'full')):
            modified = copy.deepcopy(config)
            modified[section][key] = value
            with self.assertRaises(ValueError):
                exp.validate_config(modified)


if __name__ == '__main__':
    unittest.main()

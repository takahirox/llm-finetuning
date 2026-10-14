"""Hardware-independent workflow tests. Generated code is never run on the host."""

import copy
import io
import json
import os
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
        manifest = exp.read_json(path / 'manifest.json')
        manifest['source_sha256'] = exp.DATA_SHA256  # Fixture source stands in for pinned download.
        exp.write_json(path / 'manifest.json', manifest)
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

    def test_full_official_split_has_exactly_500_ids_once_and_seeded_order(self):
        self.rows = [row(i) for i in range(1, 975)]
        groups = exp.select_data(self.rows, 42, 500, 20)
        self.assertEqual([len(g) for g in groups], [354, 20, 500])
        ids = [r['task_id'] for r in groups[2]]
        self.assertEqual(sorted(ids), list(range(11, 511)))
        self.assertEqual(len(set(ids)), 500)
        self.assertEqual(ids[:50], [r['task_id'] for r in exp.select_data(self.rows, 42, 50, 20)[2]])
        self.assertEqual(groups, exp.select_data(list(reversed(self.rows)), 42, 500, 20))
        with self.assertRaises(ValueError):
            exp.select_data([r for r in self.rows if r['task_id'] != 11], 42, 500, 20)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'full'
            raw = '\n'.join(json.dumps(r) for r in self.rows).encode()
            with patch.object(exp.urllib.request, 'urlopen', return_value=io.BytesIO(raw)), \
                 patch.object(exp, 'DATA_SHA256', exp.hashlib.sha256(raw).hexdigest()):
                exp.prepare_data(SimpleNamespace(data=path), {'data_seed': 42, 'eval_size': 500, 'validation_size': 20})
                manifest = exp.dataset_info(path)
            self.assertEqual(sorted(manifest['ids']['eval']), list(range(11, 511)))
            self.assertEqual(manifest['source_revision'], exp.DATA_REVISION)
            self.assertEqual(manifest['test_version'], exp.TEST_VERSION)
            self.assertEqual(manifest['seed'], 42)
            self.assertTrue(set(manifest['ids']['eval']).isdisjoint(manifest['ids']['train'] + manifest['ids']['valid']))

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
    settings = {'startup_seconds': 30, 'wall_seconds': 5, 'cpu_seconds': 2, 'memory_mb': 128}

    def process(self, code=0):
        process = Mock(returncode=code)
        process.poll.return_value = code
        return process

    def test_startup_is_separate_from_execution_and_limits_are_preserved(self):
        process = self.process()
        with patch.object(sandbox.subprocess, 'Popen', return_value=process) as launch, \
             patch.object(sandbox, 'await_ready') as ready, \
             patch.object(sandbox, 'container_state', return_value={'Status': 'exited', 'ExitCode': 0}), \
             patch.object(sandbox.time, 'monotonic', side_effect=[0, 25, 26]), \
             patch.object(sandbox, 'remove_container') as cleanup:
            outcome = sandbox.score('pass', row(11), 'image', self.settings)
        self.assertTrue(outcome['passed'])
        self.assertEqual(outcome['startup_seconds'], 25)
        self.assertEqual(outcome['execution_wait_seconds'], 1)
        self.assertEqual(ready.call_args.args[:2], (process, 30))
        self.assertEqual(process.communicate.call_args.kwargs['timeout'], 5)
        payload = json.loads(ready.call_args.args[2])
        self.assertEqual(payload['tests'], row(11)['test_list'])
        command = launch.call_args.args[0]
        for flag in ('--read-only', '--network', 'none', '--memory', '--pids-limit', '--cap-drop', 'ALL'):
            self.assertIn(flag, command)
        self.assertNotIn('-v', command)
        cleanup.assert_called_once()
        process.wait.assert_called_once_with(timeout=15)

    def test_pass_failure_timeout_and_infrastructure_errors(self):
        for code in (0, 1, 124, 137, 125, 126, 127):
            process = self.process(code)
            with self.subTest(code=code), patch.object(sandbox.subprocess, 'Popen', return_value=process), \
                 patch.object(sandbox, 'await_ready'), patch.object(sandbox, 'remove_container') as cleanup, \
                 patch.object(sandbox, 'container_state', return_value={'Status': 'exited', 'ExitCode': code}):
                if code in (125, 126, 127):
                    with self.assertRaisesRegex(ValueError, 'Docker execution failed'):
                        sandbox.score('pass', row(11), 'image', self.settings)
                else:
                    self.assertEqual(sandbox.score('pass', row(11), 'image', self.settings)['passed'], code == 0)
                cleanup.assert_called_once()
        process = self.process()
        process.poll.return_value = None
        process.communicate.side_effect = subprocess.TimeoutExpired('docker', 5)
        with patch.object(sandbox.subprocess, 'Popen', return_value=process), \
             patch.object(sandbox, 'await_ready'), patch.object(sandbox, 'remove_container') as cleanup, \
             patch.object(sandbox, 'container_state', return_value={'Status': 'running'}):
            self.assertEqual(sandbox.score('pass', row(11), 'image', self.settings)['status'], 'timeout')
        cleanup.assert_called_once()
        process.kill.assert_called_once()

    def test_delayed_cli_exit_does_not_turn_completed_code_into_timeout(self):
        for code, status in ((0, 'passed'), (1, 'failed'), (124, 'timeout')):
            process = self.process()
            process.communicate.side_effect = subprocess.TimeoutExpired('docker', 5)
            with patch.object(sandbox.subprocess, 'Popen', return_value=process), \
                 patch.object(sandbox, 'await_ready'), patch.object(sandbox, 'remove_container'), \
                 patch.object(sandbox, 'container_state', return_value={'Status': 'exited', 'ExitCode': code}):
                self.assertEqual(sandbox.score('pass', row(11), 'image', self.settings)['status'], status)
        self.assertIn("signal.setitimer(signal.ITIMER_REAL, payload['wall_seconds'])", sandbox.RUNNER)

    def test_startup_failure_never_enters_execution_and_cleans_up(self):
        process = self.process()
        with patch.object(sandbox.subprocess, 'Popen', return_value=process), \
             patch.object(sandbox, 'await_ready', side_effect=ValueError('startup timeout')), \
             patch.object(sandbox, 'remove_container') as cleanup:
            with self.assertRaisesRegex(ValueError, 'startup timeout'):
                sandbox.score('pass', row(11), 'image', self.settings)
        process.communicate.assert_not_called()
        cleanup.assert_called_once()

    def test_readiness_handshake_eof_partial_and_deadline(self):
        for data, error in ((sandbox.READY, None), (b'', 'before readiness'), (b'bad', 'before readiness')):
            read_fd, write_fd = os.pipe()
            input_read, input_write = os.pipe()
            with os.fdopen(read_fd, 'rb', buffering=0) as stream, os.fdopen(input_write, 'wb', buffering=0) as stdin:
                os.write(write_fd, data)
                os.close(write_fd)
                process = SimpleNamespace(stdout=stream, stdin=stdin)
                try:
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            sandbox.await_ready(process, 1, b'{}')
                    else:
                        sandbox.await_ready(process, 1, b'{}')
                        self.assertEqual(os.read(input_read, 3), b'{}\n')
                finally:
                    os.close(input_read)
        with patch.object(sandbox.os, 'set_blocking'), \
             patch.object(sandbox.select, 'select', return_value=([], [], [])):
            with self.assertRaisesRegex(ValueError, 'startup/readiness timeout'):
                sandbox.await_ready(SimpleNamespace(stdout=Mock(), stdin=Mock()), 30, b'{}')
        with patch.object(sandbox.os, 'set_blocking'), \
             patch.object(sandbox.select, 'select', side_effect=[([], [1], []), ([1], [], []), ([1], [], [])]), \
             patch.object(sandbox.os, 'write', return_value=3), \
             patch.object(sandbox.os, 'read', side_effect=[b'MBPP_', b'READY\n']):
            sandbox.await_ready(SimpleNamespace(stdout=Mock(), stdin=Mock()), 30, b'{}')

    def test_daemon_errors_and_exit_mismatch_fail_closed(self):
        for state in ({'Status': 'running'}, {'Status': 'exited', 'ExitCode': 0}):
            with patch.object(sandbox.subprocess, 'Popen', return_value=self.process(1)), \
                 patch.object(sandbox, 'await_ready'), patch.object(sandbox, 'remove_container'), \
                 patch.object(sandbox, 'container_state', return_value=state):
                with self.assertRaisesRegex(ValueError, 'exit/state mismatch'):
                    sandbox.score('pass', row(11), 'image', self.settings)
        for result in (subprocess.TimeoutExpired('docker', 15), OSError(),
                       SimpleNamespace(stdout='{}'), SimpleNamespace(stdout='not JSON'),
                       SimpleNamespace(stdout='{"Status":"exited","Error":"daemon failure"}')):
            with patch.object(sandbox.subprocess, 'run', side_effect=[result]):
                with self.assertRaisesRegex(ValueError, 'state unavailable'):
                    sandbox.container_state('name')
        process = self.process()
        process.communicate.side_effect = subprocess.TimeoutExpired('docker', 5)
        with patch.object(sandbox.subprocess, 'Popen', return_value=process), \
             patch.object(sandbox, 'await_ready'), patch.object(sandbox, 'remove_container') as cleanup, \
             patch.object(sandbox, 'container_state', side_effect=ValueError('state unavailable')):
            with self.assertRaisesRegex(ValueError, 'state unavailable'):
                sandbox.score('pass', row(11), 'image', self.settings)
        cleanup.assert_called_once()

    def test_cleanup_requires_confirmed_removal(self):
        for check, success in ((SimpleNamespace(returncode=0, stdout=''), True),
                               (SimpleNamespace(returncode=0, stdout='container'), False),
                               (SimpleNamespace(returncode=1, stdout=''), False),
                               (subprocess.TimeoutExpired('docker', 15), False), (OSError(), False)):
            with patch.object(sandbox.subprocess, 'run', side_effect=[SimpleNamespace(returncode=1), check]):
                if success:
                    sandbox.remove_container('mbpp-test')
                else:
                    with self.assertRaisesRegex(ValueError, 'clean up.*docker rm -f'):
                        sandbox.remove_container('mbpp-test')
        with patch.object(sandbox.subprocess, 'Popen', return_value=self.process()), \
             patch.object(sandbox, 'await_ready'), \
             patch.object(sandbox, 'container_state', return_value={'Status': 'exited', 'ExitCode': 0}), \
             patch.object(sandbox, 'remove_container', side_effect=ValueError('cleanup failed')):
            with self.assertRaisesRegex(ValueError, 'cleanup failed'):
                sandbox.score('pass', row(11), 'image', self.settings)

    def test_missing_docker_aborts(self):
        with patch.object(sandbox.subprocess, 'run', side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, 'Docker/image unavailable'):
                sandbox.resolve_image('image')


class EvaluationTests(unittest.TestCase):
    def test_persists_each_solution_and_compares_same_examples(self):
        config = exp.read_json(exp.CONFIG)
        config['eval_size'] = 2
        problems = [row(11), row(12)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evaluation.json'
            args = SimpleNamespace(model='model', data='data', image='image', adapter=None, output=output)
            with patch.object(exp, 'model_info', return_value={'revision': 'base'}), \
                 patch.object(exp, 'dataset_info', return_value={'ids': {'eval': [11, 12]}, 'seed': 42}), \
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
        config['eval_size'] = 1
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'evaluation.json'
            args = SimpleNamespace(model='model', data='data', image='image', adapter=None, output=output)
            with patch.object(exp, 'model_info', return_value={}), \
                 patch.object(exp, 'dataset_info', return_value={'seed': 42}), \
                 patch.object(exp, 'read_jsonl', return_value=[row(11)]), \
                 patch.object(exp, 'resolve_image', return_value='image'), \
                 patch.object(exp, 'load_generator', return_value=(Mock(return_value='pass'), None)), \
                 patch.object(exp, 'score', side_effect=ValueError('Docker execution failed')):
                with self.assertRaises(ValueError):
                    exp.evaluate(args, config)
            self.assertEqual(exp.read_json(output)['status'], 'running')

    def test_rescore_reuses_complete_generations_after_scoring_failure(self):
        config = exp.read_json(exp.CONFIG)
        config['eval_size'] = 2
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / 'original.json'
            rescored = Path(temporary) / 'rescored.json'
            args = SimpleNamespace(model='model', data='data', image='image', adapter=None, output=original)
            generator = Mock(side_effect=['def f11(): return 11', '```python'])
            with patch.object(exp, 'model_info', return_value={}), \
                 patch.object(exp, 'dataset_info', return_value={'seed': 42}), \
                 patch.object(exp, 'read_jsonl', return_value=[row(11), row(12)]), \
                 patch.object(exp, 'resolve_image', return_value='image'), \
                 patch.object(exp, 'load_generator', return_value=(generator, None)), \
                 patch.object(exp, 'score', side_effect=ValueError('startup failure')):
                with self.assertRaises(ValueError):
                    exp.evaluate(args, config)
            saved = exp.read_json(original)
            self.assertEqual(saved['generation_status'], 'complete')
            self.assertEqual(len(saved['problems']), 2)
            original_hash = exp.file_hash(original)
            with patch.object(exp, 'resolve_image', return_value='image'), \
                 patch.object(exp, 'load_generator', side_effect=AssertionError('Must not regenerate')), \
                 patch.object(exp, 'score', return_value={'passed': True, 'status': 'passed', 'exit_code': 0}) as scoring:
                exp.rescore(SimpleNamespace(input=original, output=rescored, image='image'), config)
            result = exp.read_json(rescored)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(exp.generation_rows(result), exp.generation_rows(saved))
            self.assertEqual(result['generations_sha256'], saved['generations_sha256'])
            self.assertEqual(result['rescored_from']['sha256'], original_hash)
            self.assertEqual(exp.file_hash(original), original_hash)
            self.assertEqual(scoring.call_count, 1)  # Invalid formatting stays invalid.
            self.assertEqual(result['protocol']['generation_harness'], saved['protocol']['generation_harness'])
            exp.validate_scored(result)

    def test_rescore_rejects_partial_or_changed_code_and_preserves_generation_provenance(self):
        baseline, _ = ComparisonTests().results()
        baseline['generations_sha256'] = exp.digest(exp.generation_rows(baseline))
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / 'original.json'
            output = Path(temporary) / 'rescored.json'
            args = SimpleNamespace(input=original, output=output, image='new-image')
            for change in ('partial', 'solution', 'response', 'checksum'):
                invalid = copy.deepcopy(baseline)
                if change == 'partial':
                    invalid['generation_status'] = 'running'
                elif change == 'solution':
                    invalid['problems'][0]['solution'] = 'changed'
                elif change == 'response':
                    invalid['problems'][0].update(response='changed', solution='changed\n')
                else:
                    invalid.pop('generations_sha256')
                exp.write_json(original, invalid)
                with self.assertRaises(ValueError):
                    exp.rescore(args, exp.read_json(exp.CONFIG))
                self.assertFalse(output.exists())
            legacy = copy.deepcopy(baseline)
            legacy.pop('generation_status')
            legacy.pop('generations_sha256')
            legacy['protocol']['scorer_version'] = 'docker-mbpp-v1'
            legacy['protocol']['harness'] = {'experiment.py': legacy['protocol'].pop('generation_harness')}
            exp.write_json(original, legacy)
            config = exp.read_json(exp.CONFIG)
            config['scoring']['wall_seconds'] = 20
            with patch.object(exp, 'resolve_image', return_value='new-image-id'), \
                 patch.object(exp, 'score', return_value={'passed': False, 'status': 'failed', 'exit_code': 1}):
                exp.rescore(args, config)
            result = exp.read_json(output)
            for key in ('model', 'dataset', 'examples', 'prompt_version', 'decoding', 'versions'):
                self.assertEqual(result['protocol'][key], baseline['protocol'][key])
            self.assertEqual(result['protocol']['scoring']['wall_seconds'], 20)
            self.assertEqual(result['protocol']['scorer_version'], exp.SCORER_VERSION)
            self.assertEqual(result['protocol']['image'], 'new-image-id')

    def test_configuration_must_match_prepared_evaluation_size_and_seed(self):
        config = exp.read_json(exp.CONFIG)
        args = SimpleNamespace(model='model', data='data')
        with patch.object(exp, 'model_info', return_value={}), \
             patch.object(exp, 'dataset_info', return_value={'seed': 42}), \
             patch.object(exp, 'read_jsonl', return_value=[row(11)]), \
             patch.object(exp, 'load_generator') as load:
            with self.assertRaisesRegex(ValueError, 'size/seed'):
                exp.evaluate(args, config)
            load.assert_not_called()



class ComparisonTests(unittest.TestCase):
    def results(self):
        baseline = {'status': 'complete', 'generation_status': 'complete', 'adapter': None,
                    'protocol': {'model': {'revision': 'base'}, 'dataset': {'ids': [11, 12], 'created_at': 'yesterday'},
                                 'examples': [row(11), row(12)], 'decoding': {'seed': 42},
                                 'prompt_version': exp.PROMPT_VERSION, 'versions': {'mlx': 'fixture'},
                                 'scorer_version': exp.SCORER_VERSION, 'test_version': exp.TEST_VERSION,
                                 'image': 'sha256:fixture', 'image_reference': 'image', 'scoring_python': 'fixture',
                                 'scoring_harness': {'sandbox.py': 'hash'}, 'generation_harness': 'generation-hash',
                                 'scoring': {'startup_seconds': 30, 'wall_seconds': 10, 'cpu_seconds': 3, 'memory_mb': 256}},
                    'problems': [{'task_id': 11, 'response': 'pass', 'solution': 'pass\n',
                                  'passed': True, 'status': 'passed', 'exit_code': 0},
                                 {'task_id': 12, 'response': 'pass', 'solution': 'pass\n',
                                  'passed': False, 'status': 'failed', 'exit_code': 1}]}
        baseline['generations_sha256'] = exp.digest(exp.generation_rows(baseline))
        baseline['summary'] = exp.aggregate(baseline['problems'])
        adapted = copy.deepcopy(baseline)
        adapted['adapter'] = {'path': 'adapter'}
        self.set_pass(adapted, 1, True)
        return baseline, adapted

    def set_pass(self, result, index, passed):
        result['problems'][index].update(passed=passed, status='passed' if passed else 'failed', exit_code=0 if passed else 1)
        result['summary'] = exp.aggregate(result['problems'])

    def test_scores_changes_and_negative_difference(self):
        baseline, adapted = self.results()
        adapted['protocol']['dataset']['created_at'] = 'today'
        report = exp.compare_results(baseline, adapted)
        self.assertEqual(report['baseline']['pass_at_1_percent'], 50)
        self.assertEqual(report['adapted']['pass_at_1_percent'], 100)
        self.assertEqual(report['difference_percentage_points'], 50)
        self.assertEqual([r['changed'] for r in report['outcomes']], [False, True])
        self.set_pass(adapted, 0, False)
        self.set_pass(adapted, 1, False)
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

    def test_all_four_transitions_and_changed_task_ids(self):
        baseline, adapted = self.results()
        for result in (baseline, adapted):
            result['protocol']['examples'] += [row(13), row(14)]
            result['problems'] += [{**result['problems'][0], 'task_id': 13},
                                   {**result['problems'][1], 'task_id': 14}]
        for result in (baseline, adapted):
            result['generations_sha256'] = exp.digest(exp.generation_rows(result))
        self.set_pass(baseline, 3, False)
        self.set_pass(adapted, 2, False)
        self.set_pass(adapted, 3, False)
        report = exp.compare_results(baseline, adapted)
        self.assertEqual(report['transitions'], {'fail_to_pass': 1, 'pass_to_fail': 1,
                                                'unchanged_pass': 1, 'unchanged_fail': 1})
        self.assertEqual(report['changed_task_ids'], {'fail_to_pass': [12], 'pass_to_fail': [13]})
        self.assertEqual(report['net_correct_change'], 0)
        self.assertEqual(report['difference_percentage_points'], 0)

    def test_provenance_and_invalid_scoring_are_rejected(self):
        for key, value in (('model', {'revision': 'other'}), ('image', 'other-image'),
                           ('prompt_version', 'other'), ('versions', {'mlx': 'other'}),
                           ('scoring', {'startup_seconds': 1, 'wall_seconds': 20, 'cpu_seconds': 3, 'memory_mb': 256}),
                           ('scoring_harness', {'sandbox.py': 'other-hash'})):
            baseline, adapted = self.results()
            adapted['protocol'][key] = value
            with self.assertRaisesRegex(ValueError, 'protocols differ'):
                exp.compare_results(baseline, adapted)
        for change in ('missing-status', 'infra-status', 'pass-flag', 'summary', 'old-scorer', 'missing-protocol'):
            baseline, adapted = self.results()
            if change == 'missing-status':
                adapted['problems'][0].pop('status')
            elif change == 'infra-status':
                adapted['problems'][0]['status'] = 'startup_error'
            elif change == 'pass-flag':
                adapted['problems'][0]['passed'] = False
            elif change == 'summary':
                adapted['summary']['passed'] = 0
            elif change == 'old-scorer':
                adapted['protocol']['scorer_version'] = 'docker-mbpp-v1'
            else:
                adapted['protocol'].pop('scoring_harness')
            with self.assertRaises((ValueError, KeyError)):
                exp.compare_results(baseline, adapted)

    def test_default_config_and_rejects_unbounded_or_full_training(self):
        config = exp.read_json(exp.CONFIG)
        exp.validate_config(config)
        for section, key, value in (('decoding', 'max_tokens', -1), ('scoring', 'wall_seconds', 0), ('scoring', 'startup_seconds', 0),
                                    ('training', 'fine_tune_type', 'full')):
            modified = copy.deepcopy(config)
            modified[section][key] = value
            with self.assertRaises(ValueError):
                exp.validate_config(modified)


if __name__ == '__main__':
    unittest.main()

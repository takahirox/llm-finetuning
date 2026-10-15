"""Focused data-study invariants; fixture code is parsed, never executed on host."""

import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import data_study as study
from scripts import experiment as exp
from scripts.training_exposure import Exposure


def source(i=1):
    return {'question_id': str(i), 'question': f'Calculate a sum of powers of a list of integer values with exponent {i}. ' * 2,
            'solution': f'def power_sum(values):\n    return sum(x ** {i} for x in values)\n',
            'test': 'from solution import power_sum\ndef test_values():\n' +
                    ''.join(f'    assert power_sum([{j}]) == {j ** i}\n' for j in range(1, 6)),
            'subset': 'Math' if i % 2 else 'Algorithm', 'gpt_difficulty': 'easy'}


def fixture_rows(n=30):
    return [study.convert(source(i)) for i in range(1, n + 1)]


def scored_study_pair():
    from test_padding_ab import paired_results
    baseline, adapted = paired_results()
    baseline['adapter'] = None
    for result in (baseline, adapted):
        result['protocol'].update(study_sha256='study', split='development',
            dataset={'kind': study.KIND, 'study_sha256': 'study'}, prompt_version=study.PROMPT)
    training = adapted['adapter']['training']
    training['dataset'] = {'kind': study.KIND, 'study_sha256': 'study', 'selection': 'size-3',
                           'ids': {'train': ['a', 'b', 'c'], 'valid': []}, 'diagnostics': {'count': 3}}
    training['configuration'].update(iters=3, study_exposure={'policy': 'fixed-epochs', 'epochs': 1,
        'sample_ids': ['a', 'b', 'c'], 'study_sha256': 'study'})
    runtime = training['runtime']
    runtime.update(configuration_sha256=exp.digest(training['configuration']), train_batches=3,
                   elapsed_seconds=1, exposure={'sample_counts': {'a': 1, 'b': 1, 'c': 1},
        'samples_seen': 3, 'unique_samples_seen': 3, 'retained_tokens_seen': 21, 'target_tokens_seen': 15,
        'order_sha256': 'trace'})
    return baseline, adapted


class DataStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (exp.ROOT / 'data').mkdir(exist_ok=True)

    def test_source_pins_checksums_and_safe_paths(self):
        with tempfile.TemporaryDirectory(dir=exp.ROOT / 'data') as temporary:
            path = Path(temporary)
            (path / 'fixture').write_bytes(b'fixture')
            pins = {'fixture': {'revision': 'a' * 40, 'url': 'https://example.com/' + 'a' * 40,
                                'filename': 'fixture', 'sha256': exp.file_hash(path / 'fixture')}}
            with patch.object(study.exp, 'read_json', return_value=pins):
                self.assertEqual(study.verified_sources(path), pins)
                (path / 'fixture').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    study.verified_sources(path)
                (path / 'fixture').unlink()
                with patch.object(study.urllib.request, 'urlopen', return_value=io.BytesIO(b'fixture')):
                    study.verified_sources(path, download=True)
                self.assertEqual((path / 'fixture').read_bytes(), b'fixture')
                pins['fixture']['url'] = 'https://example.com/main'
                with self.assertRaisesRegex(ValueError, 'immutable'):
                    study.verified_sources(path)
        with self.assertRaisesRegex(ValueError, 'artifacts'):
            study.local_artifact('/tmp/external-study')

    def test_static_filter_and_private_tests(self):
        sample = study.convert(source())
        self.assertEqual(sample['provenance']['assertions'], 5)
        self.assertTrue(sample['provenance']['quality'])
        prompt = exp.messages(sample)[0]['content']
        self.assertIn('power_sum', prompt)
        self.assertNotIn('assert', prompt)
        self.assertNotIn('return sum', prompt)
        self.assertNotIn('from solution', sample['test_list'][0])
        self.assertIn('test_values()', sample['test_list'][0])
        for field, value, reason in (
            ('solution', '', 'missing'), ('solution', 'This is not Python at all !', 'length'),
            ('solution', 'def f(x):\n    return NotImplementedError()', 'placeholder'),
            ('solution', 'import numpy\ndef f(x):\n    return numpy.sum(x)', 'dependency'),
            ('test', 'import pytest\ndef test_bad():\n    assert True\n    assert True', 'dependency'),
            ('test', 'def test_bad(fixture):\n    assert True\n    assert True', 'test_function')):
            row = source()
            row[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, reason):
                study.convert(row)

    def test_normalized_and_near_overlap_including_history(self):
        row = source(17)
        renamed = dict(row, question_id='different', solution=row['solution'].replace('power_sum', 'sum_power').replace('values', 'data'),
                       test=row['test'].replace('power_sum', 'sum_power'))
        eligible, rejected = study.eligible_rows([row, renamed], [])
        self.assertEqual(len(eligible), 1)
        self.assertEqual(rejected[0]['reason'], 'overlap')
        eligible, rejected = study.eligible_rows([row], [('mbpp/601', row['question'], row['solution'])])
        self.assertEqual(eligible, [])
        self.assertEqual(rejected[0]['with'], 'mbpp/601')
        # Verify the prefix candidate optimization against an exhaustive Jaccard oracle.
        import random
        rng = random.Random(7)
        index = study.OverlapIndex()
        previous = []
        for i in range(150):
            a = frozenset(rng.sample(range(30), rng.randrange(8, 25)))
            b = frozenset(rng.sample(range(30), rng.randrange(8, 25)))
            signature = (a, b)
            expected = any(any(len(x & y) / len(x | y) >= threshold for x, y, threshold in
                               zip(signature, old, (0.8, 0.85))) for old in previous)
            self.assertEqual(index.match(signature) is not None, expected)
            previous.append(signature)
            index.add(signature, str(i))

    def test_nested_selection_leakage_determinism_and_shortfall(self):
        rows = fixture_rows()
        groups = study.select(rows, 42, [3, 10, 20], 2, 3)
        self.assertEqual(groups, study.select(rows[::-1], 42, [3, 10, 20], 2, 3))
        self.assertEqual(groups['size-3'], groups['size-10'][:3])
        self.assertEqual(groups['size-10'], groups['size-20'][:10])
        self.assertEqual(len(groups['quality-3']), 3)
        heldout = {r['task_id'] for k in ('valid', 'development', 'confirmation') for r in groups[k]}
        for key in ('size-20', 'quality-3'):
            self.assertFalse(heldout & {r['task_id'] for r in groups[key]})
        for sizes in ([0], [20, 10], [3, 3], [100]):
            with self.assertRaises(ValueError):
                study.select(rows, 42, sizes, 2, 3)
        with self.assertRaisesRegex(ValueError, 'quality'):
            for r in rows:
                r['provenance']['quality'] = False
            study.select(rows, 42, [3], 2, 3)

    def prepare_fixture(self, temporary):
        path = Path(temporary) / 'study'
        config = exp.read_json(study.CONFIG)
        config.update(validation_size=2, eval_size=3)
        fake = SimpleNamespace(read_table=lambda _: SimpleNamespace(to_pylist=lambda: [source(i) for i in range(1, 31)]))
        with patch.object(study, 'verified_sources', return_value=exp.read_json(study.PINS)), \
             patch.object(study, 'historical', return_value=[]), patch.object(study, 'eligible_rows', return_value=(fixture_rows(), [])), \
             patch.dict('sys.modules', {'pyarrow': SimpleNamespace(parquet=fake), 'pyarrow.parquet': fake}), \
             patch('builtins.print'):
            study.prepare(SimpleNamespace(output=path, sources=temporary, download=False, sizes=[3, 10, 20]), config)
        return path, config

    def test_manifest_integrity_and_training_policy(self):
        with tempfile.TemporaryDirectory(dir=exp.ROOT / 'data') as temporary:
            path, config = self.prepare_fixture(temporary)
            info = study.study_info(path)
            data = exp.dataset_info(path / 'size-20')
            self.assertEqual(data['study_sha256'], info['study_sha256'])
            args = SimpleNamespace(resume_adapter=None, mask_variant='corrected')
            settings = study.training_settings(config, data, args)
            self.assertEqual(settings['iters'], 20)
            config['study'] = {'policy': 'fixed-compute', 'steps': 50}
            with self.assertRaisesRegex(ValueError, 'preregistered'):
                study.training_settings(config, data, args)
            config['study'] = {'policy': 'fixed-epochs', 'epochs': 1}
            config['training']['learning_rate'] = 1e-5
            with self.assertRaisesRegex(ValueError, 'setting differs'):
                study.training_settings(config, data, args)
            (path / 'size-20/train.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                study.study_info(path)

    def test_manifest_detects_leakage_even_with_rehashed_metadata(self):
        with tempfile.TemporaryDirectory(dir=exp.ROOT / 'data') as temporary:
            path, _ = self.prepare_fixture(temporary)
            info = exp.read_json(path / 'study.json')
            info['ids']['confirmation'][0] = info['ids']['development'][0]
            info.pop('study_sha256')
            info['study_sha256'] = exp.digest(info)
            exp.write_json(path / 'study.json', info)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                study.study_info(path)

    def test_empty_coverage_records_audit_without_creating_partial_splits(self):
        with tempfile.TemporaryDirectory(dir=exp.ROOT / 'data') as temporary:
            path = Path(temporary) / 'empty'
            fake = SimpleNamespace(read_table=lambda _: SimpleNamespace(to_pylist=lambda: []))
            with patch.object(study, 'verified_sources', return_value=exp.read_json(study.PINS)), \
                 patch.object(study, 'historical', return_value=[]), \
                 patch.dict('sys.modules', {'pyarrow': SimpleNamespace(parquet=fake), 'pyarrow.parquet': fake}):
                with self.assertRaisesRegex(ValueError, 'Only 0 eligible'):
                    study.prepare(SimpleNamespace(output=path, sources=temporary, download=False, sizes=[300]),
                                  exp.read_json(study.CONFIG))
            self.assertEqual(exp.read_json(path / 'audit.json')['eligible'], 0)
            self.assertTrue((path / 'rejections.jsonl').exists())
            self.assertFalse((path / 'study.json').exists())

    def test_actual_exposure_order_tokens_reuse_and_epoch_coverage(self):
        rows = [{'messages': [{'role': 'user', 'content': str(i)}]} for i in range(3)]
        class Dataset:
            def __len__(self):
                return len(rows)
            def __getitem__(self, index):
                return rows[index]
            def process(self, row):
                i = int(row['messages'][0]['content'])
                return [1, i + 2, 3, 4, 5 + i], 2
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            exp.write_jsonl(path / 'train.jsonl', rows)
            config = {'data': str(path), 'adapter_path': str(path), 'max_seq_length': 4, 'iters': 3,
                      'study_exposure': {'policy': 'fixed-epochs', 'epochs': 1, 'sample_ids': ['a', 'b', 'c']}}
            exposure = Exposure(Dataset(), config)
            for i in (2, 0, 1):
                exposure.observe([[1, i + 2, 3, 4, 0]], [[2, 4]])
            summary = exposure.summary()
            self.assertEqual(summary['sample_counts'], {'c': 1, 'a': 1, 'b': 1})
            self.assertEqual(summary['target_tokens_seen'], 6)
            self.assertEqual(summary['retained_tokens_seen'], 12)
            self.assertEqual(summary['truncated_presentations'], 3)
            self.assertEqual(summary['order_sha256'], exp.digest(exp.read_jsonl(path / 'exposure.jsonl')))
            study.validate_exposure({'exposure': summary, 'train_batches': 3}, config)
            exposure.observe([[1, 2, 3, 4]], [[2, 4]])
            with self.assertRaisesRegex(ValueError, 'accounting'):
                study.validate_exposure({'exposure': exposure.summary(), 'train_batches': 4}, config)
            config['iters'] = 4
            with self.assertRaisesRegex(ValueError, 'every sample'):
                study.validate_exposure({'exposure': exposure.summary(), 'train_batches': 4}, config)
            config['study_exposure'] = dict(config['study_exposure'], policy='fixed-compute', steps=4)
            study.validate_exposure({'exposure': exposure.summary(), 'train_batches': 4}, config)

    def test_confirmation_requires_frozen_choice_and_consumes_once_per_model(self):
        with tempfile.TemporaryDirectory(dir=exp.ROOT / 'data') as temporary:
            path = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                study.confirmation_access(path, {'study_sha256': 'study'}, {}, None, {})
            choice = {'study_sha256': 'study', 'configuration': {}, 'protocol': {'model': {}},
                      'adapters': [{'files': {'a': 'hash'}}]}
            exp.write_json(path / 'choice.json', choice)
            with self.assertRaisesRegex(ValueError, 'differs'):
                study.confirmation_access(path, {'study_sha256': 'study'}, {}, {'files': {'a': 'other'}}, {})
            study.confirmation_access(path, {'study_sha256': 'study'}, {}, None, {})
            with self.assertRaises(FileExistsError):
                study.confirmation_access(path, {'study_sha256': 'study'}, {}, None, {})
            study.confirmation_access(path, {'study_sha256': 'study'}, {}, choice['adapters'][0], {})

    def test_study_comparison_requires_identical_protocol_and_real_exposure(self):
        baseline, adapted = scored_study_pair()
        report = exp.compare_results(baseline, adapted)
        self.assertEqual(report['transitions']['fail_to_pass'], 20)
        self.assertEqual(report['selection'], 'size-3')
        self.assertEqual(report['exposure']['samples_seen'], 3)
        self.assertIn('paired_delta_95_percent_normal_interval_pp', report)
        changed = copy.deepcopy(adapted)
        changed['protocol']['split'] = 'confirmation'
        with self.assertRaisesRegex(ValueError, 'protocols differ'):
            exp.compare_results(baseline, changed)
        adapted['adapter']['training']['runtime']['exposure']['sample_counts'] = {'a': 3}
        with self.assertRaisesRegex(ValueError, 'accounting|every sample'):
            exp.compare_results(baseline, adapted)

    def test_report_separates_quality_and_size_and_rejects_confounders(self):
        baseline, adapted = scored_study_pair()
        quality = copy.deepcopy(adapted)
        quality['adapter']['training']['dataset']['selection'] = 'quality-3'
        with tempfile.TemporaryDirectory(dir=exp.ROOT / 'data') as temporary:
            path = Path(temporary)
            for name, result in (('base', baseline), ('size', adapted), ('quality', quality)):
                exp.write_json(path / f'{name}.json', result)
            args = SimpleNamespace(baseline=path / 'base.json', adapted=[path / 'size.json', path / 'quality.json'],
                                   output=path / 'report.json')
            with patch('builtins.print'):
                study.report(args, {})
            result = exp.read_json(args.output)
            self.assertEqual(len(result['quality_contrasts']), 1)
            self.assertEqual(result['size_contrasts'], [])
            self.assertIsNone(result['summary']['size-3']['seed_stdev_percentage_points'])
            config = quality['adapter']['training']['configuration']
            config['learning_rate'] *= 2
            quality['adapter']['training']['runtime']['configuration_sha256'] = exp.digest(config)
            exp.write_json(path / 'quality.json', quality)
            with self.assertRaisesRegex(ValueError, 'controls differ'):
                study.report(args, {})

    def test_freeze_accepts_only_development_and_preserves_choice(self):
        baseline, adapted = scored_study_pair()
        with tempfile.TemporaryDirectory(dir=exp.ROOT / 'data') as temporary:
            path = Path(temporary)
            exp.write_json(path / 'base.json', baseline)
            exp.write_json(path / 'adapter.json', adapted)
            args = SimpleNamespace(study=path, baseline=path / 'base.json', adapted=[path / 'adapter.json'],
                                   decision='Choose the registered same-size comparison.')
            with patch.object(study, 'study_info', return_value={'study_sha256': 'study'}), patch('builtins.print'):
                study.freeze(args, {})
                self.assertEqual(exp.read_json(path / 'choice.json')['decision'], args.decision)
                with self.assertRaisesRegex(ValueError, 'already frozen'):
                    study.freeze(args, {})

    def test_external_evaluation_uses_only_locked_questions_and_preserves_protocol(self):
        with tempfile.TemporaryDirectory(dir=exp.ROOT / 'data') as temporary:
            path, config = self.prepare_fixture(temporary)
            args = SimpleNamespace(data=path, model='model', adapter=None, split='development',
                                   image='image', output=Path(temporary) / 'result.json')
            with patch.object(exp, 'model_info', return_value={}), \
                 patch.object(exp, 'scoring_protocol', return_value={}), \
                 patch.object(exp, 'load_generator', return_value=(lambda _: 'pass', None)), \
                 patch.object(exp, 'score_saved'):
                exp.evaluate(args, config)
            result = exp.read_json(args.output)
            self.assertEqual(len(result['problems']), 3)
            self.assertIsNone(result['adapter'])
            self.assertEqual(result['protocol']['study_sha256'], study.study_info(path)['study_sha256'])
            self.assertNotIn('code', result['protocol']['examples'][0])
            args.split = 'confirmation'
            with patch.object(exp, 'model_info', return_value={}), \
                 patch.object(exp, 'scoring_protocol', return_value={}), \
                 patch.object(exp, 'load_generator', return_value=(lambda _: 'pass', None)):
                with self.assertRaises(FileNotFoundError):
                    exp.evaluate(args, config)


if __name__ == '__main__':
    unittest.main()

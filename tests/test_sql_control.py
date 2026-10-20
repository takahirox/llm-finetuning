"""Focused SQL checks. No model weights or network are needed."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec('sqlglot') is None:
    raise unittest.SkipTest('Install SQL checks with uv sync --locked --extra sql')

from scripts.freight import digest, read, sha, write
from scripts.sql_control import data, fixtures, run, scoring


def record(i, **changes):
    return {'context': f'CREATE TABLE t{i} (x INTEGER)', 'question': f'Get x from t{i}.',
            'answer': f'SELECT x FROM t{i}'} | changes


class DataTests(unittest.TestCase):
    def test_seeded_split_sizes_ids_order_and_no_leakage(self):
        rows = [record(i) for i in range(60)]
        rows += [dict(rows[0], question=rows[0]['question'].upper() + ' ?')]
        sizes = {'test': 5, 'valid': 5, 'train': 10}
        first, _ = data.select(rows, sizes)
        self.assertEqual(first, data.select(rows, sizes)[0])
        self.assertNotEqual(first, data.select(rows, sizes, seed=41)[0])
        data.validate_splits(first, sizes)
        self.assertEqual([len(first[k]) for k in sizes], [5, 5, 10])
        self.assertEqual(len({r['id'] for split in first.values() for r in split}), 20)

    def test_near_duplicates_filtered_and_rejected_across_splits(self):
        left = record(0)
        right = record(0, question='GET x FROM t0 ?')
        self.assertTrue(data.near(left, right))
        self.assertTrue(data.near(left, dict(right, context='create  TABLE t0(x integer)')))
        rows = [left, right, record(2), record(3)]
        splits, excluded = data.select(rows, {'test': 1, 'train': 2})
        self.assertEqual(sum(map(len, splits.values())), 3)
        self.assertLessEqual(sum(data.near(left, r) for s in splits.values() for r in s), 1)
        invalid = {'test': [left | {'id': data.identity(left)}],
                   'train': [right | {'id': data.identity(right)}]}
        with self.assertRaisesRegex(ValueError, 'leakage'):
            data.validate_splits(invalid, {'test': 1, 'train': 1})
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            data.select([left, right], {'train': 2})

    def test_invalid_fields_and_reserved_fixtures(self):
        fixture = fixtures.cases()[0]
        rows = [fixture, record(2), record(3), record(4), record(5, answer='')]
        splits, _ = data.select(rows, {'test': 2}, reserved=fixtures.cases())
        self.assertFalse(any(r['context'] == fixture['context'] for r in splits['test']))
        with self.assertRaises(ValueError):
            data.select([record(1, answer=None)], {'train': 1})

    def test_prompt_never_reads_gold_and_target_is_only_reference_sql(self):
        row = record(1)
        without_gold = {k: v for k, v in row.items() if k != 'answer'}
        self.assertEqual(data.messages(row), data.messages(without_gold))
        self.assertNotIn(row['answer'], data.prompt(row))
        self.assertEqual([m['role'] for m in data.messages(row, True)], ['user', 'assistant'])
        self.assertEqual(data.messages(row, True)[-1]['content'], row['answer'])

    def test_data_receipts_reject_modified_split_source_and_training_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.json'
            source.write_text(json.dumps([record(i) for i in range(20)]))
            cfg = run.config() | {'dataset_sha256': sha(source), 'dataset_rows': 20,
                                  'split_sizes': {'test': 2, 'valid': 2, 'train': 5}}
            with patch.object(run, 'config', return_value=cfg):
                out = root / 'data'
                run.prepare_data(out, source)
                run.data_fingerprint(out)
                with self.assertRaises(FileExistsError):
                    run.prepare_data(out, source)
                target = out / 'train.jsonl'
                saved = target.read_text()
                target.write_text(saved.replace('SELECT', 'DELETE'))
                with self.assertRaisesRegex(ValueError, 'target mismatch'):
                    run.data_fingerprint(out)
                target.write_text(saved)
                evaluation = out / 'test-records.jsonl'
                evaluation.write_text('\n'.join(reversed(evaluation.read_text().splitlines())) + '\n')
                with self.assertRaisesRegex(ValueError, 'deterministic'):
                    run.data_fingerprint(out)
                (out / 'source.json').write_text('[]')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    run.data_fingerprint(out)


class ScoringTests(unittest.TestCase):
    def test_normalization_matches_published_text_metric_without_fence_extraction(self):
        self.assertEqual(scoring.normalize(' SELECT   X\nFROM t; '), 'select x from t')
        self.assertEqual(scoring.normalize("SELECT 'A'"), scoring.normalize("select 'a'"))
        self.assertNotEqual(scoring.normalize('```sql\nSELECT 1\n```'), scoring.normalize('SELECT 1'))
        self.assertFalse(scoring.outcome({'id': 'x', 'answer': 'SELECT 1'}, 'SELECT 1 AS one')['exact_match'])

    def test_strict_parse_and_non_select_rejection(self):
        for query in ['SELECT 1', 'WITH a AS (SELECT 1 AS n) SELECT n FROM a',
                      'SELECT a FROM t WHERE a IN (SELECT a FROM t)']:
            self.assertIsNotNone(scoring.parsed_select(query), query)
        for query in ['', 'DELETE FROM t', 'SELECT 1; DROP TABLE t', 'SELECT 1; SELECT 2',
                      'SELECT * FROM', 'SELECT x INTO y FROM t', 'PRAGMA table_info(t)',
                      "ATTACH DATABASE '/tmp/x' AS x", '```sql\nSELECT 1\n```']:
            self.assertIsNone(scoring.parsed_select(query), query)

    def test_result_comparison_preserves_duplicates_null_columns_and_order(self):
        result = lambda rows, columns=1: {'rows': rows, 'columns': columns}
        a = result([[1], [None], [1]])
        b = result([[None], [1.0], [1]])
        self.assertTrue(scoring.same_results(a, b, False))
        self.assertFalse(scoring.same_results(a, b, True))
        self.assertFalse(scoring.same_results(a, result([[1], [None]]), False))
        self.assertFalse(scoring.same_results(result([], 1), result([], 2), False))

    def test_every_fixture_gold_executes_and_seeded_rows_are_repeatable(self):
        self.assertEqual(fixtures.population(1701), fixtures.population(1701))
        self.assertNotEqual(fixtures.population(1701), fixtures.population(1702))
        for case in fixtures.cases():
            self.assertTrue(scoring.execution(case, case['answer'])['execution_match'], case['id'])

    def test_execution_catches_literal_case_wrong_projection_and_order(self):
        case = fixtures.cases()[1]
        wrong = case['answer'].replace('Engineering', 'engineering')
        # Same count would conceal case bugs; use row names on a filtering case.
        case = dict(case, answer="SELECT name FROM control17_staff WHERE team = 'Engineering'")
        wrong = case['answer'].replace('Engineering', 'engineering')
        self.assertEqual(scoring.normalize(case['answer']), scoring.normalize(wrong))
        self.assertFalse(scoring.execution(case, wrong)['execution_match'])
        self.assertFalse(scoring.execution(fixtures.cases()[0], 'SELECT * FROM control17_staff')['execution_match'])
        ordered = fixtures.cases()[4]
        self.assertFalse(scoring.execution(ordered, ordered['answer'].replace('DESC', 'ASC'))['execution_match'])

    def test_execution_denies_write_attach_extensions_and_expensive_functions(self):
        case = fixtures.cases()[0]
        for query in ['DELETE FROM control17_staff', 'SELECT 1; DELETE FROM control17_staff']:
            self.assertEqual(scoring.execution(case, query)['execution_category'], 'non_select_or_parse_error')
        for query in ["SELECT load_extension('/tmp/nope')", 'SELECT randomblob(1000000000)',
                      'SELECT * FROM missing_table', "SELECT readfile('/etc/passwd')"]:
            self.assertEqual(scoring.execution(case, query)['execution_category'], 'execution_error')

    def test_worker_timeout_is_a_failure_not_an_exclusion(self):
        import subprocess
        with patch.object(scoring.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 4)):
            result = scoring.execution(fixtures.cases()[0], 'SELECT 1')
        self.assertFalse(result['execution_match'])
        self.assertEqual(result['execution_category'], 'resource_limit')

    def test_paired_arithmetic_and_failure_categories(self):
        records = [{'id': str(i), 'answer': 'SELECT 1'} for i in range(4)]
        base = [scoring.outcome(r, s) for r, s in zip(records, ['SELECT 1', 'SELECT 2', '', 'SELECT 1'])]
        adapted = [scoring.outcome(r, s) for r, s in zip(records, ['', 'SELECT 1', 'SELECT 1', 'SELECT 1'])]
        metrics = scoring.paired(base, adapted)['exact_match']
        self.assertEqual(metrics['base']['correct'], 2)
        self.assertEqual(metrics['adapted']['correct'], 3)
        self.assertEqual(metrics['delta_pp'], 25)
        self.assertEqual(metrics['fail_to_pass'], 2)
        self.assertEqual(metrics['pass_to_fail'], 1)
        self.assertEqual(scoring.aggregate(base)['categories']['empty'], 1)
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            scoring.paired(base, list(reversed(adapted)))


class ReceiptTests(unittest.TestCase):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return 'chat:' + messages[0]['content']

    def training(self, protocol):
        effective = deepcopy(run.config()['training'])
        return {'format_version': 1, 'status': 'complete', 'protocol': deepcopy(protocol),
                'recipe': deepcopy(effective), 'effective_configuration': effective,
                'runtime': {'status': 'complete', 'train_batches': 5000,
                            'train': {'count': 5000}, 'valid': {'count': 100},
                            'integration': {'variant': 'corrected'},
                            'effective_configuration': deepcopy(effective)},
                'adapter_files': {n: 'testhash' for n in
                                  ('adapter_config.json', 'adapters.safetensors', 'trainer_runtime.json')}}

    def receipt(self):
        model = {'model_id': run.config()['model_id'], 'revision': run.config()['model_revision'],
                 'quantization': run.config()['quantization'], 'environment': run.environment()}
        protocol = {'data': {'frozen': True}, 'model': model, 'config': run.config(),
                    'sources': run.sources(), 'environment': run.environment(), 'fixtures': fixtures.fingerprint()}
        records = [record(i) | {'id': str(i)} for i in range(2)]
        cohorts = {'test': records, 'sqlite_sanity': fixtures.cases()}
        result = {'format_version': 1, 'status': 'complete', 'kind': 'base', 'training': None,
                  'protocol': protocol, 'cohorts': {}}
        for name, examples in cohorts.items():
            rows = [scoring.outcome(r, '', name == 'sqlite_sanity') |
                    {'prompt_sha256': digest(run.generation_prompt(self.Tokenizer(), r))} for r in examples]
            result['cohorts'][name] = {'records': examples, 'rows': rows, 'summary': scoring.aggregate(rows)}
        return result, records

    def test_result_scores_prompts_order_and_provenance_are_recomputed(self):
        receipt, records = self.receipt()
        with patch.object(run, 'data_fingerprint', return_value={'frozen': True}), \
             patch.object(run, 'read_jsonl', return_value=records):
            validate = lambda r: run.validate_result(r, Path('unused'), receipt['protocol'], self.Tokenizer())
            validate(receipt)
            mutations = [lambda r: r.update(status='running'),
                         lambda r: r['protocol']['config']['decoding'].update(max_tokens=128),
                         lambda r: r['protocol']['environment'].update(sqlglot='0'),
                         lambda r: r['cohorts']['test']['rows'].reverse(),
                         lambda r: r['cohorts']['test']['rows'][0].update(exact_match=True),
                         lambda r: r['cohorts']['test']['rows'][0].update(prompt_sha256='forged'),
                         lambda r: r['cohorts']['test']['summary']['metrics']['parse_valid'].update(correct=1),
                         lambda r: r['cohorts'].pop('sqlite_sanity')]
            for mutate in mutations:
                broken = deepcopy(receipt)
                mutate(broken)
                with self.assertRaises(ValueError):
                    validate(broken)

    def test_training_receipt_requires_final_complete_matching_recipe(self):
        with self.assertRaisesRegex(ValueError, 'training receipt'):
            run.check_training(None, {})
        with self.assertRaisesRegex(ValueError, 'training receipt'):
            run.check_training({'status': 'failed'}, {})
        protocol = self.receipt()[0]['protocol']
        receipt = self.training(protocol)
        run.check_training(receipt, protocol)
        for mutate in [lambda r: r['runtime'].update(train_batches=4999),
                       lambda r: r['runtime']['integration'].update(variant='legacy'),
                       lambda r: r['effective_configuration'].update(resume_adapter_file='old-adapter'),
                       lambda r: r['protocol']['model'].update(revision='different'),
                       lambda r: r['adapter_files'].pop('adapters.safetensors')]:
            broken = deepcopy(receipt)
            mutate(broken)
            with self.assertRaises(ValueError):
                run.check_training(broken, protocol)

    def test_comparison_writes_report_and_rejects_unpaired_protocol_and_overwrite(self):
        base, records = self.receipt()
        adapted = deepcopy(base)
        adapted.update(kind='adapted', training=self.training(base['protocol']))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'base.json', base)
            write(root / 'adapted.json', adapted)
            with patch.object(run, 'protocol', return_value=base['protocol']), \
                 patch.object(run, 'data_fingerprint', return_value={'frozen': True}), \
                 patch.object(run, 'read_jsonl', return_value=records), \
                 patch('transformers.AutoTokenizer.from_pretrained', return_value=self.Tokenizer()):
                args = (root / 'base.json', root / 'adapted.json', root, root, root / 'report',
                        'Unit test only; no model inference.')
                run.compare(*args)
                self.assertIn('0/2 (0.00%)', (root / 'report/comparison.md').read_text())
                self.assertEqual(read(root / 'report/comparison.json')['metrics']['test']['exact_match']['delta_pp'], 0)
                with self.assertRaises(FileExistsError):
                    run.compare(*args)
                adapted['protocol']['model']['revision'] = 'different'
                write(root / 'adapted.json', adapted)
                with self.assertRaisesRegex(ValueError, 'protocol'):
                    run.compare(*args)

    def test_model_hashes_and_quantization_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'config.json', {'quantization': run.config()['quantization']})
            (root / 'model.safetensors').write_bytes(b'test weights')
            manifest = {'format_version': 1, 'model_id': run.config()['model_id'],
                        'revision': run.config()['model_revision'], 'quantization': run.config()['quantization'],
                        'conversion_dtype': 'bfloat16',
                        'environment': run.environment(), 'upstream_files': {'weights': 'sourcehash'},
                        'files': run.files(root)}
            write(root / 'model_manifest.json', manifest)
            run.model_fingerprint(root)
            (root / 'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                run.model_fingerprint(root)


@unittest.skipUnless(os.environ.get('RUN_MLX_TESTS') == '1', 'Native MLX checks require RUN_MLX_TESTS=1')
class NativeTests(unittest.TestCase):
    def test_sql_chat_boundary_and_corrected_completion_mask(self):
        import mlx.core as mx
        from scripts.loss_masking import target_mask, integration_info
        from scripts.masked_lora import audit_dataset
        from mlx_lm.tuner.datasets import ChatDataset
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                prefix = [7, 8]
                if messages[-1]['role'] != 'assistant':
                    return prefix
                return prefix + [9] * len(messages[-1]['content']) + [0]
        dataset = ChatDataset([{'messages': data.messages(record(1), True)}], Tokenizer(), mask_prompt=True)
        tokens, offset = dataset.process(dataset[0])
        audit = audit_dataset(dataset, 128)
        self.assertEqual(audit['intended_targets'], len(tokens) - offset)
        lengths = mx.array([[offset, len(tokens)]])
        mask = target_mask(lengths, len(tokens) + 2, 'corrected', mx).tolist()[0]
        self.assertEqual(sum(mask), len(tokens) - offset)
        self.assertTrue(mask[len(tokens) - 2])  # Actual end token remains supervised.
        self.assertFalse(mask[len(tokens) - 1])  # Synthetic padding does not.
        self.assertFalse(mask[0])  # Prompt is masked.
        self.assertEqual(integration_info('corrected')['variant'], 'corrected')

    def test_all_seven_lora_modules_use_alpha_over_rank_scale(self):
        from mlx_lm.models.qwen2 import Model, ModelArgs
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.tuner.lora import LoRALinear
        model = Model(ModelArgs(model_type='qwen2', hidden_size=32, num_hidden_layers=2,
                                intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                                rms_norm_eps=1e-6, vocab_size=64))
        model.freeze()
        linear_to_lora_layers(model, -1, run.config()['training']['lora_parameters'])
        layers = [layer for _, layer in model.named_modules() if isinstance(layer, LoRALinear)]
        self.assertEqual(len(layers), 14)
        self.assertTrue(all(layer.scale == 2.0 and layer.lora_b.shape[0] == 8 for layer in layers))


if __name__ == '__main__':
    unittest.main()

"""Independent benchmark tests; no generated program executes on the host."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

from scripts import experiment as ex
from scripts import independent_eval as he


def problem(i=0):
    return {'task_id': f'HumanEval/{i}', 'prompt': f'def f{i}(x):\n    """Return x plus one."""\n',
            'entry_point': f'f{i}', 'test': 'def check(candidate):\n    assert candidate(2) == 3\n',
            'canonical_solution': '    return x + 1\n'}


def result(seed=None, passed_ids=()):
    examples = [{k: v for k, v in problem(i).items() if k != 'canonical_solution'} for i in range(164)]
    rows = [{'task_id': p['task_id'], 'response': 'pass', 'solution': 'pass\n',
             'passed': i in passed_ids, 'status': 'passed' if i in passed_ids else 'failed',
             'exit_code': 0 if i in passed_ids else 1} for i, p in enumerate(examples)]
    value = {'status': 'complete', 'generation_status': 'complete',
             'protocol': {'examples': examples, 'prompt_version': he.PROMPT_VERSION, 'test_version': he.TEST_VERSION},
             'adapter': None if seed is None else {'training': {'configuration': {'seed': seed}}}, 'problems': rows}
    value['generations_sha256'] = ex.digest(ex.generation_rows(value))
    value['summary'] = ex.aggregate(rows)
    return value


class HumanEvalTests(unittest.TestCase):
    def test_prompt_cannot_leak_hidden_test_or_reference(self):
        p = problem(); p['test'] = 'HIDDEN_TEST_SENTINEL'; p['canonical_solution'] = 'REFERENCE_SENTINEL'
        prompt = he.messages(p)[0]['content']
        self.assertIn(p['prompt'], prompt)
        self.assertNotIn('SENTINEL', prompt)
        self.assertIn('complete Python module', prompt)

    def test_program_assembly_preserves_prefix_and_keeps_tests_separate(self):
        p = problem(); p['prompt'] = 'from typing import List\n' + p['prompt']
        code = he.assembled_solution(p, 'def f0(x): return x+1\n')
        self.assertTrue(code.startswith(p['prompt']))
        self.assertNotIn('check(', code)
        case = he.test_case(p)
        self.assertEqual(case['test_setup_code'], '')
        self.assertTrue(case['test_list'][0].endswith('check(f0)\n'))
        for entry in ('bad);exit()', 'class', 'a.b'):
            with self.assertRaises(ValueError): he.test_case({**p, 'entry_point': entry})

    def prepare(self, path):
        rows = [problem(i) for i in reversed(range(164))]
        raw = gzip.compress('\n'.join(json.dumps(r) for r in rows).encode())
        import hashlib
        archive = path.parent / 'source.gz'; archive.write_bytes(raw)
        with patch.object(he, 'ARCHIVE_SHA256', hashlib.sha256(raw).hexdigest()):
            he.prepare(archive, path)
            manifest, records = he.dataset_info(path)
        return manifest, records

    def test_preparation_separates_references_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data'
            manifest, records = self.prepare(path)
            self.assertEqual(len(records), 164)
            self.assertEqual(records[0]['task_id'], 'HumanEval/0')
            self.assertTrue(all(set(r) == {'task_id','prompt','entry_point','test'} for r in records))
            self.assertEqual(set(manifest['files']), {'problems.jsonl','references.jsonl'})
            with self.assertRaises(ValueError): self.prepare(path)

    def test_archive_hash_and_dataset_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data'; self.prepare(path)
            archive = Path(temporary) / 'source.gz'
            with self.assertRaises(ValueError): he.prepare(archive, Path(temporary) / 'other')
            manifest = ex.read_json(path / 'manifest.json')
            with patch.object(he, 'ARCHIVE_SHA256', manifest['archive_sha256']):
                (path / 'problems.jsonl').write_text('{}\n')
                with self.assertRaises(ValueError): he.dataset_info(path)

    def test_function_normalization_detects_rename_not_changed_logic(self):
        left = 'def f(x: int):\n    """Description"""\n    return x + 1\n'
        same = 'def g(y):\n    return y+1\n'
        other = 'def g(y):\n    return y+2\n'
        self.assertEqual(he.function_fingerprints(left), he.function_fingerprints(same))
        self.assertNotEqual(he.function_fingerprints(left), he.function_fingerprints(other))
        self.assertEqual(he.function_fingerprints('not python !'), set())

    def test_overlap_is_frozen_rule_based_and_retains_unmatched_tasks(self):
        ps = [problem(0), problem(1)]
        refs = [{'task_id': p['task_id'], 'program': p['prompt'] + ('    return x+1\n' if i==0 else '    return x*3\n')} for i,p in enumerate(ps)]
        mbpp = [{'task_id': 601, 'text': 'An unrelated description.', 'code': 'def renamed(y): return y+1'}]
        audit = he.overlap_audit(ps, refs, mbpp)
        self.assertEqual(audit['excluded_ids'], ['HumanEval/0'])
        self.assertEqual(audit['primary_ids'], ['HumanEval/1'])

    def test_reference_audit_routes_execution_to_existing_sandbox(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'data'; manifest, problems = self.prepare(path)
            with patch.object(he, 'dataset_info', return_value=(manifest, problems)), \
                 patch.object(ex, 'scoring_protocol', return_value={'image':'sha256:x','scoring':{}}), \
                 patch.object(he, 'score', return_value={'passed':True,'status':'passed','exit_code':0}) as score:
                output = Path(temporary) / 'audit.json'
                he.audit_references(path, {}, output)
                self.assertEqual(score.call_count, 164)
                self.assertIn('return x + 1', score.call_args[0][0])
                self.assertEqual(ex.read_json(output)['status'], 'complete')
                with self.assertRaises(ValueError): he.audit_references(path, {}, output)

    def test_summary_mean_sample_sd_and_paired_transitions(self):
        runs = [result(None,(0,1)), result(43,(0,2)), result(44,(0,1,2)), result(45,(0,))]
        summary = he.summarize(runs, ['HumanEval/0','HumanEval/1','HumanEval/2'])
        self.assertAlmostEqual(summary['primary']['mean_percent'], 200/3)
        self.assertAlmostEqual(summary['primary']['sample_sd_percentage_points'], 100/3)
        pair = summary['primary']['pairs'][0]
        self.assertEqual(pair['task_ids']['fail_to_pass'], ['HumanEval/2'])
        self.assertEqual(pair['task_ids']['pass_to_fail'], ['HumanEval/1'])

    def test_comparison_rejects_duplicate_seeds_partial_and_mismatched_protocols(self):
        original = [result(None), result(43), result(44), result(45)]
        for mutation in ('seed','partial','protocol','summary','generation'):
            runs = copy.deepcopy(original)
            if mutation=='seed': runs[3]['adapter']['training']['configuration']['seed']=43
            elif mutation=='partial': runs[1]['status']='running'
            elif mutation=='protocol': runs[1]['protocol']['prompt_version']='changed'
            elif mutation=='summary': runs[1]['summary']['passed']=10
            else: runs[1]['problems'][0]['response']='changed'
            with self.assertRaises(ValueError): he.summarize(runs, ['HumanEval/0'])
        with self.assertRaises(ValueError): he.summarize(original, [])
        with self.assertRaises(ValueError): he.summarize(original, ['unknown'])

    def test_complete_module_evaluation_never_exposes_test_to_generator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = ex.read_json(ex.CONFIG)
            examples = [{k:v for k,v in problem(i).items() if k!='canonical_solution'} for i in range(164)]
            manifest = {'fixture': True}
            plan = root/'plan.json'
            ex.write_json(plan, {'dataset':manifest,'decoding':config['decoding'],'scoring':config['scoring']})
            tokenizer = Mock(); tokenizer.apply_chat_template.return_value='rendered visible prompt'
            generate = Mock(return_value='def f0(x): return x+1')
            lm = SimpleNamespace(load=Mock(return_value=(object(),tokenizer)),generate=generate)
            core = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
            modules = {'mlx_lm':lm,'mlx_lm.sample_utils':SimpleNamespace(make_sampler=Mock()),
                       'mlx':SimpleNamespace(core=core),'mlx.core':core}
            args = SimpleNamespace(model='fixture',data='fixture',plan=str(plan),adapter=None,output=str(root/'result.json'))
            with patch.dict('sys.modules',modules), patch.object(ex,'require_mlx'), \
                 patch.object(ex,'model_info',return_value={'fixture':True}), \
                 patch.object(he,'dataset_info',return_value=(manifest,examples)), \
                 patch.object(ex,'scoring_protocol',return_value={'image':'sha256:fixture','scoring':{}}), \
                 patch.object(he,'score',return_value={'passed':True,'status':'passed','exit_code':0}) as scorer:
                he.evaluate(args,config)
                he.validate_result(ex.read_json(args.output))
                self.assertEqual(generate.call_count,164)
                for call in tokenizer.apply_chat_template.call_args_list:
                    self.assertNotIn('def check(candidate)',call.args[0][0]['content'])
                    self.assertNotIn('return x + 1',call.args[0][0]['content'])
                self.assertTrue(scorer.call_args.args[0].startswith(examples[-1]['prompt']))
                self.assertIn('check(f163)',scorer.call_args.args[1]['test_list'][0])
                with self.assertRaises(ValueError): he.evaluate(args,config)

    def test_invalid_score_and_truncated_protocol_are_rejected(self):
        for mutation in ('exit','boolean','cohort'):
            r=result(None,(0,))
            if mutation=='exit': r['problems'][0]['exit_code']=1
            elif mutation=='boolean': r['problems'][0]['passed']=1
            else:
                r['protocol']['examples']=r['protocol']['examples'][:2]
                r['problems']=r['problems'][:2]
                r['generations_sha256']=ex.digest(ex.generation_rows(r))
                r['summary']=ex.aggregate(r['problems'])
            with self.assertRaises(ValueError): he.validate_result(r)


if __name__ == '__main__': unittest.main()

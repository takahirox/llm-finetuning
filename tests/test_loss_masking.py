"""Native token-level checks, explicitly enabled on supported MLX hosts.

RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
"""

import importlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import experiment as exp
from scripts import loss_masking as masking
from scripts import masked_lora


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0  # Deliberate EOS/PAD overlap, including genuine ID zero targets.

    def apply_chat_template(self, messages, **kwargs):
        prefix = [4, 5]
        if messages[-1]['role'] != 'assistant':
            return prefix
        return prefix + [6] * int(messages[-1]['content']) + [0]


@unittest.skipUnless(os.environ.get('RUN_MLX_TESTS') == '1', 'set RUN_MLX_TESTS=1 on native Apple Silicon')
class NativeMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        exp.require_mlx()
        import mlx.core as mx
        import mlx.nn as nn
        cls.mx, cls.nn = mx, nn
        cls.trainer = importlib.import_module('mlx_lm.tuner.trainer')
        cls.datasets = importlib.import_module('mlx_lm.tuner.datasets')

    def batches(self, rows, max_length=64):
        return next(self.trainer.iterate_batches(rows, len(rows), max_length, seed=42))

    def mask(self, lengths, batch, variant):
        return masking.target_mask(lengths, batch.shape[1] - 1, variant, self.mx).tolist()

    def test_mixed_lengths_masks_labels_and_matches_legacy_trainer(self):
        batch, lengths = self.batches([([4, 5, 6, 0], 2), ([4, 5, 7, 8, 0], 2)])
        corrected, legacy = (self.mask(lengths, batch, v) for v in ('corrected', 'legacy'))
        for i, (offset, length) in enumerate(lengths.tolist()):
            self.assertEqual(corrected[i], [offset <= p < length for p in range(1, batch.shape[1])])
            self.assertEqual(legacy[i], [offset <= p <= length for p in range(1, batch.shape[1])])
            self.assertEqual(sum(corrected[i]), length - offset)
            # Keep the real EOS even when its ID equals padding; remove synthetic padding.
            self.assertEqual(batch[i, length - 1].item(), 0)
            self.assertTrue(corrected[i][length - 2])
            self.assertFalse(corrected[i][length - 1])
            self.assertFalse(corrected[i][0])  # Prompt remains masked.
        logits = self.mx.zeros((*batch[:, :-1].shape, 10))
        model = lambda _: logits
        original = self.trainer.default_loss(model, batch, lengths)
        reproduced = masking.make_loss('legacy')(model, batch, lengths)
        self.assertEqual(original[1].item(), reproduced[1].item())
        self.assertAlmostEqual(original[0].item(), reproduced[0].item())
        self.assertEqual(original[1].item() - masking.make_loss('corrected')(model, batch, lengths)[1].item(), 2)

    def test_single_example_rounding_also_adds_one_legacy_padding_target(self):
        batch, lengths = self.batches([([4, 5, 6, 0], 2)])
        self.assertEqual(batch.shape, (1, 33))
        self.assertEqual(sum(self.mask(lengths, batch, 'legacy')[0]), 3)
        self.assertEqual(sum(self.mask(lengths, batch, 'corrected')[0]), 2)

    def test_truncation_exact_length_and_first_completion(self):
        batch, lengths = self.batches([([4, 5, 6, 7, 8, 9], 2), ([4, 5, 6], 2)], max_length=4)
        for i, (offset, length) in enumerate(lengths.tolist()):
            self.assertEqual(self.mask(lengths, batch, 'corrected')[i], [offset <= p < length for p in range(1, 4)])
            self.assertTrue(self.mask(lengths, batch, 'corrected')[i][offset - 1])
        batch, lengths = self.batches([([4, 5, 6, 7], 2)], max_length=4)
        self.assertEqual(self.mask(lengths, batch, 'legacy'), self.mask(lengths, batch, 'corrected'))

    def test_padding_logits_have_no_corrected_loss_or_gradient(self):
        mx = self.mx
        batch, lengths = self.batches([([4, 5, 6, 0], 2)])
        logits = mx.zeros((1, batch.shape[1] - 1, 10))
        for variant in masking.VARIANTS:
            loss = masking.make_loss(variant)
            gradient = mx.grad(lambda values: loss(lambda _: values, batch, lengths)[0])(logits)
            self.assertEqual(gradient[0, 0].abs().sum().item(), 0)  # Prompt.
            self.assertGreater(gradient[0, 1].abs().sum().item(), 0)  # Completion.
            self.assertGreater(gradient[0, 2].abs().sum().item(), 0)  # Real EOS.
            self.assertEqual(gradient[0, 3].abs().sum().item() > 0, variant == 'legacy')
            self.assertEqual(gradient[0, 4:].abs().sum().item(), 0)

    def test_prompt_tokenization_and_empty_truncations_fail_explicitly(self):
        data = self.datasets.ChatDataset([{'messages': [
            {'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': '2'}]}], Tokenizer(), mask_prompt=True)
        self.assertEqual(data.process(data[0]), ([4, 5, 6, 6, 0], 2))
        self.assertEqual(masked_lora.audit_dataset(data, 4)['intended_targets'], 2)
        with self.assertRaisesRegex(ValueError, 'no completion'):
            masked_lora.audit_dataset(data, 2)
        with patch.object(data.tokenizer, 'apply_chat_template', side_effect=[[9, 5, 6], [4, 5], [4, 5]]):
            with self.assertRaisesRegex(ValueError, 'boundary mismatch'):
                masked_lora.audit_dataset(data, 4)

    def test_source_guard_and_real_tiny_lora_training_for_both_variants(self):
        mx, nn = self.mx, self.nn

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 8)

            def __call__(self, x):
                return self.proj(x)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(10, 8)
                self.layers = [Layer()]
                self.output = nn.Linear(8, 10)

            def __call__(self, x):
                return self.output(self.layers[0](self.embedding(x)))

        lora = importlib.import_module('mlx_lm.lora')
        original_train = lora.train
        with patch.dict(masking.UPSTREAM_HASHES, {'lora.py': 'incorrect'}):
            with self.assertRaisesRegex(ValueError, 'source differs'):
                masking.integration_info('corrected')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples = [{'messages': [{'role': 'user', 'content': 'task'},
                                      {'role': 'assistant', 'content': str(n)}]} for n in (1, 2)]
            for name in ('train', 'valid'):
                exp.write_jsonl(root / f'{name}.jsonl', examples)
            weights = []
            for variant in ('legacy', 'corrected'):
                output = root / variant
                config = dict(exp.read_json(exp.CONFIG)['training'], train=True, test=False,
                              data=str(root), model='tiny-fixture', adapter_path=str(output),
                              batch_size=2, iters=1, num_layers=1, grad_checkpoint=False,
                              steps_per_report=1, steps_per_eval=1, save_every=1,
                              lora_parameters={'rank': 2, 'scale': 2.0, 'dropout': 0.0, 'keys': ['proj']})
                mx.random.seed(123)
                model = TinyModel()
                # All tokenization, batching, LoRA conversion, optimizer updates and
                # validation below use the real installed MLX-LM, with no Qwen weights.
                with patch.object(lora, 'load', return_value=(model, Tokenizer())):
                    masked_lora.run(config, variant)
                runtime = exp.read_json(output / 'trainer_runtime.json')
                self.assertEqual(runtime['status'], 'complete')
                self.assertEqual(runtime['integration']['variant'], variant)
                self.assertEqual(runtime['train_batches'], 1)
                self.assertGreater(runtime['validation_batches'], 0)
                self.assertEqual(runtime['tokenizer']['pad_token_id'], runtime['tokenizer']['eos_token_id'])
                self.assertIs(lora.train, original_train)
                weights.append(mx.load(str(output / 'adapters.safetensors')))
            self.assertTrue(any(not mx.array_equal(weights[0][k], weights[1][k]).item() for k in weights[0]))
            config.update(adapter_path=str(root / 'no-targets'), max_seq_length=2)
            with patch.object(lora, 'load', return_value=(TinyModel(), Tokenizer())):
                with self.assertRaisesRegex(ValueError, 'no completion'):
                    masked_lora.run(config, 'corrected')
            runtime = exp.read_json(root / 'no-targets' / 'trainer_runtime.json')
            self.assertEqual(runtime['status'], 'failed')
            self.assertEqual(runtime['train_batches'], 0)
            self.assertIs(lora.train, original_train)

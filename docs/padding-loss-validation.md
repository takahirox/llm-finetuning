# Issue #9 implementation validation

Checks performed on 2026-10-15 in the assigned GitWeave worktree. These are
implementation checks, not measured Qwen correctness outcomes.

## Passed checks

- `uv sync --locked`, `uv lock --check`, `uv pip check`: passed with uv 0.12.23.
  No dependency or lockfile changes were needed.
- `.venv/bin/python scripts/check_environment.py`: passed on native arm64 macOS
  26.6.2, Python 3.12.13, MLX 0.32.3, MLX-LM 0.32.0, Metal available.
- `RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -v`:
  **48 tests passed**, including all six native MLX tests. Those tests assert
  target IDs/masks, prompt exclusion, mixed lengths, batch-size-one padding,
  truncation, exact-width examples, first completion inclusion, EOS/PAD overlap,
  zero padding gradients, nonzero completion/EOS gradients, and numerical
  agreement between the diagnostic loss and installed upstream default loss.
- The native suite ran both variants through the real pinned LoRA conversion,
  optimizer, batching, training and validation machinery for **one step each on a
  tiny synthetic model**. Only model loading was replaced with a tiny in-memory
  fixture. Both produced complete runtime evidence and different adapter updates.
  A fully truncated completion failed before optimization and restored the trainer
  binding. This is evidence of hook execution, not a Qwen or MBPP A/B measurement.
- Hardware-independent tests cover deterministic 294/80 splitting, reference-free
  validation prompts, split/hash integrity, validation scoring selection, all four
  paired transitions, positive/negative/zero signed differences, available failure
  counts, variant selection and rejection of confounded or incomplete provenance.
  A successful child exit without correct runtime evidence is rejected.
  `.venv/bin/python -m unittest discover -s tests -q` separately passed the
  42 hardware-independent tests with the six opt-in native tests skipped.
- `python -m compileall -q scripts tests`: passed with the project interpreter.
- CLI help for `scripts.masked_lora` and `scripts.experiment compare-masks`: passed.
- Changed Markdown relative links/fence closure, JSON configuration consistency,
  ignored artifact paths, and `git diff --check`: passed.

## Pinned real-data/tokenizer audit

Real `prepare-data` downloaded and verified the pinned MBPP source and produced
294 train, 80 validation and 50 unused official-test prompts in ignored
`data/mbpp-padding-ab/`. The ordered split checksum was
`9a76fe7a40460694018c079e015031b4dd28e3b1e16e68b2e818778e9d81cba0`.
Training/validation IDs are disjoint, cover only 601–974, and exclude every official
test ID 11–510. Generated validation scoring records have no reference solutions.

Only tokenizer/configuration files were downloaded from Qwen revision
`cdbee75f17c01a7cc42f958dc650907174af0554` into the ignored Hub cache; no model
weight files were downloaded. Its tokenizer loaded locally without remote code.
The actual pinned `ChatDataset` and project preflight checked every training and
validation row at the configured 1024-token limit:

| Split | Examples | Truncated | Intended completion targets |
| --- | ---: | ---: | ---: |
| Training | 294 | 0 | 16,959 |
| Validation | 80 | 0 | 5,133 |

All prompt prefixes matched and every example retained completion targets.
Tokenization/boundary SHA-256 values from `audit_dataset` were
`abbb3307a6828916377b9eb2c554f4330af7c62a0f120086f52194457093affa`
(training) and
`f9fee9f9dbb6a166cd01a2f2b22c5004e154a06cc81189b57d9365a26a115a73`
(validation). Tokenizer PAD ID was 151643, EOS ID was 151645; collator padding ID
0 decoded to `!`. The synthetic native tests separately cover EOS/PAD overlap.

## Not performed / future experimentation

The two real Qwen 100-step training runs, model inference, Docker scoring of the
80 validation tasks, and resulting correctness comparison were **not executed**.
This host supports MLX, but no full Qwen weights, historical adapters or reports
were present in the assigned worktree. Full-weight download/conversion and the
real training/scoring experiments remain optional follow-up work with substantial
memory/time costs. No measured train/validation losses, elapsed training times,
Qwen pass@1 scores, format-failure rates, or causal accuracy effect are claimed.
The [experimental guide](padding-loss-experiment.md) gives complete runnable steps
and the fields to report. The existing Docker scorer was exercised by fixture
mocks in this suite; no real Docker execution was performed for this change.

The already-inspected official 500 tasks were not evaluated again. No independent
final holdout was sourced or evaluated. Historical artifacts were not modified.
PR creation against `main`, remote review and merging are delegated to the enclosing
GitWeave flow; this node only commits the implementation locally for Issue #9.
No post-merge verification is required and no training/deployment runs automatically.

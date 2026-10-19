# Controlled padding-loss experiment (Issue #9)

This workflow tests whether correcting the loss mask changes correctness. No
Qwen A/B accuracy measurements are included in this change. The historical
358/500 base versus 338/500 original-adapter scores and their reproduced
regressions are prior diagnostic evidence; they do not establish the cause.

## Diagnosis and correction

The exact locked `mlx-lm==0.32.0` wheel was inspected, including
[`ChatDataset.process`](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/tuner/datasets.py),
[`iterate_batches`, `default_loss`, `train`, and `evaluate`](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/tuner/trainer.py),
and the [`lora.train_model` dispatch](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/lora.py).
The dependency versions and lock are unchanged. `scripts/loss_masking.py` records
SHA-256 hashes of these three installed source files and refuses to run against
an uninspected implementation, even if its package version is unchanged.

For chat data, the dataset applies the full chat template, and with `mask_prompt`
it computes the offset from the preceding user turn plus generation prefix.
The project checks that these tokens are an exact prefix of the full conversation.
The collator pads with integer **0**, regardless of the tokenizer's PAD ID, to
`min(1 + 32 * ceil(longest_length / 32), max_seq_length)`. It truncates token
arrays and updates their lengths, leaving prompt offsets unchanged. The loss
shifts inputs/targets: target position `p` means `batch[:, p]`, starting at 1.
The legacy rule is `offset <= p <= length`. Since `length` is an exclusive array
bound, `p == length` is already synthetic padding. The corrected rule is
`offset <= p < length`. No other prompt or target boundary changes.

A deterministic example uses tokens `[4, 5, 6, 0]`, offset 2 and length 4. With
the normal 33-token padded batch, the masks for the first five targets are:

| Target position | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Target ID | 5 | 6 | 0 (real EOS) | 0 (padding) | 0 (padding) |
| Legacy loss mask | false | true | true | true | false |
| Corrected loss mask | false | true | true | false | false |

This reproduces **one extra padding target per padded example**, including batch
size 1, rather than loss on every padded position. When truncation fills the
whole batch width there is no extra target, so the rules coincide. Mixed-length
rows can differ even when the longest row fills the width. ID-based masking would
be incorrect: real EOS/completion tokens must retain loss even if EOS equals PAD.
At the pinned Qwen revision, tokenizer PAD is 151643 and EOS is 151645; they do not
overlap. Collator padding ID 0 decodes to `!`, and does not equal either. These
IDs are also recorded at runtime; overlap is tested with a synthetic tokenizer.

`python -m scripts.experiment train` now defaults to corrected masking. It starts
`scripts.masked_lora` in a dedicated subprocess. That integration explicitly binds
`lora.train` to a dispatcher supplying the public `trainer.train(loss=...,
iterate_batches=...)` hooks. The binding is restored in `finally`; no virtual
environment files are patched. The upstream trainer propagates the same loss and
batch iterator to validation. The legacy hook is numerically checked against the
installed `default_loss`, and differs from the corrected hook only in the upper
bound. Both variants use identical tokenization, batch order, prompt masking,
truncation and optimizer code. Fully truncated completions or inconsistent prompt
prefixes fail before optimizer updates; examples are never silently dropped.

## Split and interpretation

Use the separate checked-in `configs/padding-ab.json`. It retains the original
training/decoding settings, with validation enlarged from 20 to **80**. The
result is **294 training + 80 validation** tasks, drawn solely from IDs 601–974.
Sorted IDs are shuffled with `random.Random(42)`; validation takes the first 80.
`manifest.json` records exact ordered IDs, their combined SHA-256, source revision,
source SHA-256, and individual file hashes. Preparing this configuration gives
split SHA-256 `9a76fe7a40460694018c079e015031b4dd28e3b1e16e68b2e818778e9d81cba0`.
The 50 official-test IDs also present in that manifest are disjoint and are not
evaluated by the commands below. All official test IDs 11–510 are excluded from
training and validation, including those outside that 50-task sample.

`train.jsonl` contains only training prompts/reference completions. `valid.jsonl`
contains the 80 held-out reference completions for teacher-forced loss only.
`validation_eval.jsonl` contains the same 80 prompts/assertions/setup in exactly
the manifest order, with **no reference solutions or challenge tests**. Only the
user prompt is passed to generation; assistant references never enter generation
prompts or the scorer. The project verifies hashes, IDs, split separation, and
that the validation scoring prompts match the loss-validation user turns.

The original configuration and all historical adapters/reports retain their
354/20 split. Comparing an old adapter against a new 294/80 adapter changes the
training examples as well as the objective and **cannot isolate the PAD effect**.
Both arms below must be fresh runs on the same new data. Internal validation is
development data, not an independent final holdout: some of these 80 tasks were
in the historical adapter's training set. Do not use that historical adapter as
an arm. Public MBPP pretraining contamination is also unresolved. Enlarging
validation reduces sampling noise but does not establish broad generalization.

## Copy-and-run commands on Apple Silicon

Run from a single committed checkout, and keep that code/dependency environment
unchanged across both arms and evaluations. All output paths below are ignored;
commands refuse existing adapter directories and result files. Use a new suffix
for reruns. Existing model conversion can be reused only if its pinned revision,
dtype and file hashes match. See [legacy README setup](../README-legacy.md#setup) for uv/Python
installation and [model preparation](../README-legacy.md#1-download-and-convert-the-model-once)
for model/license/storage details.

```sh
uv sync --locked
uv lock --check
uv pip check
.venv/bin/python scripts/check_environment.py
RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.experiment prepare-model --model models/qwen3-4b-2507
.venv/bin/python -m scripts.experiment --config configs/padding-ab.json prepare-data \
  --data data/mbpp-padding-ab

# Diagnostic faulty-objective arm, explicitly opted in.
.venv/bin/python -m scripts.experiment --config configs/padding-ab.json train \
  --model models/qwen3-4b-2507 --data data/mbpp-padding-ab \
  --mask-variant legacy --adapter-output adapters/padding-ab-legacy

# Corrected objective; this is also the default when the flag is omitted.
.venv/bin/python -m scripts.experiment --config configs/padding-ab.json train \
  --model models/qwen3-4b-2507 --data data/mbpp-padding-ab \
  --mask-variant corrected --adapter-output adapters/padding-ab-corrected

docker pull python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36
.venv/bin/python -m scripts.experiment --config configs/padding-ab.json evaluate \
  --model models/qwen3-4b-2507 --data data/mbpp-padding-ab --split validation \
  --adapter adapters/padding-ab-legacy --output outputs/padding-ab-legacy.json
.venv/bin/python -m scripts.experiment --config configs/padding-ab.json evaluate \
  --model models/qwen3-4b-2507 --data data/mbpp-padding-ab --split validation \
  --adapter adapters/padding-ab-corrected --output outputs/padding-ab-corrected.json
.venv/bin/python -m scripts.experiment --config configs/padding-ab.json compare-masks \
  --legacy outputs/padding-ab-legacy.json --corrected outputs/padding-ab-corrected.json \
  --output outputs/padding-ab-comparison.json
```

Do not pass `--resume-adapter`: this comparison requires two fresh runs from the
same immutable base weights, 100 steps each, batch size 1, seed 42, Adam at 1e-5,
8 layers, rank 8, scale 20, dropout 0, q/v projections, length 1024 and gradient
checkpointing. The remaining settings are shared by the single configuration.
There is no automatic checkpoint selection: evaluate each final 100-step adapter.
`val_batches=4` remains identical in both arms; it is a small loss diagnostic,
while correctness scoring covers **all 80** validation tasks.

The bounded Docker scorer, decoding seed/greedy settings, prompts and execution
limits remain identical. Each evaluation saves raw responses, extracted solutions,
per-task results and model/adapter/data/scorer provenance. If scoring aborts after
all generations were saved, rescore **both** result files under the same current
scorer into new paths, then compare those files:

```sh
.venv/bin/python -m scripts.experiment --config configs/padding-ab.json rescore \
  --input outputs/padding-ab-legacy.json --output outputs/padding-ab-legacy-rescored.json
.venv/bin/python -m scripts.experiment --config configs/padding-ab.json rescore \
  --input outputs/padding-ab-corrected.json --output outputs/padding-ab-corrected-rescored.json
```

## Evidence and reporting

Each adapter's `experiment_training.json` includes the variant, pinned upstream
source hashes, project source/lock hashes, Git revision, dataset manifest, base
weight hashes, versions and requested configuration. `trainer_runtime.json`
records the actual hook dispatch, train/validation batch counts, effective upstream
configuration including defaults, tokenizer IDs, tokenization/boundary hashes,
truncation counts and elapsed seconds. The parent verifies this evidence before
accepting training as complete, then hashes the runtime/config files with the
adapter files. Training/validation losses remain in `training.log`.

`compare-masks` rejects missing runtime evidence, resumed or non-100-step adapters,
reversed variants, old splits, official-test scoring, different ordered validation
examples, changed data/model/code/dependencies/tokenization/training settings,
or different decoding/scoring conditions. Output-directory paths may differ.
An ordinary base-versus-adapter comparison still uses `compare`.

Report each arm's success count and pass@1, the signed percentage-point difference
**corrected minus legacy**, net successes, fail→pass, pass→fail, unchanged-pass,
unchanged-fail, and the per-task outcomes. JSON keys `baseline` and `adapted` mean
legacy and corrected respectively for `comparison: legacy-to-corrected-mask`.
The report also counts available failure categories: `failed`, `timeout` and
`format_error`. This scorer groups syntax, assertion and other runtime failures
under `failed`; it does not diagnose logic-error subclasses. Preserve both full
result files, adapter metadata and logs when recording the comparison. A negative
or zero result is valid; do not assume masking explains the historical -4 points.
Even a positive paired difference on these 80 tasks needs uncertainty/replication
and a genuinely independent final holdout before broader claims. Do not repeatedly
tune on the already-inspected 500 official test tasks. Learning-rate and step-count
studies are separate follow-up experiments.

[Implementation validation](padding-loss-validation.md) records checks actually
run and pending real experiments. No deployment, benchmark, or other post-merge
verification runs automatically. PR creation/review against `main` is delegated to
the enclosing GitWeave flow; this implementation node only commits local changes.

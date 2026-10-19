# Paired evaluation of the existing 100-step adapter

This procedure addresses [Issue #7](https://github.com/takahirox/llm-finetuning/issues/7).
Run from the repository root with the existing locked Apple Silicon environment,
converted `models/qwen3-4b-2507` checkpoint, and completed `adapters/mbpp-lora`
100-step adapter from the earlier experiment. If your existing artifacts use other
paths, substitute those paths consistently. Do not run `prepare-model` or `train`
for this evaluation. Missing artifacts are an error, not a reason to silently
substitute a model or train another adapter.

## Copy the configuration and prepare fresh data

The following creates fresh ignored paths and preserves the original 50-task
configuration, dataset, adapter, and results. It copies the experiment settings,
changing only the evaluation size; the checked-in scoring settings now include
30 seconds for startup and 10 seconds for execution.

```sh
.venv/bin/python scripts/check_environment.py
.venv/bin/python - <<'PY'
from pathlib import Path
from scripts.experiment import CONFIG, read_json, write_json
config = read_json(CONFIG)
assert config['training']['iters'] == 100
config['eval_size'] = 500
Path('outputs/mbpp-500').mkdir(parents=True, exist_ok=False)
write_json('outputs/mbpp-500/config.json', config)
PY
.venv/bin/python -m scripts.experiment --config outputs/mbpp-500/config.json \
  prepare-data --data data/mbpp-500
```

The pinned MBPP source is downloaded and its SHA-256 is verified. The manifest
records source URL/revision/hash, seed 42, prompt and standard-test format, file
hashes, and ordered IDs. Full-split preparation requires **exactly IDs 11–510,
once each**. Ordering uses the same seeded shuffle as the 50-task experiment;
its first 50 IDs match the earlier selection, but the full score is a different
sample. Training/validation remain disjoint IDs 601–974, with 354/20 examples.
Reference solutions and challenge tests are absent from `eval.jsonl`.

Verify the existing model and adapter, and that preparation preserved the
adapter's training/validation data. This reads existing weight hashes and training
metadata; it does not load the model or train.

```sh
.venv/bin/python - <<'PY'
from scripts.experiment import adapter_info, dataset_info, model_info, read_json
config = read_json('outputs/mbpp-500/config.json')
model = model_info('models/qwen3-4b-2507', config)
adapter = adapter_info('adapters/mbpp-lora', model)
training = adapter['training']
assert training['status'] == 'complete' and training['resume'] is None
assert training['configuration']['iters'] == 100
for key, value in config['training'].items():
    assert training['configuration'][key] == value, key
data = dataset_info('data/mbpp-500')
assert len(data['ids']['eval']) == 500
assert sorted(data['ids']['eval']) == list(range(11, 511))
for split in ('train', 'valid'):
    assert data['ids'][split] == training['dataset']['ids'][split]
    name = split + '.jsonl'
    assert data['files'][name] == training['dataset']['files'][name]
print('Verified unchanged base, original 100-step adapter, and full test split.')
PY
```

## Generate and score both models

Start Docker, then pull the documented immutable image if it is not already
local. Both commands resolve it to the same immutable local image ID. Keep the
repository code, environment, data, config, and Docker VM/resource allocation
unchanged between runs. Neither command changes base weights or adapter weights.

```sh
docker pull python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36
.venv/bin/python -m scripts.experiment --config outputs/mbpp-500/config.json \
  evaluate --model models/qwen3-4b-2507 --data data/mbpp-500 \
  --output outputs/mbpp-500/base.json
.venv/bin/python -m scripts.experiment --config outputs/mbpp-500/config.json \
  evaluate --model models/qwen3-4b-2507 --data data/mbpp-500 \
  --adapter adapters/mbpp-lora --output outputs/mbpp-500/adapted.json
.venv/bin/python -m scripts.experiment --config outputs/mbpp-500/config.json \
  compare --baseline outputs/mbpp-500/base.json \
  --adapted outputs/mbpp-500/adapted.json --output outputs/mbpp-500/comparison.json
```

Each evaluation first saves all 500 raw responses and extracted solutions, then
scores them. One greedy completion per task uses temperature 0, top-p 1, max 512
new tokens, and MLX seed 42 reset per problem. The same test-conditioned chat
prompt and ordered assertions are used for both models. Model file hashes,
revision/dtype, adapter hashes and recorded training metadata, dataset manifest
and examples, decoding settings, library versions, generation/scoring harness
hashes, image ID/reference, and resource limits are recorded.

Scoring uses a fresh non-root container per task, with no network/host mounts,
read-only root, dropped capabilities, no-new-privileges, one CPU, 256 MB memory/no
additional swap, 32 processes, 64 file descriptors, and a 16 MB temporary filesystem.
A **30-second startup deadline** covers Docker startup, delivery of the payload,
and trusted Python setup. Generated code has not executed when the readiness
marker is emitted. A separate **10-second execution deadline** begins upon that
marker, with an independent in-container wall timer; the runner also enforces
three CPU seconds and discards program output. Setup
code, solution, and all standard assertions execute inside that deadline.
`startup_seconds` and `execution_wait_seconds` are recorded separately per scored
task. The latter measures the host wait after readiness, including CLI exit delivery;
it is not pure Python runtime. If CLI exit delivery reaches the host deadline,
verified exited-container state supplies the outcome; the in-container timer
still bounds actual execution. A running container at that deadline is a timeout.
Assertion/runtime/syntax/resource failures, format errors, and execution timeouts
count as model failures. Readiness/startup failures, Docker state/exit mismatches,
or unconfirmed cleanup abort scoring and leave the result incomplete. The CLI
is stopped and the container forcibly removed in cleanup, including on timeouts.
Cleanup/state operations have their own bounded deadlines. Containers reduce
risk; use a disposable Docker VM without sensitive workloads as described in the
[legacy README](../README-legacy.md#4-evaluate-the-adapter-free-baseline).

## Rescore saved generations

If scoring aborts after generation completes, fix Docker and use a fresh output
filename. Rescore **both** saved runs with the same config/image/environment and
harness. No MLX, model directory, or adapter directory is needed for rescoring.
The config supplies scoring limits; generation settings and examples come from
the saved result, even if the config's other fields differ.

```sh
.venv/bin/python -m scripts.experiment --config outputs/mbpp-500/config.json \
  rescore --input outputs/mbpp-500/base.json \
  --output outputs/mbpp-500/base-rescored.json
.venv/bin/python -m scripts.experiment --config outputs/mbpp-500/config.json \
  rescore --input outputs/mbpp-500/adapted.json \
  --output outputs/mbpp-500/adapted-rescored.json
.venv/bin/python -m scripts.experiment --config outputs/mbpp-500/config.json \
  compare --baseline outputs/mbpp-500/base-rescored.json \
  --adapted outputs/mbpp-500/adapted-rescored.json \
  --output outputs/mbpp-500/comparison-rescored.json
```

Rescoring preserves responses, extracted code, task order, adapter metadata, and
original generation provenance. It verifies extraction and the generation-set
checksum, records the input file's SHA-256 and prior scoring protocol, and writes
new outcomes/settings/timings without modifying the input. Empty/malformed
completions remain format failures. Partial generation sets are rejected;
interrupted generation requires fresh evaluation. Completed legacy result files
can also be rescored, including the original 50-task runs. Old files lack a
separately recorded generation checksum, so the input-file hash establishes the
rescoring source and a new generation checksum is recorded. Never compare a
50-task score directly with a 500-task score.

Comparison rejects incomplete scoring, invalid/missing outcomes or inconsistent
summaries, task-order differences, obsolete startup-inclusive scores, baseline
adapter contamination, absent adapted selection, and differing protocols
(including model/data/generation provenance, image, execution limits, scorer
implementation, and libraries). Two freshly rescored runs can share the new
scoring protocol while retaining their original, matching generation provenance.

## Read and report the measured outcome

The comparison prints base and adapted successes/500 and percentages, the signed
percentage-point difference (`adapted - base`), net correct change, counts for
`fail_to_pass`, `pass_to_fail`, `unchanged_pass`, and `unchanged_fail`, and task IDs
for the two changed groups. The JSON also contains all 500 paired outcomes,
input-file paths/hashes, and interpretation. Negative and zero changes are valid.

Report those actual values from `comparison.json` (or the rescored comparison)
when the run completes. The earlier 38/50 and 39/50 results are context, not a
measurement of this full split. Public MBPP may have appeared in pretraining;
showing the standard assertions in prompts also conditions the evaluation. The
full split avoids relying on a small 50-task sample, but a single paired benchmark
still does not prove a general improvement. Repeats and independent held-out
benchmarks belong in follow-up work. Do not change training steps or hyperparameters
here. All data, weights, generations, and results remain local and Git-ignored.

See [Issue #7 validation](mbpp-500-validation.md) for performed implementation
checks and unperformed real-model runs. No post-merge verification is required.

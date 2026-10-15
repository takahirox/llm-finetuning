# llm-finetuning

This ProjectWeave-managed repository runs a small Python code-generation learning
experiment on Apple Silicon: generate with Qwen, measure a held-out MBPP baseline,
train a LoRA adapter, and compare the adapted model on exactly the same problems.
Improvement is an experimental question, not a guaranteed outcome.

For new training development, use the [controlled padding-loss experiment](docs/padding-loss-experiment.md):
two fresh 100-step adapters, a deterministic 294/80 internal split, validation
pass@1, and a provenance-checked legacy-versus-corrected comparison. Corrected
completion-only loss is now the training default. The workflow below retains the
historical 354/20 configuration for reproducibility; its official-test evaluations
are diagnostic evidence and must not become a repeated model-selection loop.

## Supported environment

- Apple Silicon Mac with native **arm64** Python and a terminal running without Rosetta.
- **macOS 14 or newer**, following the [MLX installation requirements](https://ml-explore.github.io/mlx/build/html/install.html).
- **Python 3.12**; `.python-version` selects 3.12.13 for reproducible setup.
- **uv 0.12.23**, the version used to generate and validate `uv.lock`.
- **MLX 0.32.3** and **MLX-LM 0.32.0**, plus their locked runtime dependencies.
  The Hub download client is pinned to 2.2.0. Local JSONL training needs no
  MLX-LM extras or `datasets` package; scoring uses Docker instead of `lm-eval`.

The lock is restricted to Apple Silicon macOS. Intel Macs, Linux, Windows, and
other Python minor versions cannot run the MLX stages. Data preparation, comparison,
and unit tests use only Python standard-library modules. The environment check
does not establish that a model or training run will fit in memory.

## Setup

Install the pinned [uv tool](https://docs.astral.sh/uv/getting-started/installation/)
if it is not already available, then add it to the current shell:

```sh
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Clone the repository and install the locked dependencies into `.venv`:

```sh
git clone https://github.com/takahirox/llm-finetuning.git
cd llm-finetuning
uv python install
uv sync --locked
.venv/bin/python scripts/check_environment.py
```

`uv python install` reads `.python-version`. `uv sync --locked` reads
`pyproject.toml` and `uv.lock`, creates the isolated environment, and refuses to
silently update an out-of-date lock. These steps download Python (if needed) and
Python packages, but no model weights or datasets. No environment activation is
required; using `.venv/bin/python` selects the project interpreter explicitly.

## Environment check

From the repository root, run:

```sh
.venv/bin/python scripts/check_environment.py
```

The check reports Python, OS, architecture, virtual-environment status, package
versions/imports, and Metal availability. It exits with status `0` on success or
`1` with recovery instructions on an unsupported or incomplete setup. It only
imports packages and queries Metal availability; it does not load models,
evaluate tensors, or run training. Once packages are installed, it can run offline.

If `.venv` does not exist, run `uv sync --locked` first. For platform diagnostics
before installation, `python3 scripts/check_environment.py` can run using only
the standard library on Python 3.9+; it reports unsupported versions/platforms
and skips package imports until the prerequisites are met. If it reports
`x86_64` on Apple Silicon, disable Rosetta for the terminal and recreate `.venv`
using native arm64 Python. To recreate it, move the old `.venv` outside the
repository, then run `uv sync --locked` again.

Keep model weights, dataset caches, checkpoints, and generated results in the
ignored `models/`, `data/`, `checkpoints/`, and `outputs/` directories (create them
only when needed). Local caches and secrets such as `.cache/` and `.env` are also
ignored. Keep useful source and configuration files outside artifact directories.

The hardware-independent diagnostic tests use the standard library:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Qwen / MBPP experiment

Run all commands from the repository root. The checked-in
[configuration](configs/experiment.json) records the model revision, data seed,
decoding, execution limits, and training settings. To change settings, copy it to
an ignored path (for example `outputs/custom-config.json`) and pass
`--config outputs/custom-config.json` **before** the subcommand at every stage.
Use new data, adapter, and result paths for each configuration. Commands refuse to
overwrite outputs; an interrupted evaluation remains marked `running` and cannot
be compared. If all generations were saved, use `rescore` with a fresh output
filename; otherwise rerun evaluation with a fresh filename.

### 1. Download and convert the model once

The selected model is
[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
revision `cdbee75f17c01a7cc42f958dc650907174af0554`, verified during implementation.
Its card identifies 4.0B parameters, the `qwen3` architecture, non-thinking chat
behavior, and Apache-2.0 licensing. Review the
[license and notices at that revision](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)
before use or redistribution; preserve the applicable license and notices.
“Base” below means this instruction-tuned checkpoint without our adapter.

MLX-LM 0.32.0 includes a
[native Qwen3 implementation](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/models/qwen3.py).
The workflow downloads the pinned official checkpoint to `.cache/huggingface/`
and uses MLX-LM's [conversion API](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/convert.py)
to save **non-quantized bfloat16** weights. No remote model code, quantization,
full fine-tuning, or weight fusion is enabled.

```sh
.venv/bin/python -m scripts.experiment prepare-model --model models/qwen3-4b-2507
```

Output: `models/qwen3-4b-2507/`, including `experiment_model.json` with source
revision, conversion settings, library versions, and SHA-256 file hashes.
Subsequent model stages verify these hashes, including weights; expect disk reads.
Missing models/adapters produce an error instead of an implicit remote load.

Plan for roughly 8 GB per weight copy plus download/conversion overhead and
substantial additional unified memory for activations, KV caches, and training.
The cache and converted copy both occupy disk. No exact RAM capacity is assumed.
Close memory-heavy applications, watch memory pressure, and start with shorter
sequences/fewer steps if necessary. Training and 50 generations may take many
minutes or longer; no runtime or memory-fit promise is made.

### 2. Generate a solution

```sh
.venv/bin/python -m scripts.experiment infer \
  --model models/qwen3-4b-2507 \
  --task 'Write a function square(n) returning n squared.' \
  --test 'assert square(3) == 9' \
  --output outputs/inference-base.json
```

The response is printed and saved with provenance. Inference does not execute it.
`--test` is optional and repeatable for user tasks. Defaults are one greedy
completion, temperature `0.0`, top-p `1.0`, maximum 512 new tokens, and MLX seed 42
reset for each problem. The same chat template and instruction are used for
training and evaluation.

### 3. Prepare disjoint data

```sh
.venv/bin/python -m scripts.experiment prepare-data --data data/mbpp
```

This downloads the original full MBPP JSONL from
[Google Research](https://github.com/google-research/google-research/tree/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp),
commit `f82046ba5aabbbb427dbfd38a254d26bff08b533`, and verifies SHA-256
`ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f`.
The [dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp)
identifies CC-BY-4.0; attribute MBPP to Austin et al., *Program Synthesis with Large
Language Models* (2021), and retain attribution for derived data.

Following the authors' task-ID split, IDs 601–974 are the only training source.
Sorted IDs are shuffled with Python's `random.Random(42)`; the first 20 become
internal loss-validation data and the other 354 become training data. A separate
shuffle with seed 42 selects the first 50 of test IDs 11–510. IDs 1–10 and
511–600 are unused. Increase `eval_size` to 500 in a copied configuration to use
the full official test split, then prepare a new directory and rerun both evaluations.

Outputs in `data/mbpp/`:

- `train.jsonl`, `valid.jsonl`: MLX-LM chat examples from training IDs only.
- `eval.jsonl`: held-out task descriptions, assertions, and setup code; no reference solutions.
- `validation_eval.jsonl`: reference-free scoring prompts for the internal validation IDs.
- `manifest.json`: source/checksum, exact ordered split IDs, seed, file hashes, and versions.

This is **test-conditioned MBPP**: the task's three standard assertions appear in
the prompt to specify function names and expected behavior, following MBPP's
published evaluation style. Training prompts contain only their own training
assertions; assistant completions are their reference solutions. No held-out
prompt, solution, or test is put into training or validation. Only standard
`test_list` assertions are scored, not challenge tests. Public MBPP may already
have appeared in model pretraining; this experiment cannot rule that out.

### 4. Evaluate the adapter-free baseline

Install/start Docker Desktop (or an equivalent Linux Docker engine). Pull this
immutable Python image once; scoring itself has no network access:

```sh
docker pull python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36
.venv/bin/python -m scripts.experiment evaluate \
  --model models/qwen3-4b-2507 --data data/mbpp \
  --output outputs/baseline.json
```

Omitting `--adapter` explicitly selects the base model. The command saves one raw
response, extracted Python solution, pass/fail status, and exit code per problem,
plus a final `pass_at_1_percent` summary. All generations are saved before scoring;
`generation_status: complete` and a generation SHA-256 make them reusable even if
scoring aborts. See the [500-problem procedure](docs/mbpp-500-evaluation.md).
A problem passes only when all its
assertions finish successfully. Failed assertions, syntax/runtime/resource errors,
format errors, and wall timeouts score zero. Pass@1 is the percentage of problems
passing all assertions with one completion, not the percentage of individual
assertions passed. Multiple or incomplete fences are rejected; a single Python
fence is accepted. Plain code is otherwise used unchanged.

Generated code is **untrusted**. Scoring requires Docker and has no host-execution
fallback. Each problem gets a fresh non-root container with no network, host mounts,
or credentials; a read-only root filesystem; dropped capabilities; no-new-privileges;
32-process and 64-file-descriptor limits; a 16 MB temporary filesystem; 256 MB RAM
and no additional swap; one CPU; three CPU seconds; and a ten-second code-execution
wall deadline. A separate 30-second startup
deadline covers container startup, payload delivery, and trusted Python setup.
Python signals readiness before executing setup code, the solution, or assertions;
only then do independent host and in-container execution timers start. Startup
and execution-wait timings are recorded per task; verified container exits avoid
counting delayed Docker CLI exit delivery as a model timeout. Output is discarded
to avoid unbounded logs. Setup,
solution, and assertions execute in that order; Python random/hash seeds are fixed.
The container is forcibly removed after every attempt, including wall timeouts.
Container state is checked against the CLI exit before accepting
an outcome; startup/readiness, Docker state, and cleanup errors abort scoring.
Docker infrastructure failures abort the run instead of masquerading as model errors.

Containers reduce risk but are not a proof of security. Use a disposable Docker VM
without sensitive workloads for hostile code. The benchmark harness is intended
for ordinary model completions, not adversarial submissions that forge process exit
status or tamper with tests. OS scheduling and resource limits can affect results;
keep the same limits, image, and libraries across both evaluations. `--image`
selects an alternative pre-pulled Python image; its resolved immutable ID and
reference are recorded and checked during comparison.

### 5. Train a LoRA adapter

```sh
.venv/bin/python -m scripts.experiment train \
  --model models/qwen3-4b-2507 --data data/mbpp \
  --adapter-output adapters/mbpp-lora
```

The wrapper invokes the pinned MLX-LM LoRA trainer through a project-owned loss
hook using only `train.jsonl` and `valid.jsonl`. Padding and prompt tokens are
excluded by position; real completion/EOS tokens retain loss. No installed package
files are patched. `--mask-variant legacy` is an explicit diagnostic option for the
[controlled experiment](docs/padding-loss-experiment.md), which explains the
upstream off-by-one error. Defaults: seed 42, Adam learning rate `1e-5`, 100 optimizer
iterations (not epochs), batch size 1, last 8 transformer layers, rank 8, scale 20,
dropout 0, attention `q_proj`/`v_proj` targets, 1024-token sequence limit, gradient
checkpointing, and masked prompt loss. Four validation batches run every 25 steps;
reports run every 10 and checkpoints save every 25. Longer examples may be
truncated at the sequence limit. The checked-in config makes these choices explicit.

Outputs: `adapters/mbpp-lora/adapters.safetensors`, periodic checkpoints,
`adapter_config.json`, `training_config.json`, `training.log`, and
`experiment_training.json` (model/data/configuration, mask variant, code/dependency
versions, timestamps, completion status, adapter hashes), and `trainer_runtime.json`
(verified loss-hook execution and tokenization audit). Fully truncated completions
fail explicitly. Base weights are preserved. Inspect the log
for progress/failures. Inference/evaluation accepts only completed, matching adapters.

To continue from saved adapter weights into a new directory:

```sh
.venv/bin/python -m scripts.experiment train \
  --resume-adapter adapters/mbpp-lora \
  --adapter-output adapters/mbpp-lora-continued
```

MLX-LM resumes **weights**, not optimizer state, RNG position, or data cursor;
this is a new training segment, not a bit-identical interrupted-run continuation.
Keep model, layer count, and LoRA parameters compatible. For a two-step smoke run,
copy the config, set `training.iters` to 2, shorten report/eval/save intervals, and
use fresh adapter/results paths. Do not interpret smoke-run scores as evidence of
robust improvement.

### 6. Reuse the adapter, evaluate, and compare

```sh
.venv/bin/python -m scripts.experiment infer \
  --adapter adapters/mbpp-lora \
  --task 'Write a function square(n) returning n squared.' \
  --test 'assert square(3) == 9' \
  --output outputs/inference-adapted.json
.venv/bin/python -m scripts.experiment evaluate \
  --model models/qwen3-4b-2507 --data data/mbpp \
  --adapter adapters/mbpp-lora --output outputs/adapted.json
.venv/bin/python -m scripts.experiment compare \
  --baseline outputs/baseline.json --adapted outputs/adapted.json \
  --output outputs/comparison.json
```

Comparison prints/saves both percentages, the signed absolute percentage-point
difference (`adapted - baseline`), net correct change, four transition counts
(fail→pass, pass→fail, unchanged-pass, unchanged-fail), and changed task IDs. The
JSON also contains every problem's before/after outcome and change flag. Negative
or zero differences are valid outcomes. Comparison rejects
incomplete runs, adapter-contaminated baselines, missing adapted selections, or
changes in model hashes/revision, ordered examples, dataset files, prompt/scorer
implementation, decoding, execution settings, image, or library versions. Results
retain raw generations and useful provenance so a score can be audited. Old
startup-inclusive scores must be rescored before comparison with this scorer.

For the complete official test split using the existing 100-step adapter, follow
the [copy-and-run 500-problem procedure](docs/mbpp-500-evaluation.md), including
rescoring commands. Do not retrain or overwrite the original 50-task outputs.

All weights, downloaded data, adapters, and results belong in the ignored local
paths shown above. Do not add them to Git. Useful reusable configuration belongs
in `configs/`; experiment-specific copies and outputs belong in ignored paths.

## Validation

```sh
uv sync --locked
uv pip check
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m unittest discover -s tests -v
RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -p test_loss_masking.py -v
.venv/bin/python -m compileall -q scripts tests
git diff --check
```

Unit tests use small fixtures/mocks and need neither MLX imports nor Docker.
They cover split separation, data/model/adapter integrity, prompts, formatting,
base/adapter selection, trainer invocation, scoring errors/timeouts, and comparison.
The separately enabled native mask tests require the supported MLX environment;
they exercise token-level masks/gradients and a tiny one-step LoRA model, without
downloading Qwen weights or running MBPP scoring.
See [original implementation validation](docs/experiment-validation.md) and
[padding-loss validation](docs/padding-loss-validation.md) for measured checks
and unperformed model experiments. No 4B benchmark score or learning improvement
is claimed by this implementation. No post-merge verification is required.

## Collaboration guidance

- [Development flow and English collaboration convention](docs/development-flow.md)
- [AI review and merge guidelines](docs/review-guidelines.md)
- [Issue template](.github/ISSUE_TEMPLATE/issue.md)
- [Pull request template](.github/pull_request_template.md)
- [AI contributor entry point](AGENTS.md)

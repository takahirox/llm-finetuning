# llm-finetuning

A reproducible Apple Silicon positive-control experiment: fine-tune the original
4-bit **Qwen2.5-0.5B-Instruct** with LoRA to extract five shipment fields as JSON.
The fixed synthetic task tests whether local fine-tuning can improve this workflow;
it does not establish general coding ability or real-world freight competence.

The [upstream author reports 71.3% → 100.0%](https://github.com/openatlaspro-AI/mlx-lora-finetune/blob/24f663d1a276203356d957d6e340ee2182ff83d8/results/comparison.md)
all-fields-exact accuracy on 150 examples. Those numbers belong to the upstream
experiment. See our [execution record](docs/freight-validation.md) for actual local
measurements and [reproduction notes](docs/freight-reproduction.md) for the pinned
protocol, intentional differences, licensing, and interpretation.

## Setup

Use a native arm64 Apple Silicon Mac, macOS 14+, Python **3.12.13**, and
**uv 0.12.23**. MLX needs Metal access. Allow several GB of available unified
memory and disk; memory fit and runtime depend on the machine. Our lock pins
MLX 0.32.3, MLX-LM 0.32.0, Pydantic 2.12.3, and all transitive dependencies.
The lock targets Apple Silicon macOS; other platforms can inspect the source but
cannot run MLX training. This workflow needs no Docker or code execution sandbox.

Install [uv 0.12.23](https://docs.astral.sh/uv/getting-started/installation/) if needed:

```sh
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
source "$HOME/.local/bin/env"

git clone https://github.com/takahirox/llm-finetuning.git
cd llm-finetuning
uv python install
uv sync --locked
.venv/bin/python scripts/check_environment.py
```

The environment check imports packages and checks Metal without loading a model.
It diagnoses Rosetta, unsupported Python/macOS, missing dependencies, and version
mismatches. Network access is needed for setup/model download, not synthetic data.

## Run the fixed experiment

Run from the repository root. Use **new output paths** for every new run. Commands
refuse to overwrite any existing destination, including interrupted runs. No
command deletes old artifacts. Weights, caches, data, adapters, and raw results
remain in ignored directories; never add them to Git.

```sh
# Deterministic, byte-verified upstream splits: 600 train / 150 valid / 150 test.
.venv/bin/python -m scripts.freight prepare-data --data data/freight

# Download original 4-bit weights at an immutable revision, with no conversion.
.venv/bin/python -m scripts.freight prepare-model --model models/qwen25-05b-4bit

# Fresh seed-0 adapter: 600 updates, batch 4, Adam 1e-4, 8 layers, rank 8.
.venv/bin/python -m scripts.freight train \
  --model models/qwen25-05b-4bit --data data/freight \
  --adapter adapters/freight-seed0

# Training progress is saved here (inspect from another terminal if desired):
tail -n 20 adapters/freight-seed0/training.log

# Identical test examples, chat template, greedy decoding, and scoring.
.venv/bin/python -m scripts.freight evaluate \
  --model models/qwen25-05b-4bit --data data/freight \
  --output outputs/freight-base
.venv/bin/python -m scripts.freight evaluate \
  --model models/qwen25-05b-4bit --data data/freight \
  --adapter adapters/freight-seed0 --output outputs/freight-adapted

# Validates saved rows/provenance and recomputes every score before comparison.
.venv/bin/python -m scripts.freight compare \
  --base outputs/freight-base/result.json \
  --adapted outputs/freight-adapted/result.json \
  --output outputs/freight-comparison
cat outputs/freight-comparison/comparison.md
```

Training sees only a copied train/validation pair. The held-out test records are
never used for training, adapter selection, or prompt examples. Use the final
600-step adapter, regardless of test outcomes. Do not tune the fixed test/scorer
or repeat model selection against these examples.

Each evaluation saves raw responses, per-example IDs/gold/field outcomes,
counts, fractions, timestamps, runtime, package versions, code/data/model hashes,
and adapter training provenance. A comparison reports JSON validity,
all-fields-exact accuracy, origin/destination/weight_lbs/mode/on_time accuracy,
and absolute percentage-point deltas. `json_valid` follows the upstream parser
and accepts embedded JSON; `strict_json_valid` additionally requires a bare JSON
object. Neither guarantees schema validity. All-fields scoring follows upstream
canonicalization and tolerates extra keys and some value coercions.

Completed receipts use format version 1. Incomplete, reordered, changed-data,
changed-code, incompatible-training, or different-protocol receipts are rejected.
To compare an old run after changing experiment code, check out its recorded
source revision in another worktree; do not edit historical receipts.

## SQL code-generation control

A separate [SQL LoRA experiment](docs/sql-control.md) provides pinned data/model
revisions, fresh rank-8 MLX training, paired base/adapter metrics, and a small
SQLite execution sanity check. The implementation is tested; the full SQL model
experiment has **not been performed**, so no SQL accuracy gain is claimed.
This narrow code-generation control does not establish general programming ability.

## UI/UX

Apply these principles to existing CLI commands, messages, and examples, and to
any future graphical interface:

1. **Keep it simple.** Show only necessary information and actions; avoid
   unnecessary UI elements.
2. **Show, don't tell.** Prefer layout, spacing and visual hierarchy over
   explanatory labels or prose to communicate structure.
3. **Intuitive and predictable.** Keep appearance and behavior consistent with
   user expectations; don't interrupt the user's work.

Preserve necessary control labels, accessible names, error messages, and
information needed to use the interface. Simplification must preserve meaningful
consent, destructive-action safeguards, and necessary warnings.

## Checks

```sh
uv lock --check
uv pip check
.venv/bin/python -m unittest discover -s tests -v
# Also run preserved legacy native MLX checks on a supported host:
RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
git diff --check
```

The new focused tests cover deterministic upstream byte equality, split integrity,
schema/canonicalization, prompt parity, scorer semantics, aggregation, model
fingerprints, and saved-result compatibility. Tests never download a model.

## Legacy work and collaboration

Qwen3/MBPP/HumanEval/data-quality work is retired as the primary experiment.
Its [previous workflow](README-legacy.md), source, configs, tests, and historical
docs remain intact because [PR #12](https://github.com/takahirox/llm-finetuning/pull/12)
is still open and imports them. [PR #14](https://github.com/takahirox/llm-finetuning/pull/14)
is merged. This intentional archival decision avoids silently breaking the open
PR; removal or relocation requires a coordinated follow-up. Existing ignored
artifacts were not deleted, moved, or overwritten.

Contributors should follow [AGENTS.md](AGENTS.md), the
[development flow](docs/development-flow.md), and
[review guidelines](docs/review-guidelines.md), using the
[Issue template](.github/ISSUE_TEMPLATE/issue.md) and
[PR template](.github/pull_request_template.md). Third-party reference code retains
its [MIT notice](THIRD_PARTY_NOTICES.md); model licensing is separate.

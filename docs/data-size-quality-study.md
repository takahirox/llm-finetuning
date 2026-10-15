# Controlled Python data studies (Issue #13)

This extends the existing Qwen/MLX-LM experiment, preserving `prepare-data` and the
historical MBPP workflow. PR #12 was **open and unmerged** when this implementation
started on 2026-10-15; its HumanEval evaluator was not copied or overwritten.
The previous MBPP and HumanEval scores remain historical diagnostics.

## Sources and permitted use

The selected training and new heldout source is
[KodCode/KodCode-Light-RL-10K](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K),
by Zhangchen Xu, Yang Liu, Yueqin Yin, Mingyuan Zhou, and Radha Poovendran,
[KodCode (2025)](https://arxiv.org/abs/2503.02951). The publisher's
[dataset card](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K/blob/dcf78a8bbba9a613b596ce993c4921a38687dfcc/README.md)
licenses it under **CC BY-NC 4.0**. This workflow is for noncommercial research.
Preserve attribution, the source/license link, and an indication of modifications
when sharing permitted derivatives; commercial use requires separate permission.
The [license terms](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)
apply to the corpus independently of this repository's code. The corpus is
synthetic and derives from several task families; its license does not establish
independent authorship or remove possible upstream rights/semantic overlap.
No corpus, generated split, or model artifact is committed here.

The canonical pinned download is the publisher's `data/train-00000-of-00001.parquet`
file at revision `dcf78a8bbba9a613b596ce993c4921a38687dfcc`. Its SHA-256 is
`cd777b078dd700d7e87c2f42ae2cb1d28de91625da92066fffda219eb9193c93`.
It has 10,000 rows with `question_id`, `question`, `solution`, `test`, subset,
difficulty, and generation metadata. We use `solution`, not `r1_solution` or
reasoning conversations. The checked-in
[source lock](../configs/data-study-sources.json) records URLs, revisions,
checksums, license declarations and attribution for every download.

For exclusions only, the workflow downloads **all 974 MBPP** tasks from Google
Research at `f82046ba5aabbbb427dbfd38a254d26bff08b533` (CC BY 4.0;
[Austin et al.](https://github.com/google-research/google-research/tree/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp))
and **all 164 HumanEval** tasks at
`6d43fb980f9fee3c892a914eda09951f772ad10d` (MIT;
[OpenAI/Chen et al.](https://github.com/openai/human-eval/tree/6d43fb980f9fee3c892a914eda09951f772ad10d)).
They never become new study holdouts. Checksums cover the raw MBPP JSONL and
compressed HumanEval JSONL bytes, respectively.

TACO and APPS were considered. Their larger competitive-programming formats need
more adaptation; TACO's card also reports mixed upstream terms and unresolved
HackerRank rights. KodCode's small, directly accessible function tasks provide a
narrower integration with the current scorer, subject to its noncommercial limit.

## Preparation and audit

```sh
uv sync --locked --extra study
.venv/bin/python -m scripts.data_study prepare --download \
  --output data/kodcode-study-v1
.venv/bin/python -m scripts.data_study verify --study data/kodcode-study-v1
```

Downloads go under `datasets/issue13-sources/`. Preparation verifies every byte
checksum before parsing Parquet using the optional pinned PyArrow dependency.
It never imports or executes external solutions/tests on the host. Artifacts must
be below ignored `data/`, `datasets/`, `runs/`, `outputs/`, or `adapters/` paths.
Existing outputs are rejected. Do not reuse a path for a new study or edit an old
manifest to change its meaning.

The reproducible static eligibility rules are:

- Require nonempty question, solution and tests; question 40–12,000 characters,
  solution 30–12,000, tests at most 30,000; reject Python syntax errors,
  placeholder bodies, classes/async solutions, unsupported dependencies and
  obvious dynamic execution/file/input operations.
- Accept an explicit small stdlib import allowlist in `scripts/data_study.py`.
  Accept plain undecorated zero-argument `test_*` functions with at least two
  assertions. Reject pytest fixtures, pytest dependencies, aliases from
  `solution`, module imports of `solution`, and other unsupported test structures.
  Remove `from solution import ...`; call every accepted test function in the
  existing Docker assertion scope. No framework or dependency installation runs
  inside the scoring container.
- Normalize question words to lowercase five-word shingles. Normalize solution
  ASTs by removing docstrings and function/variable naming and annotation
  differences, preserving constants, builtins and structure; use seven-token
  AST shingles. Reject an example if either text Jaccard similarity is at least
  0.80 or code similarity is at least 0.85 to any historical or retained example.
  Process immutable source IDs in sorted order. An exact inverted-index candidate
  filter preserves these thresholds; it is not approximate LSH. This intentionally
  conservative code-only exclusion also removes some distinct tasks with trivial
  shared solutions. Historical checks include the prompt plus canonical solution
  for HumanEval. Split after global deduplication, so the same checks also apply
  between proposed training, validation, development and confirmation items.

`audit.json` contains counts and subset/difficulty, length, dependency and
assertion diagnostics. `rejections.jsonl` identifies each excluded source ID,
reason, and matched historical/corpus ID for overlaps. `samples.jsonl` records
immutable source IDs, source-row and sample hashes and diagnostics. There is one
source row per retained task, with no oversampling to reach a requested size.
Static testability and assertion count are proxies, not proof of solution/test
correctness; unsupported tasks are excluded rather than silently scored as wrong.
Semantic paraphrases and public pretraining contamination remain uncertain.

Selection uses SHA-256 ordering of `[selection_seed, sample_id]`, default seed 42.
First reserve 80 loss-validation, 100 development and 100 confirmation records.
The remaining order supplies prefix-nested **300, 1,000 and 3,000** training sets.
Separately select **300 quality** records from the identical remaining pool:
require at least five assertions, question length 80–2,400 and solution length at
most 2,400; take round-robin from `(source subset, difficulty)` buckets in seeded
order. The baseline 300 is statically eligible but has no additional quality
criterion or balancing. Compare `size-300` with `quality-300` to study this combined
selection intervention at constant count. The three `size-*` groups isolate
quantity within one selection rule; comparing `quality-300` to `size-3000` does
not isolate quality. The groups may share training examples, by design.

The root `study.json` locks source/rule/configuration fingerprints, all sample
IDs, relationships and artifact hashes **before any model scoring**. Training
consumes only MLX conversation `train.jsonl` and `valid.jsonl`; the validation
set is identical across training subsets. Heldout JSONL has no solution or source
metadata. Prompts include task text and required signatures, **no hidden tests**.
Both new holdouts are from a task pool not used in previous project tuning.
They are synthetic internal splits, not a claim of independently authored or
pretraining-clean benchmarks. Do not inspect confirmation outcomes during
selection. The lock is an audit safeguard, not access control against an operator
who can read local files.

If coverage is insufficient, preparation saves the audit and rejection ledger
and exits with the actual count. It never pads, duplicates, or labels an
undersized set as 3,000. A failed output directory must remain distinct from a
subsequent corrected preparation.

## Training policy and exposure

The [study config](../configs/data-study.json) retains nonquantized bfloat16
`Qwen/Qwen3-4B-Instruct-2507` at
`cdbee75f17c01a7cc42f958dc650907174af0554`, Adam, learning rate **1e-6**, eight
LoRA layers, rank 8, scale 20, q/v projection keys, zero dropout, corrected PAD
mask, batch 1 and completion-only loss. Maximum sequence length is preregistered
as **4096**, enlarged from the historical 1024 for these longer tasks. No
learning-rate sweep is performed. The original decoding (greedy, seed 42,
512-token limit) and Docker image/resource settings are fixed, even though this
may truncate some solutions. Training starts from fresh adapters; resume and
legacy masking are rejected for studies.

Default policy is **one fixed epoch**, seed 43: 300, 1,000 and 3,000 optimizer
steps, plus 300 for the quality comparison. Every selected example must be
observed exactly once. This matches epochs, not total steps, tokens or wall time;
larger sets consume more compute. The same-size contrast still varies token
lengths and composition. For replication, preregister seeds **43/44/45**, using
config copies that change only `training.seed` and separate output paths.

```sh
.venv/bin/python -m scripts.experiment --config configs/data-study.json train \
  --data data/kodcode-study-v1/size-300 --adapter-output adapters/study-size300-s43
# Repeat for size-1000, size-3000 and quality-300 with distinct adapter paths.
```

The existing model path is `models/qwen3-4b-2507`; if absent, use the existing
`prepare-model` command with this config. Never reconvert over historical weights.
Training writes configuration, source/study fingerprints, code/library versions,
logs, runtime evidence and adapter hashes into its fresh output directory.
`exposure.jsonl` records actual trainer order, sample ID, original and retained
tokens, target tokens and truncation for every presentation. Runtime totals
include coverage, reuse, input/target tokens, discarded tokens, wall time, and
an order checksum. Padding is excluded. Ambiguous identities after truncation,
empty completion targets, distributed training, accumulation and batch sizes
other than one fail closed. The pinned trainer shuffles length-sorted examples;
the trace records its actual order, including effects of validation RNG usage.
A completed fixed-epoch run with missing/repeated examples is rejected.

For a separate fixed-compute protocol, copy the config **before preparing a new
study**, replacing `study` with `{"policy":"fixed-compute","steps":3000}`.
Use the same selection seed/rules; verify identical subset IDs across protocols.
This matches optimizer steps, not FLOPs, tokens or wall time. Small sets repeat;
large sets have less reuse. A 50-step run cannot represent exposure to 3,000
examples. Reports separate incomplete-pool contrasts from size contrasts and
record actual coverage; never infer a size effect from an unsampled pool.
Do not change exposure policy after preparation.

## Evaluation, choice freeze and reporting

Use the existing Docker setup from the README. All generations run through the
same bounded, networkless, read-only, unprivileged container scorer. A Docker
infrastructure failure aborts scoring; format errors and model failures remain
separate. Base generation explicitly loads with no adapter.

```sh
.venv/bin/python -m scripts.experiment --config configs/data-study.json evaluate \
  --data data/kodcode-study-v1 --split development --output outputs/study-base-dev.json
.venv/bin/python -m scripts.experiment --config configs/data-study.json evaluate \
  --data data/kodcode-study-v1 --split development \
  --adapter adapters/study-size300-s43 --output outputs/study-size300-s43-dev.json
.venv/bin/python -m scripts.data_study report --baseline outputs/study-base-dev.json \
  --adapted outputs/study-size300-s43-dev.json --output outputs/study-dev-report.json
```

Pass all applicable adapter result files to `report --adapted` for the full
study. Each base/adapter comparison requires identical questions/order, base
weights, decoding, prompt, scorer, Docker image, libraries and study fingerprint.
It reports passed/total, pass@1, delta versus base, paired fail→pass/pass→fail
IDs, failures/timeouts/format errors, effective exposure and corpus diagnostics.
The study report separates size and same-size quality contrasts, reports mean
and sample SD over distinct training seeds, and rejects training confounders or
duplicate seed/selection results. Paired normal 95% delta intervals are
**descriptive, unadjusted** and unreliable with small samples/few discordant
pairs (including degenerate zero-width intervals). One seed has no estimated
seed variability. Do not treat many development comparisons as independent
confirmation or claim general improvement from this internal split.

After development, freeze the exact chosen adapters and written decision:

```sh
.venv/bin/python -m scripts.data_study freeze --study data/kodcode-study-v1 \
  --baseline outputs/study-base-dev.json --adapted outputs/study-size300-s43-dev.json \
  --decision 'Frozen selected adapter and comparison based on development; no further tuning.'
.venv/bin/python -m scripts.experiment --config configs/data-study.json evaluate \
  --data data/kodcode-study-v1 --split confirmation --output outputs/study-base-final.json
.venv/bin/python -m scripts.experiment --config configs/data-study.json evaluate \
  --data data/kodcode-study-v1 --split confirmation --adapter adapters/study-size300-s43 \
  --output outputs/study-size300-s43-final.json
.venv/bin/python -m scripts.data_study report --baseline outputs/study-base-final.json \
  --adapted outputs/study-size300-s43-final.json --output outputs/study-final-report.json
```

`choice.json` binds development result hashes, selected adapter provenance,
configuration and decision. Confirmation permits only the base and declared
adapters with the same model/config/scorer/libraries. An exclusive marker consumes
one attempt per model **before generation**, including failed attempts. It does
not allow fresh-generation retries, another freeze, or more development scoring
after confirmation starts. Infrastructure recovery may use existing `rescore`
on the saved complete generations, documenting the reason; it never authorizes
new generations. Preserve these files and the original study. Locking does not
make a public task set inaccessible to the model's pretraining corpus.

## Small smoke check

CPU-only invariants need only stdlib:

```sh
.venv/bin/python -m unittest discover -s tests -p test_data_study.py -v
```

For a small preparation, copy the study config to `runs/study-smoke.json`, change
`validation_size` and `eval_size` to 2, then run:

```sh
.venv/bin/python -m scripts.data_study --config runs/study-smoke.json prepare \
  --download --sizes 3 5 --output data/study-smoke
.venv/bin/python -m scripts.data_study verify --study data/study-smoke
RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -p test_loss_masking.py -v
```

This still audits the complete 74 MB source for leakage before selecting tiny
subsets. The native tests use tiny repository-owned weights, real tokenization,
LoRA, optimization and exposure tracking, without downloading Qwen weights.
A real Qwen smoke can train `data/study-smoke/size-3` with the smoke config and
evaluate its **development** split. Keep smoke artifacts separate from the full
study and do not open its confirmation set.

See [implementation validation](data-study-validation.md) for actual checks and
coverage. Full Qwen training and accuracy measurements are future local work;
merging does not start training or publication. No post-merge check is required.

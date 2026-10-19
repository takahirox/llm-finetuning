# Issue #15 execution record

The full local experiment completed on 2026-10-19 UTC (2026-10-20 JST), using
Apple M4 Max, 64 GiB unified memory, macOS 26.6.2, native arm64 Python 3.12.13,
uv 0.12.23, and the committed dependency lock. This is an independently trained
local adapter and 300 fresh generations, not the upstream adapter or saved scores.

**Result: exact-record accuracy increased from 107/150 (71.33%) to 150/150
(100.00%), an absolute gain of 28.67 percentage points.** The gain reproduces
the upstream report at its published precision on this fixed synthetic task.
It does not demonstrate general freight extraction or coding competence.

## Measured results

| Metric | Base | Fresh LoRA | Absolute delta |
| --- | --- | --- | --- |
| JSON object valid, upstream parser | 150/150 (100.00%) | 150/150 (100.00%) | +0.00 pp |
| Strict bare JSON object valid | 150/150 (100.00%) | 150/150 (100.00%) | +0.00 pp |
| All five fields exact | 107/150 (71.33%) | 150/150 (100.00%) | +28.67 pp |
| origin | 150/150 (100.00%) | 150/150 (100.00%) | +0.00 pp |
| destination | 150/150 (100.00%) | 150/150 (100.00%) | +0.00 pp |
| weight_lbs | 150/150 (100.00%) | 150/150 (100.00%) | +0.00 pp |
| mode | 120/150 (80.00%) | 150/150 (100.00%) | +20.00 pp |
| on_time | 133/150 (88.67%) | 150/150 (100.00%) | +11.33 pp |
| Field micro-average | 703/750 (93.73%) | 750/750 (100.00%) | +6.27 pp |

There were 43 exact-record gains and zero regressions. Base errors comprise 30
mode errors and 17 on_time errors, with four records failing both. All freshly
adapted outputs are correct under the unchanged upstream scorer. The upstream
author's reported base field values and before/after metrics match ours when
rounded to one decimal. No test, prompt, training recipe, or scorer was adjusted
after viewing outcomes.

An independent recount parsed every raw response as bare JSON and compared
fields without calling the project scorer. Applying the reference's mode case
normalization reproduced all saved counts. A first raw case-sensitive recount
exposed three otherwise-correct lowercase modes (`intermodal`, `intermodal`,
`parcel`): 117 direct mode matches become 120 under the already-frozen scorer.
This is expected reference normalization, not a post-result scoring change.

## Commands actually executed

Setup: `uv sync --locked`, followed by the native environment check. Adding the
exact upstream schema required Pydantic 2.12.3; `uv lock` updated the dependency
graph and `uv sync --locked` installed it. The existing MLX/MLX-LM pins stayed fixed.

```sh
.venv/bin/python -m scripts.freight prepare-data --data data/freight-issue15
.venv/bin/python -m scripts.freight prepare-model --model models/qwen25-05b-4bit-issue15
.venv/bin/python -m scripts.freight train \
  --model models/qwen25-05b-4bit-issue15 --data data/freight-issue15 \
  --adapter adapters/freight-issue15-seed0-final
.venv/bin/python -m scripts.freight evaluate \
  --model models/qwen25-05b-4bit-issue15 --data data/freight-issue15 \
  --output outputs/freight-issue15-base-final
.venv/bin/python -m scripts.freight evaluate \
  --model models/qwen25-05b-4bit-issue15 --data data/freight-issue15 \
  --adapter adapters/freight-issue15-seed0-final --output outputs/freight-issue15-adapted-final
.venv/bin/python -m scripts.freight compare \
  --base outputs/freight-issue15-base-final/result.json \
  --adapted outputs/freight-issue15-adapted-final/result.json \
  --output outputs/freight-issue15-comparison-final
```

Fresh training ran 600 optimizer iterations from 21:20:46 to 21:22:59 UTC:
**132.96 seconds** wall time including wrapper overhead. Base generation took
22.21 seconds; adapted generation 25.29 seconds. Peak memory was not measured;
64 GiB is the host's capacity, not a minimum requirement or measured usage.
No runtime guarantee is inferred from these observations. Training's initial
validation loss was 3.192; the final validation loss was 0.159. Loss did not
select an adapter: the final 600-step adapter was used as specified.

## Evidence kept in Git

- [freight-results.json](freight-results.json): compact summary export, both raw
  receipt SHA-256 hashes, input/source/software fingerprints, recipe, seed,
  timestamps, durations, model revision, adapter hashes, counts and deltas.
- [freight-outcomes.csv](freight-outcomes.csv): all 150 paired content IDs,
  JSON-valid/strict-valid/exact-record flags, and every base/adapted field outcome
  (1 = correct, 0 = incorrect). IDs and row order regenerate from the fixed test
  data. This small synthetic summary contains no private shipment data.
- [configs/freight.json](../configs/freight.json): immutable reference/model
  revisions, upstream data hashes, recipe, decoding and scorer identifier.

The compact JSON is a summary export, not an input to the CLI `compare` command;
that command consumes the complete evaluation receipts. Full raw generations,
complete effective training config/log, model files, train/validation copies and
all adapter checkpoints remain at the ignored paths above on this machine. They
are not part of the Git checkpoint and may be unavailable in a later worktree.
Regenerate them using new paths; do not overwrite historical artifacts.

The run recorded initial Git HEAD `8f58dd5fa829ca7ea529e69826f7b0d9fc1b4380`
plus SHA-256 hashes of the then-uncommitted implementation. Those exact source
hashes, not HEAD alone, identify the code used. They match the implementation
committed with this report; no experiment source changed during or after the run.

## Checks and limitations

- The staged whitespace check found one extra blank line at the end of the
  vendored scorer after the first successful run. It was removed without
  changing any scoring logic. The entire fixed training/evaluation was repeated
  into new `*-final` paths so this report's source fingerprints identify the
  committed files. The first run's artifacts remain intact. No data, prompt,
  scorer behavior, or recipe changed between runs. Both runs produced identical
  adapter weight hashes and identical raw responses for all 300 generations.
- `RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -v`:
  **71 passed, zero skipped**, including ten new freight tests and the preserved
  native MLX checks. The first full run found one stale mocked dependency list
  in the environment test; adding the new Pydantic pin to that mock fixed it.
- `uv lock --check`, `uv pip check`, native environment check, Python syntax,
  documentation links and `git diff --check`: passed.
- All three locally generated JSONL files match the pinned upstream SHA-256
  hashes. All 900 IDs are unique across splits. Source copies and scoring
  functions were checked against the pinned reference.
- Training's effective config matched the requested recipe, including fresh
  initialization/no resume. Model/data/source hashes stayed unchanged. Only
  train/valid files exist in the trainer's data directory. Every full-chat
  record tokenizes below 512 tokens: maxima train 172, valid 173, test 173.
- All saved scores and paired IDs were validated and independently recounted;
  CSV totals, receipt hashes, and source fingerprints were checked. Generated
  model/data/adapter/raw-result paths are confirmed ignored by Git.
- PR #12 is open and PR #14 is merged. Existing coding pipeline dependencies
  were inspected and retained as [legacy](../README-legacy.md); the scope
  decision is explained in [reproduction notes](freight-reproduction.md).

One fixed training seed and one synthetic 150-example test are the scope of this
replication. A separate transfer dataset, additional seeds, real shipment data,
and peak-memory profiling were not run. Software/model version differences that
cannot be reconstructed from upstream are documented in the reproduction notes,
even though the measured scores match. No outstanding implementation check or
required post-merge verification remains. No independent review is claimed.

This node intentionally did not push, create a PR, merge, or comment, following
its execution contract. A later authorized publication step can use this local
commit and report for the Issue-linked PR.

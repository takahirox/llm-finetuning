# Independent three-seed LoRA validation (Issue #11)

The fixed recipe did not improve mean functional correctness over base on the preregistered independent cohort. The measurements do not demonstrate a reproducible coding improvement from this additional LoRA training.

## Measured results

These are new measurements of the base and three fresh adapters on the same
HumanEval problems. The historical 80% base score belongs to MBPP and is not
a baseline for this experiment. No evaluation outcome changed the recipe.

| Model | Primary correct / 159 | Primary pass@1 | Change vs base | Full correct / 164 | Full pass@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 134/159 | 84.28% | +0.00 pp | 139/164 | 84.76% |
| LoRA seed 43 | 133/159 | 83.65% | -0.63 pp | 138/164 | 84.15% |
| LoRA seed 44 | 134/159 | 84.28% | +0.00 pp | 139/164 | 84.76% |
| LoRA seed 45 | 134/159 | 84.28% | +0.00 pp | 139/164 | 84.76% |

Primary LoRA mean: **84.07%**, sample SD **0.36 percentage points** (three seeds, `ddof=1`).
Mean paired change from base: **-0.21 pp**.
Full-set mean: 84.55%; sample SD 0.35 pp; mean change -0.20 pp.

The SD describes training-seed variation, not a confidence interval over
unseen tasks. The runs share the same problems; they are not 477 independent
primary observations. These small score differences do not establish a
population-level improvement or decline.

## Paired task changes

| Seed | Improved vs base | Regressed vs base | Improved IDs | Regressed IDs |
| --- | ---: | ---: | --- | --- |
| 43 | 1 | 2 | HumanEval/115 | HumanEval/76, HumanEval/95 |
| 44 | 1 | 1 | HumanEval/38 | HumanEval/76 |
| 45 | 1 | 1 | HumanEval/38 | HumanEval/76 |

Improved in all three seeds: none.
Regressed in all three seeds: HumanEval/76.
Different correctness between training seeds: HumanEval/38, HumanEval/95, HumanEval/115.

Saved-code inspection and bounded Docker failure replays distinguish logic
errors, missing imports and missing type guards from extraction/syntax errors.

- **HumanEval/115:** Required import: the base and seeds 44/45 call math.ceil without importing math in the returned function/module. The import in the original prompt is local to a stub that is replaced, so it cannot supply math to the new function. Seed 43 imports math and passes. This is import completeness, not evidence of a new numerical algorithm.

- **HumanEval/38:** Inverse cyclic shift: the failing base and seed-43 solutions apply a left rotation and then a right rotation, cancelling each other; seeds 44 and 45 instead apply only the inverse right rotation. This is a functional logic change.

- **HumanEval/76:** Power-of-n boundary case: the base explicitly returns True for x==1. All three LoRAs initialize power=1 but test equality only inside while power<x, then return False; x=1 and n>1 therefore fails. Docker replays reproduce the first failing hidden assertion candidate(1,12)==True. This regression is also exposed by an example in the visible prompt.

- **HumanEval/95:** Dictionary key types: the base and seeds 44/45 check isinstance(key,str) before calling islower/isupper. Seed 43 omits that guard, causing AttributeError on an integer key. This is an execution failure rather than a syntax or extraction problem.

## Fixed design and independence

- Base: Qwen/Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, non-quantized bfloat16.
- Training: corrected PAD target mask, Adam `1e-6`, 50 updates, fresh seeds 43/44/45, fixed existing MBPP 294/80 split with data seed 42, batch 1, last eight layers, q/v projections, rank 8, scale 20, dropout 0, maximum length 1024, prompt mask and gradient checkpointing. Effective configurations match except seed and output directory.
- The training seed changes adapter initialization and batch order. Fifty batch-size-one updates cover less than one 294-example pass. Checkpoint selection is fixed at 50; validation losses do not select a checkpoint. Previously selected seed 42 is excluded from the confirmation mean.
- Official [HumanEval source](https://github.com/openai/human-eval/tree/6d43fb980f9fee3c892a914eda09951f772ad10d), SHA256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`, all 164 tasks. Dataset manifests separate canonical answers from generation records. Hidden tests never enter chat messages.
- Before outputs, screen against all 974 previously used MBPP records using the [fixed overlap rules](independent-seed-evaluation.md). Exclude HumanEval/13, /23, /28, /35 and /55 from the primary cohort; report all 164 secondarily. These syntactic/textual rules do not prove semantic independence.
- One greedy response per task/model, temperature 0, top-p 1, generation seed 42 for every model, 512-token limit, unchanged extraction. This is a full-module chat adaptation, not the upstream completion/leaderboard protocol. Original prefix imports/helpers are preserved and the generated module overrides its stub before the original hidden check.
- All programs execute only in the existing Docker sandbox: pinned Python image, no network/host mounts, non-root, read-only, 256 MB, three CPU seconds, 10-second execution bound and separate 30-second startup bound.

HumanEval had not selected this project recipe, but it is public and may have
appeared in the pretrained model data. Pretraining contamination is unknown.
The matched comparison measures the incremental effect of this recipe on this
evaluation, not contamination-free general coding capability. Three seeds and
one benchmark also limit generalization. A single final checkpoint cannot
identify a learning-curve turning point or establish overfitting as the cause.

## Next experiments

Keep this evaluation fixed. It has now been inspected and must not be treated
as an untouched final holdout for further tuning. Use MBPP development data or
another explicitly designated development set to investigate training-data
diversity, import completeness, type handling and boundary coverage without
copying these HumanEval cases into training. Before selecting a revised recipe,
reserve another untouched final set, preferably newly authored/private problems
with audited tests; confirm it across training seeds and report every seed.
A best seed or lower reference-token loss alone is insufficient evidence.

## Completed validation and artifacts

- All **59** unit tests passed, including **11** new HumanEval tests and **six** native MLX tests; zero skipped. `uv lock --check`, `uv pip check` and the native environment check passed. No dependency or lockfile changes.
- Original canonical programs: **164/164** passed Docker tests before model generation. Supplementary full-module assembly control: **164/164** passed during generation, without changing conditions.
- Three real 50-update training runs, distinct adapter weight hashes, four complete 164-task evaluations (**656** fresh responses), identical evaluation protocols, paired task ordering and generation hashes checked.
- All **280** pre-existing artifact fingerprints matched; the base weights, original dataset and adapters were reused without overwriting. New outputs and adapters have separate directories.
- Full report: `outputs/independent-seeds-20261015/report.txt` and `report.json`. Per-model tables: `summary.csv`; all task outcomes: `problem-outcomes.csv`; code changes/failure traces: `changed-cases.json` and `case-diagnostics.json`.
- Graphs: `independent-performance.{png,svg,pdf}` and `independent-task-transitions.{png,svg,pdf}` in the same output directory. Raw generations, scored outcomes, frozen plan, overlap audit, training/config hashes, logs and preservation checks are retained there.
- Adapters: `adapters/independent-1e6-seed{43,44,45}-20261015/`, each with its final 50-update weights, checkpoints and provenance.
- Frozen experiment implementation: `710f1d7c046898cf54229eecc72beb92c8757319`; frozen-plan SHA256 `9ee9f5136dc7532c836276a69101a642f8a6b0731ebd53446c07341a19c04818`. Source checks include the Git revision: reproduce the saved run from that checkout, not a later documentation commit. All training/evaluation output destinations must be fresh; do not rerun into existing directories.

Results and large artifacts remain local ignored files. This document records
the completed measurements for the source PR. No required post-merge verification.

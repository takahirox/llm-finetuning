# Fixed Qwen2.5 freight positive control

This is the focused workflow for [Issue #15](https://github.com/takahirox/llm-finetuning/issues/15).
Use the [README commands](../README.md#run-the-fixed-experiment) from the repository
root. Run one fixed recipe; no model or data sweep is part of this benchmark.

## Reference and licensing

Reference: [openatlaspro-AI/mlx-lora-finetune at
24f663d1a276203356d957d6e340ee2182ff83d8](https://github.com/openatlaspro-AI/mlx-lora-finetune/tree/24f663d1a276203356d957d6e340ee2182ff83d8).
The generator, prompt, schema, evaluator, config, committed adapter config,
committed split files, comparison, project dependencies, and MIT license were
inspected. No upstream adapter weights were downloaded or reused. The vendored
generator/prompt/schema and pure scorer retain Mark Teji's
[MIT notice](../THIRD_PARTY_NOTICES.md). The scorer's runner and unused imports
were removed; its scoring functions were preserved unchanged apart from removing
trailing blank lines at EOF.

Model: [mlx-community/Qwen2.5-0.5B-Instruct-4bit at
a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3).
The model card identifies the instruction checkpoint and Apache-2.0 licensing,
and says conversion used MLX-LM 0.18.1. Its `base_model` metadata names the
non-Instruct parent, while its prose names the Instruct source; we use the exact
published 4-bit repository requested by the reference, with no reconversion.
The card links the original [Qwen license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE),
which was inspected separately. Preserve model licensing/notices if redistributing
weights; the reference code's MIT notice does not license the model. Model weights
are not committed here.

## Frozen primary protocol

[configs/freight.json](../configs/freight.json) pins the following:

| Component | Fixed setting |
| --- | --- |
| Data | 600 training / 150 validation / 150 held-out test |
| Generation seeds | train 1001, valid 2002, test 3003; independent Python `random.Random` streams |
| Algorithm | 20 cities; distinct origin/destination; integer 50–44000 lbs; four uniform modes; on_time probability 0.55; four weight renderings; three mode renderings per mode; four status renderings per boolean; eight templates; reject duplicate text within each split |
| Prompt | Exact upstream system prompt and `Shipment: {text}\nExtract the JSON:` user turn |
| Serialization | Exact upstream Pydantic schema and key-ordered `json.dumps`; all three JSONL SHA-256 values match the committed reference |
| Chat formatting | Pinned model tokenizer's `apply_chat_template`; evaluation passes system/user only and `add_generation_prompt=True, tokenize=False` |
| Training | Fresh LoRA; Adam; 600 optimizer iterations; batch 4; lr 1e-4; last 8 layers; rank 8; scale 20; dropout 0 |
| Other training settings | seed 0; max sequence 512; mask_prompt false; no resume; gradient accumulation 1; no gradient checkpointing; no schedule; no remote model code |
| Validation | 25 batches every 150 steps (plus MLX-LM's initial validation); final adapter always selected, no test-based selection |
| Saving | Every 200 steps and final adapter; MLX-LM effective config saved and checked |
| Decoding | One greedy completion per example; 128 new tokens; seed 0 reset per example |
| Scoring | Exact upstream permissive object parser and canonicalization; all 150 examples in each denominator |

The data validator independently checks size, full deterministic content, and
global uniqueness of IDs and complete examples across splits. IDs hash the user
message, independent of split position. Generated data must also match all three
upstream SHA-256 hashes. Any unexpected collision causes failure rather than
silently changing a held-out example. Training copies only `train.jsonl` and
`valid.jsonl` into a new adapter directory; the loader never sees `test.jsonl`.
There are no test examples in the system prompt.

## Scoring semantics

The upstream parser first tries a bare JSON object, then a greedy brace block
embedded in prose/fences. `json_valid` counts successful object parsing, including
empty objects, and does not validate the five-field schema. The additional
`strict_json_valid` counts a bare JSON object, rejecting non-finite constants;
it also does not validate field types or keys.

For exact-field scoring, city values are trimmed, whitespace-collapsed and title
cased. Mode accepts case-insensitive canonical names only, not phrase aliases.
Weight uses Python `int(value)` conversion, including its permissive float/bool
behavior. on_time accepts booleans and strings true/yes/false/no, ignoring case
and surrounding whitespace. Missing fields fail. Extra keys do not penalize the
all-fields score. These permissive rules are intentional reference compatibility,
not a claim of strict schema conformance. Every invalid prediction contributes
five field misses and an exact-record miss. No invalid output is dropped.

Reports include raw numerator/denominator, fractions, percentage-point deltas,
upstream micro-averaged field accuracy, raw responses, IDs, golds, and per-field
outcomes. Saved-result validation regenerates the fixed records and recomputes
all outcomes before comparison. It rejects incomplete/unknown versions, altered
rows/order/counts, summaries, model/data revisions, decoding, scorer or source
fingerprints, unpaired protocols, and incompatible training provenance.

## Intentional differences and limits

- Upstream declares `mlx-lm>=0.31` and `pydantic>=2.6`, with no dependency lock
  or original runtime receipt. We lock Python 3.12.13 and the complete dependency
  graph, retaining this repository's MLX 0.32.3 / MLX-LM 0.32.0. This may differ
  from the author's actual software. We cannot reconstruct unrecorded versions.
- Upstream names a floating model ID. We pin the immutable revision resolved
  during implementation; we cannot establish the author's original revision.
- We explicitly set greedy decoding and reset seed 0. The inspected MLX-LM
  0.32.0 `generate_step` default is also greedy; upstream does not record its
  inference seed or runtime version. The reset should not affect greedy output.
- We preserve upstream `mask_prompt: false` and the stock pinned MLX-LM loss,
  including its inclusive end-bound padding behavior. The legacy coding
  workflow's corrected loss hook is deliberately not used: changing the
  training objective would diverge from this positive-control recipe. This is
  a replication choice, not a reversal of the legacy padding-loss diagnosis.
- Data are generated locally and ignored, with exact upstream-byte verification;
  unlike upstream, no dataset, cached model, adapter weights, or raw result is
  committed. Only suitable small synthetic experiment receipts may be retained
  with the validation record. No upstream stored result is treated as a local run.
- We add provenance, physically isolate test data from training, disallow output
  overwrites, and report strict JSON validity separately. No fallback to another
  model is supported. A format/code mismatch requires the original source in
  another worktree, not editing the old receipt or weakening checks.

Freeze data, prompts, scoring and recipe before viewing results. If the measured
improvement differs from the upstream 71.3% → 100.0%, inspect the saved model/data
hashes, prompts/tokenizer, effective config, software versions and per-field
errors. Report uncertainty; do not change the test/scorer to reach the reference
score. A high score on these eight synthetic templates does not establish
transfer to noisy real shipment text or general coding ability.

## Legacy and collaboration scope

At implementation, PR #12 (`independent-seeds-humaneval`, Issue #11) was open and
PR #14 (Issue #13) was merged. PR #12 adds `scripts/independent_eval.py`, its tests,
and two docs, importing the existing experiment and sandbox modules. Inspection
also found the data-study and loss-masking pipeline importing those modules.
Therefore we archived the old README as [README-legacy.md](../README-legacy.md)
and retargeted legacy model/evaluation links to that archived README. We retired
those workflows from the primary navigation, while preserving all
their source, configs, tests and docs in place. This is an explicit alternative
to deletion until the open PR's work can be reconciled. No PR branch was changed.
Ignored local artifacts and all collaboration policies/templates are preserved.

No post-merge execution is required automatically. Merging never implies that
training or evaluation ran. The [validation record](freight-validation.md)
distinguishes actual execution, limitations, and checks. Publication is outside
this GitWeave implementation node: it commits locally without pushing, opening
a PR, merging, or commenting.

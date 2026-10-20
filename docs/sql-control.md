# SQL generation LoRA positive control

This separate experiment maps a natural-language question and table definitions
to a SQL query with `Qwen/Qwen2.5-1.5B-Instruct`. SQL is executable code with a
small grammar and a practical schema-grounding task. Improvement here would be
a specialized precursor to later Python code-quality work, not evidence of
general Python or 3D-programming ability. The [freight experiment](../README.md)
remains the primary, already measured positive control.

**Status:** implementation and native tests validated; full model-weight download,
conversion, 5,000-example LoRA training, and model inference **not performed** for
this change. No local base/adapted accuracy or improvement is claimed. See the
[validation record](sql-control-validation.md).

## Copy and run

Use a native Apple Silicon Mac with macOS 14+, Python 3.12.13 and uv 0.12.23.
Run from the repository root. MLX needs Metal. Allow disk space for the original
BF16 checkpoint (about 3.1 GB), converted checkpoint, caches, adapter and results;
unified-memory fit and runtime depend on the machine. The SQL extra pins
SQLGlot 26.33.0; the existing lock pins MLX 0.32.3 / MLX-LM 0.32.0 and all
transitive packages. Setup and downloads need network access.

```sh
uv python install
uv sync --locked --extra sql
.venv/bin/python scripts/check_environment.py

# Verify pinned source bytes and freeze 5000 train / 100 valid / 500 held out.
.venv/bin/python -m scripts.sql_control.run prepare-data --data data/sql-control

# Download the immutable Qwen checkpoint, then convert ONCE to MLX affine 4-bit.
.venv/bin/python -m scripts.sql_control.run prepare-model --model models/sql-qwen25-15b-4bit

# Fresh seed-42 rank-8 adapter. No resume or adapter selection against test results.
.venv/bin/python -m scripts.sql_control.run train \
  --model models/sql-qwen25-15b-4bit --data data/sql-control \
  --adapter adapters/sql-r8-seed42

# Inspect progress from another terminal:
tail -n 20 adapters/sql-r8-seed42/training.log

# Same converted weights, cohorts, tokenizer, prompt, sampler and scorer.
.venv/bin/python -m scripts.sql_control.run evaluate \
  --model models/sql-qwen25-15b-4bit --data data/sql-control \
  --output outputs/sql-base
.venv/bin/python -m scripts.sql_control.run evaluate \
  --model models/sql-qwen25-15b-4bit --data data/sql-control \
  --adapter adapters/sql-r8-seed42 --output outputs/sql-adapted

# Recompute scores, verify model/data/source receipts and paired prompt hashes.
# Replace notes with your observations; do not infer causes from scores alone.
.venv/bin/python -m scripts.sql_control.run compare \
  --model models/sql-qwen25-15b-4bit --data data/sql-control \
  --base outputs/sql-base/result.json --adapted outputs/sql-adapted/result.json \
  --output outputs/sql-comparison \
  --notes 'Describe measured changes and inspect raw errors; record possible causes and uncertainty.'
cat outputs/sql-comparison/comparison.md
```

Use new destination paths for every attempt. Existing destinations, including
interrupted runs, are rejected. Data, model weights, adapters, logs and raw
receipts remain under ignored `data/`, `models/`, `adapters/`, `outputs/` and
`.cache/`. Commands never delete historical artifacts. A previously downloaded
source can be supplied with `prepare-data --source path/to/sql_create_context_v4.json`;
its checksum is still mandatory. Train copies only `train.jsonl` and
`valid.jsonl` into a new adapter directory; held-out and fixture answers never
enter the trainer. Use the final adapter regardless of test results.

## Immutable provenance and reference audit

[Reference repository at the reviewed commit](https://github.com/W-Kaski/qwen-qlora-sql-benchmark/tree/b90759ab610ae29a0e8744098eda6ea0d559df36):
`b90759ab610ae29a0e8744098eda6ea0d559df36`.
[Reviewed file checksums](../configs/sql-reference-audit.json) identify the actual
recipe, download/split script, training code, generators, scorers and execution
notes. No explicit source-code license appears in its tree, README or project
metadata at that revision. This implementation is independently written; no
reference source or fixtures are vendored.

| Property | Published reference | This MLX control |
| --- | --- | --- |
| Model | Qwen/Qwen2.5-1.5B-Instruct; revision unspecified | Same model, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Precision | Training/adapter: CUDA bitsandbytes NF4, BF16 compute; baseline generator loads `torch_dtype=auto` without NF4 | Both runs use the exact same locally converted affine 4-bit, group-size-64 BF16 checkpoint |
| Data | Dataset train split; revision/checksums unspecified | JSON at dataset revision `9d80a6a118b838d9defc3798d659a54a2ac2ff37`, SHA-256 verified |
| Selection | Python `Random(42).shuffle` row indices: first 5000 train, next 500 eval, next 100 serving prompts; no deduplication | Seed 42; test-first 500, separate validation 100, training 5000; duplicate protection |
| Input | Plain text, no chat template, heading and schema/question/SQL sections | Independent SQL-only instruction + schema/question, pinned Qwen chat template with default system message |
| Target/loss | Reference SQL plus EOS, prompt labels -100; max length 1024 | Reference SQL assistant turn, completion-only corrected loss, length 1024; end-of-turn tokens included |
| LoRA | All layers, q/k/v/o and gate/up/down projections, rank 8, alpha 16, dropout .05 | Same seven projections and rank/dropout; MLX direct scale **2 = alpha/rank**, not 16 |
| Training | One epoch, batch 1, accumulation 16, LR 1e-4, warmup ratio .1, Trainer default linear decay/AdamW, no weight decay | 5000 microsteps, batch 1, accumulation 8 = 625 updates; AdamW, no weight decay; 63-update warmup + linear decay |
| Validation | Same 500 eval records used for training loss every 100 steps | Separate 100 validation records; no test loss evaluation; final adapter only |
| Generation | Seed 42, greedy, max 256 new tokens, top-p 1 | Same decoding settings; both use identical Qwen chat prompts |
| Text metric | Whitespace collapse, strip one trailing semicolon, lowercase all text | Same text normalization, no fence stripping or SQL extraction |
| Parse metric | SQLGlot SQLite `parse_one`, root must be SELECT; dependency only lower bounded | SQLGlot 26.33.0, exactly one SELECT including SELECT-root CTEs; reject embedded DML/DDL/INTO/commands |
| Execution | Separate 30 curated rank-32 cases, not 500-case execution accuracy | Separate 12 original fixtures, three fixed seeded databases per fixture, both base and rank-8 adapter |

The source prompt begins “You are a Text-to-SQL assistant.” followed by
“Use the SQL schema to answer the question with one SQL query.” and the schema,
question and SQL headings. See its immutable prompt-builder checksum for the
exact layout. Neither upstream training nor inference applies a chat template.
Our instruction asks explicitly for SQLite SELECT and bare SQL. These prompt,
cohort, validation, framework, optimizer-update count and quantization changes
can materially affect scores. MLX does not reproduce CUDA NF4 numerics, and the
MLX trainer does not implement the Transformers default gradient clipping.
This is an independent positive control, **not bit-identical reproduction**.

The source author reports **22/500 (4.4%) → 342/500 (68.4%) normalized exact
match**, +64.0 percentage points; **490/500 (98.0%) → 496/500 (99.2%) parse
validity**, +1.2 pp. These are third-party single-seed results, not our results or
execution accuracy. The source's 30-case execution report concerns rank 32.
Matching or improving on the published score is not required. If local results
fail to improve, preserve them and inspect raw format, truncation, projection,
predicate/value and generation-length errors. Identify possible causes as
hypotheses unless tested independently; never tune the frozen test set or scorer.

## Dataset attribution and split integrity

[sql-create-context by b-mc2 (2023)](https://huggingface.co/datasets/b-mc2/sql-create-context/tree/9d80a6a118b838d9defc3798d659a54a2ac2ff37)
is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It derives from WikiSQL (Zhong, Xiong and Socher, *Seq2SQL*, 2017) and Spider
(Yu et al., *Spider*, 2018); see the pinned dataset card for citations and its
schema inference/SQLGlot transformation process. Credit b-mc2 and these sources
when redistributing derived splits. This control converts records to chat
training examples and selects new subsets; those are our modifications.
The [pinned Qwen model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306)
is Apache-2.0; its downloaded LICENSE is retained separately from dataset licensing.

The source JSON contains 78,577 records with `context`, `question`, `answer`.
Its SHA-256 is
`280961458ccb74731104ae9af715797a0332327a33a210120a6884085a5c03f0`.
Preparation checks this before selection. It shuffles all indices using the
pinned Python version and seed, then greedily fills test, valid, train in that
order. Blank or non-string fields are rejected; there is no answer-correctness,
SQL-validity, length or prediction-based filtering.

Content IDs hash the original three fields; source row indices are retained.
Near-duplicate detection groups by Unicode-NFKC/casefolded word tokens of the
schema (ignoring punctuation/whitespace), then rejects pairs whose question word
sets have Jaccard similarity >= .85. This also removes reordered question-word
variants. Detection applies within and across all selected splits and against
the execution fixtures. It is an explicit lexical definition, not a guarantee
against arbitrary semantic paraphrases or shared database knowledge. No execution
fixture uses a dataset table name; its reserved schema uses `control17_` names.

`manifest.json` saves every selected ID, exclusion count, file checksum, source
revision and fixture/population fingerprint. Every train/evaluate/compare command
reconstructs selection from the pinned source and verifies both raw records and
formatted training messages. Cohorts must never change after predictions are
inspected. Validation is for loss monitoring only, not a tuning sweep.

## Scoring and receipts

Every evaluation generates the 500 test examples and then 12 separate SQLite
fixtures. Gold answers are used only by the scorer. Prompts contain schema and
question, with no gold SQL. Responses are scored verbatim: no extraction of SQL
from Markdown, explanatory prose, or extra statements.

- `exact_match`: upstream-compatible normalized string agreement. It can reject
  equivalent queries. Lowercasing and whitespace collapse also affect quoted
  values, so it can accept queries with different semantics. It is a text metric.
- `parse_valid`: strict single SELECT accepted by the pinned SQLite parser.
  Unknown tables/functions can still parse. This is neither execution validity
  nor semantic correctness. Set-operation roots are excluded consistently.
- `execution_match`: all three fixture populations must produce the same result
  as the gold query. Order matters for the ordered top-three fixture; otherwise
  compare bags of rows, preserving duplicates, NULLs and column counts. Numeric
  values use exact Python numeric equality, without a floating-point tolerance.

Execution uses an isolated `python -I -S` subprocess and fresh in-memory SQLite
database for each seed, with only trusted repository schema/data as setup. It
rejects non-SELECT/multiple statements before execution and uses SQLite
`query_only`, a read/function allowlist authorizer, disabled extension loading,
a VM progress budget, two-second CPU limit, four-second wall timeout, and
SQL/row/column/value/output limits. There is no disk database, networking or
shell evaluation. This is a controlled fixture executor, not a general hostile
code sandbox. Writes, ATTACH, PRAGMA, extension loading and unlisted functions
fail. Execution errors/timeouts count as failures, never denominator exclusions.

Fixtures cover filters, literal case, projection, aggregates, grouping/HAVING,
NULL, DISTINCT, LIKE, BETWEEN, ordering/LIMIT, joins and subqueries. Three fixed
seeds populate 40 staff rows and four team rows with mixed-case values, ties and
NULL bonuses. Both models receive the same schemas/questions and all scorer
populations/gold results. Twelve cases remain twelve scored items (not 36
independent tasks). Limited synthetic data can admit coincidentally equivalent
queries; there is no 500-case execution benchmark, real-world coverage claim,
multiple-table schema generalization claim or general programming claim.

The report includes per-example outcomes and categories, correct/total counts,
fractions, signed percentage-point deltas, fail→pass and pass→fail transitions,
and third-party comparison/interpretation notes. Category counts are saved and
reported for both text and execution failures. Separate receipts capture model
weights/tokenizer/config hashes, conversion provenance, dataset/split hashes,
config, source/lock hashes, package and SQLite versions, timestamps, wall duration,
training mask audit/effective recipe, adapter hashes, raw predictions and peak
MLX evaluation memory. Training logs/runtime receipts remain beside the adapter.

Comparison requires the original local model/data artifacts and exact source and
environment versions. It verifies complete final training, adapter compatibility,
cohort identity/order, prompt hashes and recomputed scores. Compare historical
runs from their recorded checkout/environment; never rewrite old receipts to
make incompatible experiments pair. Source hashes include shared masking code,
so a masking fix requires fresh training and evaluation.

## Checks

```sh
uv lock --check
uv pip check
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m unittest discover -s tests -v
RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q scripts tests
git diff --check
```

Tests use no model downloads. SQL tests explicitly skip when the SQL extra is
absent; run them with the installed extra above. Native tests exercise real MLX
loss/masking and all seven LoRA projection conversions. The full pretrained-model
experiment is optional and must be reported separately from these checks.
No post-merge experiment starts automatically.

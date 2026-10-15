# Data-study implementation validation

Validated on 2026-10-16 for Issue #13, on native Apple Silicon macOS 26.6.2,
Python 3.12.13, MLX 0.32.3, MLX-LM 0.32.0, PyArrow 23.0.1 and Docker 29.1.3.
This records implementation checks and corpus coverage, **not Qwen accuracy**.
PR #12 was checked through GitHub: open, unmerged, with no merge commit.

## Actual source audit and split coverage

The pinned downloads in `configs/data-study-sources.json` were fetched and their
SHA-256 checksums verified. Two independent preparations produced identical split
IDs and byte-identical training/validation JSONL files. The second complete
preparation took **82.39 seconds** wall time; no training was involved.

| Audit | Count |
| --- | ---: |
| Source records examined | 10,000 |
| Historical exclusions indexed | 974 MBPP + 164 HumanEval |
| Eligible, deduplicated source records | 4,820 |
| Eligible records satisfying additional quality criterion, before holdouts | 4,590 |
| Available training pool after all holdouts | 4,540 |
| Loss-validation / development / locked confirmation | 80 / 100 / 100 |
| Nested baseline training sizes | 300 / 1,000 / 3,000 |
| Same-pool quality/diversity training size | 300 |

No example was duplicated to reach a target. All relationships and file hashes
passed the manifest verifier. The confirmed 3,000-example capacity is specific to
the pinned source and documented static rules, not a promise about other corpora.

| Rejection reason | Count |
| --- | ---: |
| Normalized text/code overlap | 4,053 |
| Unsupported dependency | 977 |
| Unsupported test structure | 87 |
| Unsupported test function | 27 |
| Unsupported operation | 18 |
| Insufficient tests | 13 |
| Placeholder solution | 4 |
| Test/solution function mismatch | 1 |

Overlap matches comprise 320 MBPP, 114 HumanEval, and 3,619 already retained
KodCode matches. These are excluded source rows, not unique historical tasks.
The eligible set has 2,971 easy, 993 medium and 856 hard items by the publisher's
model-derived labels; all 12 source subsets remain represented. Question length
ranges 42–7,519 characters, solution length 70–2,137, and static assertion count
2–29. The additional selection balances subset/difficulty, not proven semantic
categories. Correctness and independent authorship are not established by this
audit. Public pretraining and residual semantic overlap remain possible.

Local artifacts, deliberately ignored and absent from Git checkpoints:

- `datasets/issue13-sources/`: original downloaded files.
- `data/kodcode-study-verified/`: second audit, rejection ledger, sample provenance,
  nested conversations, separate heldouts, and locked manifest.
- `runs/issue13-validation/coverage-summary.json`: sanitized counts/checksums and
  per-split ID fingerprints; `docker-smoke.json`: execution outcomes.

The second study manifest fingerprint was
`0f0757d833f8cbcc4386dfd5961787f48bcb7d9b5419d42e3fab382680b0cab0`.
Absolute local output paths and preparation code fingerprints are captured in
that manifest, so a new directory or later code revision changes the overall
manifest hash even when sample membership and JSONL bytes are identical.

## Executed checks

- `uv sync --locked --extra study`: passed; only the optional Parquet dependency
  was added to the existing lock.
- `uv pip check`: passed, all installed packages compatible.
- `scripts/check_environment.py`: passed, including native imports and Metal.
- `RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -v`:
  **61 tests passed, no skips**. The new data tests cover checksum/pinning,
  static filtering and hidden tests, exact candidate search versus a Jaccard
  oracle, overlap including history, deterministic nesting and separation,
  coverage shortfalls, manifest tampering, training controls, actual exposure,
  compatible evaluation/comparison, report separation and confirmation locking.
  Existing MBPP, PAD-mask and Docker failure-handling regressions pass.
- Native tests executed real tiny MLX LoRA optimization and validation through
  the pinned trainer for both mask variants. The additional corrected-mask
  two-example study epoch recorded each sample exactly once and verified the
  saved token/order fingerprint. These tests use repository-owned tiny weights,
  not the production Qwen checkpoint.
- Real Docker smoke: five selected **training** reference solutions, converted
  from pinned KodCode records, passed their test suites (**5/5**). An intentionally
  wrong repository-owned function failed its assertion (negative control).
  The existing pinned Python image, no network/mounts, read-only unprivileged
  execution, 256 MB, 3 CPU seconds, 10-second execution and separate 30-second
  startup bound were used. No external solution executed on the host. These six
  cases verify conversion/scorer compatibility, not corpus-wide correctness or
  generated-model pass@1. No development or confirmation model outcomes were
  viewed.
- `python -m compileall -q scripts tests`, JSON parsing and `git diff --check`:
  passed. New documentation uses English and local links resolve.

## Not performed

No production Qwen weights were downloaded or changed. Full Qwen 300/1,000/3,000
training, same-size quality training, development generation, repeated training
seeds, model memory/accuracy measurements, and final confirmation were **not
performed**. They remain future optional local executions using the documented
frozen policy. No accuracy, improvement, or full-study compute measurements are
inferred from these checks. Confirmation remains locked and unconsumed. There
is no required post-merge verification; merging starts no training/publication.

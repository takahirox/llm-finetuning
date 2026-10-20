# SQL control validation record

Issue [#17](https://github.com/takahirox/llm-finetuning/issues/17), checked on
2026-10-20. This record describes implementation validation, not a measured SQL
LoRA improvement. [Machine-readable checksums and token audits](sql-control-validation.json)
contain no model predictions or dataset examples.

## Completed

- Read Issue #17 and confirmed it remains open with no comments. PR #12 remains
  open (`independent-seeds-humaneval` → `main`); its legacy modules, tests and
  artifacts were preserved. Freight source/config/results were preserved.
- Reviewed the immutable upstream recipe, split selection, generators, training
  masking and evaluators, including the separate rank-32 execution report.
  Verified dataset CC BY 4.0 attribution and Qwen Apache-2.0 metadata/LICENSE.
  Reference code has no explicit license; none was vendored.
- Installed the locked SQL extra on native Darwin arm64, macOS 26.6.2,
  Python 3.12.13, uv 0.12.23. Environment check passes with Metal available;
  MLX 0.32.3, MLX-LM 0.32.0 and SQLGlot 26.33.0.
- Downloaded and checked the pinned dataset JSON SHA-256; independently
  reconstructed and verified 5000 train / 100 validation / 500 test records,
  unique content IDs, near-duplicate checks and chat files. Selection required
  no blank-field or near-duplicate exclusions for this seeded cohort. The
  exclusion policy remains enforced and is tested with deliberate duplicates.
- Downloaded **tokenizer/config/LICENSE only**, not model weights, from the
  pinned Qwen revision. Audited all training and validation examples through
  real MLX-LM `ChatDataset` and the shared completion boundary audit. All 5100
  boundaries match generation prefixes, none truncate at 1024 tokens, and the
  corrected mask has 132169 training and 2701 validation target tokens.
- Executed every gold fixture query on all three seeded populations: 12/12
  fixtures pass (36 database instances). This is a scorer self-check, not model
  accuracy. Tested case-sensitive wrong values, projection/order mismatches,
  duplicates, NULLs, empty-result column counts, unsafe statements/functions,
  missing tables and worker timeout handling.
- `RUN_MLX_TESTS=1 .venv/bin/python -m unittest discover -s tests -v`: **90 tests
  pass, no skips**, including all existing suites, real tiny-model LoRA trainer
  tests, SQL prompt masks and conversion of all seven projections. Tests use
  no pretrained model weights. Comparison tests use labeled synthetic receipts.
- `uv lock --check`, `uv pip check`, Python compile checks and `git diff --check`
  pass. Markdown parsing, balanced fences and local link targets were checked.

## Not performed

The optional full SQL experiment was not run: no pretrained model weights were
downloaded, no Qwen 4-bit conversion or fresh 5000-step adapter was produced,
and no 500-example base/adapter predictions were generated. Its wall time,
peak memory and accuracy are therefore unknown. A supported native environment
alone does not establish numerical reproduction. The runnable steps are in
[sql-control.md](sql-control.md); future runs must retain their original receipts
and evidence-based interpretation, whether they improve or fail.

There is no local SQL score to compare numerically with the author's reported
4.4% → 68.4%. Known protocol differences are recorded before any predictions;
none should be adjusted to chase that score. Training and evaluation are manual
commands and are not initiated by commit or merge. No required post-merge check.

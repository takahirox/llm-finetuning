# Issue #7 implementation validation

Validated on 2026-10-14. These checks establish implementation behavior, not
Qwen pass@1 or an improvement from LoRA.

## Performed checks

- `uv sync --locked`, `uv lock --check`, and `uv pip check`: passed with uv 0.12.23,
  Python 3.12.13, MLX 0.32.3, and MLX-LM 0.32.0. No dependencies or lock changed.
- `.venv/bin/python scripts/check_environment.py`: passed on native arm64 macOS,
  with Metal available.
- `.venv/bin/python -m unittest discover -s tests -v`: 36 tests passed. Tests use
  standard-library fixtures/mocks, with no generated-code execution on the host.
  Coverage includes complete split membership/order and separation, startup and
  execution deadlines, payload readiness, delayed CLI exits, Docker state errors
  and cleanup, saved-generation reuse/integrity, legacy rescoring, mismatched
  provenance/incomplete scoring, and all four outcome transitions.
- `.venv/bin/python -m compileall -q scripts tests`: passed.
- Real `prepare-data` using the documented copied 500-task configuration: passed
  the pinned upstream download/checksum, producing 354/20/500 examples. Manifest
  and file hashes verified; evaluation IDs are exactly 11–510 once each and
  disjoint from training/validation. Local files are under ignored `data/mbpp-500/`
  and `outputs/mbpp-500/`.
- `rescore --help`, real Docker `rescore`/`compare` CLI integration on four
  **synthetic hand-written fixtures**: passed. The paired fixture report had one
  task in each transition group and correctly identified changed IDs. These are
  artificial fixtures, not MBPP/Qwen measurements.
- Markdown parsing, local relative links, fenced Python compilation, shell-snippet
  syntax, JSON configuration syntax, ignored artifact paths, and `git diff --check`:
  passed.

The first unit run exposed a fixture checksum that was restored before reading
its manifest; the synthetic fixture was corrected and the suite passed. An early
Docker probe also showed that signaling Python readiness before payload delivery
could still include stdin latency in the execution budget. Readiness now follows
payload delivery and trusted setup. A later probe showed delayed CLI exit delivery;
an independent in-container wall timer and verified exited-container state now
prevent that delay from becoming a model timeout.

## Real Docker smoke checks

Docker Engine 29.1.3 used the README's pinned Python image, already present locally.
With 30 startup seconds, two execution seconds, one CPU second, and 128 MB memory,
the final scorer produced these outcomes for hand-written functions:

| Fixture | Outcome | Startup / host execution-wait time |
| --- | --- | --- |
| Correct square function | Passed, exit 0 | 0.98 / 0.09 s |
| Incorrect function | Failed, exit 1 | 0.92 / 0.07 s |
| Invalid syntax | Failed, exit 1 | 0.93 / 0.14 s |
| Sleep for 60 seconds | Execution timeout | 0.94 / 2.00 s |
| Infinite CPU loop | CPU/resource failure, exit 137 | 0.63 / 2.00 s |

Execution-wait time includes Docker CLI exit delivery, not just Python runtime.
Separate real fixtures verified that a trusted startup delay longer than two
seconds still allows a correct solution with a one-second execution deadline,
and that a one-second startup deadline aborts that delayed startup as an
infrastructure error. No additional scorer containers remained after those probes.
An isolation fixture also passed checks for UID 65534, denied root-filesystem
writes, and failed network connection. This is bounded smoke validation, not a
security audit. Scoring never runs generated Python on the host.

## Not performed

The assigned worktree has neither `models/qwen3-4b-2507` nor the original completed
`adapters/mbpp-lora`. A model-provenance check confirmed the missing checkpoint.
The optional 500 base plus 500 adapted generations and real MBPP scores were
therefore **not performed**; no weights were downloaded and no adapter was trained.
The original 50-task outputs were also unavailable and were not rescored. No
actual successes/500, percentage difference, or changed MBPP outcomes are claimed.

The [copy-and-run procedure](mbpp-500-evaluation.md) provides exact commands for
an environment holding those existing artifacts, including verifying the original
100-step training metadata and rescoring without generation. Any real measured
report must come from those outputs. Public-data pretraining contamination and
benchmark sampling remain interpretation limits.

Proposed PR: **Make the existing 100-step MBPP evaluation reproducible on all 500 tasks**,
targeting `main`, closing [Issue #7](https://github.com/takahirox/llm-finetuning/issues/7).
PR creation/pushing/review/merge are left to the enclosing GitWeave workflow as
instructed for this node. No post-merge verification is required.

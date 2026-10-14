# Issue #5 implementation validation

Validation performed on 2026-10-14. This is implementation evidence, not a measured
Qwen benchmark or a claim that LoRA improves the model.

## Checks performed

- `uv sync --locked`, `uv lock --check`, and `uv pip check`: passed using uv 0.12.23,
  Python 3.12.13, MLX 0.32.3, MLX-LM 0.32.0, and Hub client 2.2.0.
- `scripts/check_environment.py`: passed on native arm64 macOS with Metal available.
- `python -m unittest discover -s tests -v`: 25 tests passed, using standard-library
  fixtures/mocks without native MLX imports or Docker in the tests.
- `python -m compileall -q scripts tests`: passed.
- MLX-LM's installed LoRA and conversion command help/imports: passed. The installed
  signatures and local chat-dataset loader were inspected against the wrapper.
- Real `prepare-data`: passed pinned upstream download/checksum verification and
  produced 354 train, 20 internal validation, and 50 held-out evaluation examples
  under ignored `data/mbpp/`.
- CLI help and missing-model error: checked; a missing model gives an actionable
  `prepare-model` instruction and does not download weights implicitly.
- Markdown relative links/fences and `git diff --check`: passed.

An initial unit run exposed a diagnostic fixture that assumed exactly two project
dependencies. It was updated for the explicitly declared Hub client, then the full
suite passed. No dependency versions were changed by the lock update.

## Real Docker scoring smoke checks

Docker Engine 29.1.3 successfully pulled the documented pinned Python image,
`python:3.12.13-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36`.
Small hand-written fixture solutions, not model completions, exercised the actual
scorer. With a three-second wall deadline, one CPU second, and 128 MB memory:

| Fixture | Observed outcome | Approximate elapsed time including cleanup |
| --- | --- | --- |
| Correct function | Passed, exit 0 | 0.89 s |
| Incorrect function | Failed assertions, exit 1 | 0.63 s |
| Invalid Python | Failed, exit 1 | 0.65 s |
| Sleeping for 60 seconds | Wall timeout, failed | 3.58 s |
| Infinite CPU loop | Resource termination, exit 137, failed | 3.06 s |

Separate fixture assertions confirmed writes to the root filesystem failed and
the solution process ran as UID 65534. Two additional network fixture probes hit
their five- and ten-second wall limits. A direct bounded container probe using
`--network none` then confirmed a socket connection failed with `Network is
unreachable`. These probes do not establish consistent container startup latency.
Container removal
was exercised after successful runs, failures, and the wall timeout. These are
bounded smoke checks, not a security audit or exhaustive resource-limit testing.
The production default limits are recorded in `configs/experiment.json`; these
smoke checks deliberately used shorter limits.

## Not performed

The 4B weights were not downloaded, converted, or loaded, and real Qwen inference,
LoRA training, baseline scoring, and adapted scoring were not run. Although this
host passes environment diagnostics, those optional experiments require a large
model download and substantial additional memory/time; they were outside this
implementation validation. No physical memory fit, throughput, benchmark score,
or learning effect is claimed. Run the README stages to obtain those measurements.

The model identifier, immutable revision, model card/license declaration, native
MLX-LM Qwen3 implementation, and MBPP source/split were verified against the linked
upstream sources in the README. PR creation, remote review, pushing, and merging
are delegated to the enclosing GitWeave flow; this node only makes a local commit.
No post-merge verification is required by Issue #5.

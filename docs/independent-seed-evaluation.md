# Fixed-condition independent evaluation (Issue #11)

This workflow freezes corrected PAD masking, Adam learning rate `1e-6`, 50
updates, training seeds **43, 44, 45**, and the existing MBPP 294/80 split with
data seed 42. All other settings match `configs/padding-ab.json`. The base and
each new adapter receive the same evaluation. Seed 42, already selected on MBPP,
is excluded from the primary three-seed mean. No HumanEval outcome may change
training conditions, prompts, extraction, execution limits, or cohort rules.

Use official [HumanEval](https://github.com/openai/human-eval) revision
`6d43fb980f9fee3c892a914eda09951f772ad10d`, archive SHA256
`b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.
`python -m scripts.independent_eval --config CONFIG prepare --archive ARCHIVE
--data NEW_DIRECTORY` verifies the pin and writes separate prompt/test and
canonical-reference files with file hashes. It refuses an existing directory.

This is a **full-module chat adaptation**, not the upstream completion protocol.
The user sees only the official prompt (including its original examples) and a
fixed instruction requesting a complete module. Canonical answers and the hidden
`test` field never enter chat messages. The supplied prompt prefix preserves
imports and its function stub; the returned complete module overrides that stub.
The scorer then defines the original hidden test and invokes `check(entry_point)`.
No indentation repair, alternative extraction or post-result prompt tuning is used.
The existing bounded, disposable Docker scorer executes all programs; no program
or canonical reference executes on the host. Before model generation, run
`audit-references --data DATA --output NEW_JSON` and require all 164 reference
programs to pass. Hashes of source, data, model, training and scoring are recorded.

Before any outputs, screen the official references/prompts against the previously
used MBPP corpus. Remove a task from the **primary** cohort if any canonical
function exactly matches an MBPP function after removing docstrings/annotations
and alpha-renaming local identifiers, preserving constants/operators/global calls;
or if lowercase prompt-word Jaccard similarity is at least 0.8 with at least
eight shared words. Freeze IDs and matches before generation. Report the full
164-task set secondarily. These conservative screens can miss semantic duplicates
and can flag common small helpers. Both cohorts must be reported, without
outcome-dependent removals. Public pretraining contamination cannot be ruled out.

Record the dataset manifest, training hyperparameters excluding seed, decoding,
scoring, primary IDs, input hashes and initial preregistration hash in an immutable
local plan. Evaluate with `evaluate --model MODEL --data DATA --plan PLAN
--output NEW_JSON [--adapter ADAPTER]`. Each output path must be new. Generation
is greedy with seed 42, temperature zero, top-p one and 512-token maximum. Use
the existing scoring bounds: 30-second startup, 10-second execution, three CPU
seconds, 256 MB, pinned Python image. Do not change these after model outcomes.

Report every seed's pass count/rate, paired gains/losses versus base, the mean
and **sample** standard deviation (`ddof=1`) of the three seed rates, and task
outcomes across seeds. Training-seed variability is different from uncertainty
over benchmark tasks. A reproducible improvement should have a consistent paired
direction across seeds, not just a favorable best seed. Conclusions apply to
this pinned chat evaluation and screened cohort; avoid broad or contamination-free
capability claims. No required post-merge verification or automatic execution.

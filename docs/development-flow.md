# Development Flow

This document defines the default development flow for `llm-finetuning`, including AI-assisted contributions. The default branch is `main`.

## Collaboration language

Use **English** for newly authored templates, development/review/merge documentation, AI instructions, Issue and PR titles and bodies, review comments, and commit messages. This is a collaboration-language convention; it does not require translating historical content or changing product localization.

## 1. Start with an Issue

Work should begin with an Issue using the [Issue template](../.github/ISSUE_TEMPLATE/issue.md). State the problem, expected outcome, relevant context, reproduction steps where applicable, and concrete completion criteria. An assignee or owner field is not required.

Treat the source Issue as the scope of the work. Clarify genuinely blocking ambiguity before dependent implementation; do not invent requirements.

By default, completion criteria should be executable and verifiable by an AI agent. Require human confirmation, physical-device testing, subjective evaluation, or external approval only when necessary for the actual Issue. Explain why it is necessary and what result is expected.

Separate these categories:

- **Pre-merge acceptance:** mandatory implementation requirements and checks achievable and verifiable before merge.
- **Optional validation:** useful additional checks that are not required for completion.
- **Required post-merge verification:** genuinely necessary checks possible only after merge; state that none are required when applicable.

Checks possible only after merge must not be prerequisites for pre-merge approval. This separation preserves all implementation requirements and applicable pre-merge checks. For example, if merging triggers deployment, review the implementation and applicable local checks before merge; verify publication and the published site afterward. Do not add post-merge work routinely.

## 2. Implement and Validate

Inspect the current repository structure, instructions, and tooling. Choose the smallest change that fully addresses the Issue; avoid unrelated refactoring, speculative frameworks, or unnecessary configuration.

Choose checks appropriate to the changed files and actual repository tooling. Run applicable checks and accurately record their results, including failures, skipped checks, and unperformed checks with reasons.

Use the [README checks](../README.md#checks) and relevant experiment guides for the current `uv`, `unittest`, and native MLX validation commands. There are no CI workflows configured. Follow the [test and CI performance principles](#test-and-ci-performance-principles) when selecting checks or proposing future automation; do not introduce tooling merely to document a process.

For a templates/documentation-only change, check template/Markdown syntax, links, whitespace, and consistency using applicable available tooling. No new runtime tests, hardware/device checks, or deployed-site verification are required solely for such a change.

Write a concise English commit message describing what changed and why, and reference the source Issue.

### Test and CI performance principles

**Detect the required failures at the lightest appropriate layer, without redundant execution.** Preserve required correctness, regressions, and meaningful integration coverage when redistributing checks.

1. **Prefer the lightest meaningful layer.** Use fast `unittest` checks for data splits, schema/scoring, branches, and receipt compatibility. Use the opt-in native MLX tests for actual masking, gradients, and LoRA integration; reserve real-model training/evaluation and legacy Docker execution for boundaries that require them. Add tests that catch concrete defects rather than mirror implementation.
2. **Choose combinations deliberately.** Cover distinct processing paths with representative data, models, mask variants, and adapter states. Exhaust combinations when their interactions require it, rather than multiplying every model, dataset size, seed, and evaluation path. Retain required regression cases and fixed experimental protocols.
3. **Match validation to the change.** Documentation-only changes need Markdown/link/consistency/whitespace checks. CLI/UI changes need relevant argument, message, and behavior checks; core/runtime changes need affected tests and applicable native integration checks. Dependency/build or future CI changes need environment/lock and affected execution-path checks. Use focused checks for fast feedback and the applicable full regression suite before merge; a focused pass does not replace required coverage. Report skipped native tests or missing extras explicitly. Full model experiments are optional unless the Issue requires them. Keep mandatory pre-merge checks applicable, optional validation labeled, and genuinely post-merge-only publication checks separate. No training or publication currently starts on merge.
4. **Reuse work and avoid duplicate execution.** Reuse compatible downloads, converted models, and prepared fixtures with keys covering source revisions, configuration, and environment, while keeping runs isolated. Preserve fresh training, new output paths, provenance checks, and immutable historical receipts; reuse must not substitute stale results for required execution. If CI is added, avoid equivalent push/PR runs for one revision, cancel superseded runs when safe, and preserve necessary deployment ordering. Assess parallel execution of independent work using measured CPU, Metal memory, and computation costs rather than assuming more workers help.
5. **Review added assurance and cost together.** Identify the additional defect each new or expanded test detects, whether existing coverage supplies that assurance, and its execution/maintenance cost. Watch repeated model/compiler/container setup, large fixtures, long generation or training loops, and Cartesian-product growth. Keep meaningful assertions and necessary isolation.
6. **Maintain an execution-time budget.** Normal-CI elapsed time and total runner/computation time are unmeasured here; no target is established. Historical experiment timings are not a CI baseline. Establish a representative baseline and target during actual test/CI performance work, track growth, and investigate overruns by revisiting structure and duplication while retaining required coverage. Record elapsed feedback time, total runner/computation time, and queue time separately. This guidance adds no benchmark, hardware purchase, or merge gate.

## 3. Create a Pull Request for the Issue

Propose the implementation through a PR targeting `main`, unless the source Issue or workflow explicitly selects another base branch. Use the [PR template](../.github/pull_request_template.md) to explain:

- the problem and resulting behavior
- the focused changes
- related Issues
- validation actually performed and its results
- failures, skipped checks, and unperformed checks with reasons
- any genuinely required post-merge verification, separately reported as pending

A PR should claim to close an Issue only when it fully addresses the Issue's implementation requirements. If it is intentionally partial, say so and keep the Issue open. Pending checks possible only after merge do not make an otherwise complete implementation partial, but do not claim those checks passed.

## 4. Review Before Merge

Every PR should be reviewed against its source Issue using the [review guidelines](review-guidelines.md).

Verify completeness, correctness, appropriate validation, and scope. Prefer the smallest change that completely solves the stated problem. Mandatory pre-merge acceptance must pass; record required post-merge checks separately as pending.

## 5. Revise or Wait as Needed

For agent-fixable findings, request focused fixes, update the PR, and review the revised change again.

When genuinely necessary human input is missing, record the specific question, why the answer is needed, and the expected decision or evidence. Use the workflow's wait path and wait for the answer before dependent work proceeds. Distinguish this state from agent-fixable findings; do not repeatedly request code fixes for a missing human decision. Resume implementation or review when the answer arrives.

## 6. Merge

Approve and merge through the authorized workflow when the implementation is sufficient, mandatory pre-merge checks pass, and no blocking review findings remain. Required checks possible only after merge do not block approval when recorded separately as pending.

## 7. Perform Required Post-Merge Verification

If required checks were recorded, perform them after merge and report actual results, including failures and follow-up work. Keep them pending until performed. Pre-merge evidence does not prove deployment or publication succeeded. When no post-merge verification is required, no additional step is needed.

The normal flow is:

```text
Issue → implementation and validation → PR → review
  → revision or wait for necessary human input, then review again
  → merge → post-merge verification (only if required)
```

Adapted from GitWeave's [development flow](https://github.com/takahirox/gitweave/blob/main/docs/development-flow.md), [Issue template](https://github.com/takahirox/gitweave/blob/main/.github/ISSUE_TEMPLATE/issue.md), and [PR template](https://github.com/takahirox/gitweave/blob/main/.github/pull_request_template.md).

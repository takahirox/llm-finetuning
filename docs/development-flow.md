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

The repository currently has no application scaffold, build/test commands, training setup, or CI. Do not invent commands, pipelines, dependencies, or CI merely to document a process. If implementation tooling is added later, inspect and use the actual setup.

For a templates/documentation-only change, check template/Markdown syntax, links, whitespace, and consistency using applicable available tooling. No new runtime tests, hardware/device checks, or deployed-site verification are required solely for such a change.

Write a concise English commit message describing what changed and why, and reference the source Issue.

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

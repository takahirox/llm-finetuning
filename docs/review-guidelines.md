# Review Guidelines

Review verifies that a PR is the right and complete response to its source Issue. AI agents should follow these steps and the [development flow](development-flow.md), including its [English collaboration convention](development-flow.md#collaboration-language).

## Review Against the Issue

Read the source Issue, its relevant discussion, the PR description, and the changed files. Treat the Issue as the reference for the intended problem, expected outcome, and mandatory completion criteria.

Ask whether the PR is a complete and appropriately scoped solution to the Issue it claims to resolve.

## Check for Missing Work

Verify that all claimed requirements are implemented. Do not approve a PR as closing an Issue when important implementation requirements remain unresolved.

If the change is intentionally partial, the PR should explicitly describe that scope and the Issue should remain open. Recorded verification possible only after merge does not make an otherwise complete implementation partial.

## Check for Unnecessary Work

Watch for unnecessary abstractions, speculative extensibility, unrelated refactoring, unneeded frameworks or subsystems, and policies or configuration with no demonstrated need. Additional complexity is not automatically beneficial. Prefer the smallest change that completely solves the Issue.

## Check the Result and Validation

Verify that behavior matches the expected outcome, the implementation fits the actual repository structure, and documentation remains accurate and consistent.

Review applicable interface changes against the README's [UI/UX principles](../README.md#uiux).

Check that validation is sufficient for the changed files and actual tooling. Examine evidence for checks actually performed and accurately reported results, including failures, skipped checks, and unperformed checks with reasons. Do not infer success from missing evidence.

The repository currently has no application/test scaffold or CI. For templates/documentation-only changes, check syntax, links, whitespace, and consistency using applicable available tooling. Do not invent commands or require unrelated runtime suites, new runtime tests, hardware/device checks, or deployed-site verification solely for documentation changes.

## Check Pre-Merge Acceptance and Post-Merge Verification

Require mandatory pre-merge acceptance to be achievable and verifiable before merge. Human confirmation, physical-device testing, subjective evaluation, or external approval should be mandatory only when necessary for the actual Issue, with the reason and expected result stated. Optional validation is not a merge requirement.

Checks possible only after merge must not be prerequisites for pre-merge approval. For example, if merging triggers deployment, review the implementation and applicable local checks before merge; verify publication and the published site after merge.

Record genuinely required post-merge verification separately and report it as pending until performed. Pre-merge evidence does not prove deployment or publication succeeded. This distinction does not waive implementation requirements or applicable pre-merge checks. Do not add post-merge work routinely; state that none is required when applicable.

## Review Outcome and Next Action

Distinguish these outcomes clearly:

- **Agent-fixable findings:** request focused fixes for missing requirements, correctness problems, unnecessary scope, or insufficient validation. State the finding, relevant location, and expected correction or evidence. Review the updated PR again before approval.
- **Waiting for necessary human input:** record the specific question, why only the human input can resolve it, and the expected decision or evidence. Use the workflow's wait path and wait for the answer before dependent approval or merge. Do not repeatedly request code fixes for a missing human answer. Resume review when the answer arrives.
- **Ready to merge:** approve and merge through the authorized workflow when the implementation is complete and correct, scope is justified, mandatory pre-merge acceptance passes, validation is sufficient, and no blocking findings remain.

Do not require routine human confirmation when an agent can execute and verify the criteria. Nonblocking suggestions and optional checks should be labeled accordingly and should not prevent approval.

Required checks possible only after merge do not block approval when recorded separately as pending. After merge, report their actual results, including failures and any follow-up; do not mark them complete until performed.

Adapted from GitWeave's [review guidelines](https://github.com/takahirox/gitweave/blob/main/docs/review-guidelines.md).

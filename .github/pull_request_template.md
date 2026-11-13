<!-- Write the title and body in English. Use checks appropriate to the actual repository tooling and changed files. -->

## Summary

Explain the problem and the focused changes that address it.

## Outcome

Describe the resulting behavior, including before/after behavior when useful.

## Validation

List the checks actually performed and their results. Report failures, skipped checks, and unperformed checks with reasons; do not imply they passed. Identify the evidence for mandatory pre-merge acceptance and distinguish optional validation.

For templates/documentation-only changes, check syntax, links, whitespace, and consistency using applicable available tooling. Do not invent commands or require unrelated runtime suites.

## Related Issues

Link the source Issue and other relevant Issues. Use and preserve `Closes #N` when all implementation requirements and mandatory pre-merge acceptance are satisfied. For intentional partial work, state the completed and remaining scope and omit automatic closing references for the parent Issue. Pending post-merge-only checks do not make implementation partial or block approval/closure; see the [merge lifecycle](https://github.com/takahirox/llm-finetuning/blob/main/docs/development-flow.md#6-merge).

## Required post-merge verification

List genuinely necessary checks possible only after merge and report them as pending until performed, or state that none are required. Pre-merge validation does not prove deployment or publication succeeded.

After merge and Issue closure, record results and evidence in this PR, the closed Issue, or a linked validation report, including unperformed or environment-blocked checks with reasons. Closure does not claim they passed. Follow the [post-merge lifecycle](https://github.com/takahirox/llm-finetuning/blob/main/docs/development-flow.md#7-perform-required-post-merge-verification) for actual defects: check for an equivalent Issue, then use it or create a new linked Issue instead of reopening the original.

<!-- Follow https://github.com/takahirox/llm-finetuning/blob/main/docs/development-flow.md and https://github.com/takahirox/llm-finetuning/blob/main/docs/review-guidelines.md. -->

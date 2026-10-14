---
name: Issue
about: Report a problem or propose a change
title: ""
labels: ""
---

<!-- Write the title and body in English. No assignee or owner is required. -->

## Problem

Describe the problem and why it needs to be addressed.

## Expected outcome

Describe what should be true when the Issue is resolved.

## Context

Add relevant examples, logs, constraints, or related Issues. Include reproduction steps and actual versus expected behavior where applicable.

## Pre-merge acceptance

List concrete, mandatory completion criteria that can be achieved and verified before merge. Preserve all implementation requirements and applicable pre-merge checks.

Default to criteria an AI agent can execute and verify. Require human confirmation, physical-device testing, subjective evaluation, or external approval only when necessary for this Issue; explain why and what result is expected. See the [development guidance](https://github.com/takahirox/llm-finetuning/blob/main/docs/development-flow.md#1-start-with-an-issue).

Checks possible only after merge must not be prerequisites for pre-merge approval.

## Optional validation

List useful additional checks that are not required for completion, or state that none are needed. Keep these separate from mandatory acceptance.

## Required post-merge verification

Record genuinely necessary checks possible only after merge, or state that none are required. Do not add post-merge work routinely.

For example, if merging triggers deployment, publication and published-site verification belong here. Report required checks as pending until performed; pre-merge validation does not prove deployment or publication succeeded. See the [review guidance](https://github.com/takahirox/llm-finetuning/blob/main/docs/review-guidelines.md#check-pre-merge-acceptance-and-post-merge-verification).

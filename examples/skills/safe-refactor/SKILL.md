---
name: safe-refactor
description: Refactor checklist with guardrails
---

# Safe Refactor

Refactor with minimal risk. Prefer small, reversible changes.

## Guardrails

- Keep behavior identical unless explicitly asked to change it
- Avoid public API changes
- Update or add tests for any touched behavior
- Avoid broad rewrites—prefer small, focused edits

## Process

1) Identify the smallest unit to refactor
2) Locate the exact file(s) and read them first (`read`)
3) Add or update tests (if missing)
4) Make the change in small steps (`edit` preferred over rewrite)
5) Re‑run relevant tests or explain how to validate (`bash` if available)
6) Summarize what changed and why it’s safe

## Output format

- **Goal**
- **Plan** (3–5 steps max)
- **Safety checks** (tests, edge cases)
- **Changes summary**

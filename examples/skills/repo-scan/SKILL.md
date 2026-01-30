---
name: repo-scan
description: Fast orientation for a new codebase
---

# Repo Scan

You are onboarding a new codebase. Do a quick, practical scan and report back.

## Goals

- Identify entry points and runtime paths
- Find key config, tooling, and scripts
- Highlight the “read this first” files
- Call out risky or complex areas

## Suggested approach

1) List top‑level files and directories (`ls`) and identify likely entry points.
2) Read `README`, `pyproject`, `package.json`, or equivalent (`read`).
3) Find entry points (`grep` for `__main__`, `cli`, `main`, `app`, `server`).
4) Scan `src/` for high‑level modules (`find` + `read` key files).
5) Skim tests to see how things are used (`find` tests + `read` a few).
6) Avoid heavy directories (e.g., `node_modules`, `dist`, `.venv`).

## Output format

- **Quick summary** (2–3 sentences)
- **Entry points** (files + why they matter)
- **Key config** (files + purpose)
- **Modules to read first** (3–5 files)
- **Risks / complexity** (anything surprising)

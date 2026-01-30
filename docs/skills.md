# Skills

Skills are Markdown documents with YAML frontmatter. They are loaded from multiple locations and can be injected into the prompt with `$skill-name`.

## Skill format

```markdown
---
name: my-skill
description: What this skill does
# optional
# disable_model_invocation: true
---

# My Skill

Instructions and references...
```

## Discovery order (low → high priority)

1. `~/.agent/skills/`
2. Any `skills_dirs` configured in `config.toml`
3. `.agent/skills/` (project‑local)

If there are name collisions, higher‑priority sources win.

## Validation rules

- lowercase letters, numbers, and hyphens only
- must match directory name
- max length 64 chars
- description required

## Invocation

- `$skill-name` injects the skill body into the prompt.
- `$skill-name extra args...` appends args after the injected block.

## Examples

- [`examples/skills/repo-scan/SKILL.md`](../examples/skills/repo-scan/SKILL.md) — fast codebase orientation.
- [`examples/skills/safe-refactor/SKILL.md`](../examples/skills/safe-refactor/SKILL.md) — refactor checklist.

Note: the default loader looks in `.agent/skills` and `~/.agent/skills`, so copy any of these into one of those locations or add `skills_dirs` in config to load them.

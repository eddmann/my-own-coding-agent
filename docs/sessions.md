# Sessions

Sessions are stored as JSONL files in `~/.agent/sessions` by default.

## Format

Each session is a JSONL file. The first line is the session header, the rest are entries.

**Session header**
```json
{"type":"session","version":1,"id":"abcd1234","timestamp":"2026-02-06T12:00:00Z","cwd":"/path/to/project","parentSession":"efgh5678"}
```

**Entry types**
```json
{"type":"message","id":"m1","parentId":null,"timestamp":"2026-02-06T12:00:01Z","message":{"role":"user","content":"hello"}}
{"type":"model_change","id":"m2","parentId":"m1","timestamp":"2026-02-06T12:00:02Z","provider":"openai","modelId":"gpt-4o"}
{"type":"session_state","id":"m3","parentId":"m2","timestamp":"2026-02-06T12:00:03Z","leafId":"m1"}
{"type":"compaction","id":"m4","parentId":"m3","timestamp":"2026-02-06T12:00:04Z","summary":"...","firstKeptEntryId":"m1","tokensBefore":50000,"tokensAfter":12000}
```
`session_state` entries are append-only pointers to the active leaf. They don’t add messages to context; they just select which branch is active.

## Capabilities

**Resume**
Load a previous session and continue from its active leaf.

**Tree (in-place branching)**
Entries form a tree via `id` and `parentId`. The active branch is defined by the latest `session_state` entry. Context is built by walking from the leaf to the root and using only that path.

**Fork (new file)**
Fork creates a new JSONL file by copying messages from the active branch up to a selected point. The new header includes `parentSession` to link back to the original session. Forked entries get new IDs and a new `parentId` chain; there is no shared storage between sessions.

**Compaction (append-only)**
Compaction appends a `compaction` entry with a summary and `firstKeptEntryId`. The full history remains in the file; context is rebuilt as summary + messages from the kept point onward.

## Why JSONL?

JSONL makes sessions easy to stream, inspect, and diff without needing a database.

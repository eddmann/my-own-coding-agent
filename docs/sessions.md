# Sessions

Sessions are stored as JSONL files in `~/.agent/sessions` by default.

## Format

- First line: metadata (session id, created_at, cwd)
- Remaining lines: `Message` objects

## Capabilities

- **Resume**: load the latest session
- **Fork**: branch a new session from any message
- **Compaction**: summarize older context when token limits are reached

## Why JSONL?

JSONL makes sessions easy to stream, inspect, and diff without needing a database.

# Vendored agent system prompts

Captured system prompts, one per agent, as `<agent>.txt`. The agent name must
match one in `agent_profiles.py`.

`--agent-system-prompt <agent>` reads these. They are captured automatically by
the same `scripts/capture/capture-<agent>.sh` launchers that grab the tool
schemas (the system prompt rides in the same request body): see
`../agent_schemas/README.md`.

The addon (`scripts/capture/capture_tools.py`) extracts the prompt from the
Anthropic `system` field, the OpenAI Responses `instructions` field, or leading
`system`/`developer` turns in `messages` / `input`.

**These can contain instance-specific or sensitive text — review before
committing.**

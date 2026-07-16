# CLAUDE.md

Guidance for Claude Code when working in this repository — `mlpal-assistants`,
the MLPal Assistants SDK.

## What this is

The provider-agnostic **model gateway SDK** that the mlpal core SDK is built on.
It handles chat completions, image generation, audio, and embeddings against the
MLPal assistants gateway, plus tool/MCP execution and HTTP/streaming transport.

- `facade.py` — `Assistant`, the unified client composing the services below.
- `services/` — `chat`, `images`, `audio`, `embeddings` (+ `_base`).
- `models/` — request/response/message/file/tool data models.
- `tools/` — tool execution loop, `registry`, `schema`, and `MCPBridge` /
  `mcp_config` (the MCP integration the core SDK's `get_mcp_tools()` delegates to).
- `transport/` — `http` and `streaming`.
- `config.py`, `exceptions.py`.

## Keeping docs & skills in sync (do this with every change)

The builder's docs-MCP indexes a **per-file documentation corpus for this SDK**
at `mlpal-docs-db/docs/assistants/` — one doc per module, generated from source
and filled from it. A code change here is not done until that corpus is updated
and the release is versioned:

1. **Docs.** When you add/remove/rename a module or change public API,
   regenerate + refill the affected docs from source (same template + frontmatter
   as the rest of the corpus; keep it 1:1 with `src/mlpal_assistants/`) and
   re-sync into `mlpal-docs-db/docs/assistants/`. That corpus is baked into the
   sa-builder CDE and indexed at container start.
2. **Downstream.** If a change affects how the mlpal core SDK or its authoring
   skills call this gateway (notably `MCPBridge` / the chat surface), update
   those too.
3. **Versioning.** Bump `version` in `pyproject.toml`; the core SDK pins against
   it, so treat it as a real, compatibility-bearing release.

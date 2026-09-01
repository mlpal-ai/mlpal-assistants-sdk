# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.0 (2026-09-01)

- `generate_image` now submits a background job and polls internally
  (gateway async image jobs): long renders (minutes) work with no HTTP
  connection timeout in the path. Signature unchanged; new optional
  `deadline: float = 600.0`.
- New exceptions: `ImageJobFailedError` (carries `.code` and `.job_id`;
  `worker_lost` means nothing was billed and resubmit is safe) and
  `ImageJobTimeoutError` (subclass of the SDK `TimeoutError`, carries
  `.job_id`; the server-side job keeps running).
- New primitives for custom progress UX: `Assistant.submit_image_job` /
  `Assistant.get_image_job` (and `ImageService.submit_job` / `get_job`).
- Automatic idempotency key per submit — a retried submit can never
  create or bill a second job.
- Backward compatible with pre-async gateways: a synchronous response is
  detected and returned directly.

## [0.1.0] - 2024-02-08

### Added
- Initial release of MLPal Assistants SDK
- **Chat completions** with support for:
  - Text messages
  - File attachments (images, audio, documents)
  - Structured output via Pydantic models
  - Tool/function calling with automatic execution
  - Streaming responses
- **Tool execution system**:
  - `ToolRegistry` for registering callable functions
  - `ToolExecutor` with sync/async support, timeouts, parallel execution
  - `AgentLoop` for multi-turn agentic workflows
  - JSON Schema generation from Python type hints
- **MCP (Model Context Protocol) integration**:
  - `MCPBridge` for connecting to MCP servers
  - Stdio and HTTP transport support
  - Tool discovery and execution via MCP
- **Additional services**:
  - Embeddings (`embed()`)
  - Image generation (`generate_image()`)
  - Text-to-speech (`generate_speech()`)
  - Audio transcription (`transcribe()`)
- **File handling**:
  - `File.from_path()` for local files
  - `File.from_bytes()` for in-memory data
  - `File.from_url()` for remote files
  - Automatic MIME type detection
- **Production features**:
  - Full type annotations (mypy --strict compliant)
  - Comprehensive test suite (57 tests)
  - Layered architecture (Facade → Services → Transport)
  - Detailed error hierarchy with context

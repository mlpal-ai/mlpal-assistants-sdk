# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

# MLPal Assistants SDK - Production Design

## Executive Summary

A production-grade Python SDK for the MLPal Assistants Service API. Built with clean architecture principles, proper separation of concerns, and strong typing throughout.

**Core Principles:**
- **Layered Architecture** - Facade → Services → Transport → Models
- **Single Responsibility** - Each module does one thing well
- **Dependency Inversion** - Depend on abstractions, not concretions
- **Type Safety** - Pydantic for requests, dataclasses for responses
- **Thin Client** - Service handles resilience (retries, circuit breaker, failover)
- **Testability** - Each layer testable in isolation

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Structure](#2-module-structure)
3. [Layer Specifications](#3-layer-specifications)
4. [Public API](#4-public-api)
5. [Models](#5-models)
6. [Exception Hierarchy](#6-exception-hierarchy)
7. [Implementation Phases](#7-implementation-phases)

---

## 1. Architecture Overview

### 1.1 Layered Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Application                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PUBLIC API LAYER (Facade)                            │
│                                                                              │
│  Assistants                                                                  │
│  ├── chat(messages, ...) → ChatResponse                                     │
│  ├── embed(input, ...) → EmbeddingResponse                                  │
│  ├── transcribe(file, ...) → TranscriptionResponse                          │
│  └── generate                                                                │
│      ├── image(prompt, ...) → ImageResponse                                 │
│      └── speech(text, ...) → AudioResponse                                  │
│                                                                              │
│  Thin orchestration only - delegates all work to services                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SERVICE LAYER                                     │
│                                                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ ChatService │ │EmbedService │ │AudioService │ │     ImageService        ││
│  │             │ │             │ │             │ │                         ││
│  │ • complete  │ │ • create    │ │ • transcribe│ │ • generate              ││
│  │ • stream    │ │             │ │ • synthesize│ │                         ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────────┘│
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         TOOL EXECUTION                                   ││
│  │  ToolRegistry (validates, stores)  │  ToolExecutor (runs tool calls)    ││
│  │  SchemaGenerator (func → JSON)     │  AgentLoop (orchestrates turns)    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRANSPORT LAYER                                    │
│                                                                              │
│  HTTPTransport                                                               │
│  • request(method, path, body) → dict                                        │
│  • stream(method, path, body) → AsyncIterator[dict]                          │
│  • Connection pooling (httpx)                                                │
│  • Auth headers                                                              │
│  • HTTP status → Exception mapping                                           │
│                                                                              │
│  NO business logic, NO response parsing beyond JSON                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             MODELS LAYER                                     │
│                                                                              │
│  Requests (Pydantic - validated)    │  Responses (dataclass - immutable)    │
│  ├── ChatRequest                    │  ├── ChatResponse                     │
│  ├── EmbedRequest                   │  ├── EmbeddingResponse                │
│  ├── ImageRequest                   │  ├── ImageResponse                    │
│  └── AudioRequest                   │  └── AudioResponse                    │
│                                     │                                        │
│  Domain Models                      │  Common Types                          │
│  ├── File (immutable)               │  ├── CostInfo                         │
│  ├── Message                        │  ├── RoutingInfo                      │
│  ├── ToolDefinition                 │  └── TokenUsage                       │
│  └── ToolCall                       │                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Architecture** | Layered (Facade → Service → Transport) | Separation of concerns, testability |
| **Client Class** | Thin facade (~100 lines) | Delegates to services, easy to understand |
| **Request Validation** | Pydantic models | Catch errors early, strong typing |
| **Response Types** | Frozen dataclasses | Immutable, hashable, fast |
| **Tool Execution** | Separate module | Single responsibility, reusable |
| **HTTP Layer** | Pure transport | No business logic, easy to mock |
| **Dependency Injection** | Constructor injection | Testable, flexible |

---

## 2. Module Structure

```
src/mlpal_assistants/
├── __init__.py                    # Public exports only
├── _version.py                    # Version string
├── client.py                      # Assistants facade (~100 lines)
├── config.py                      # Configuration dataclass
├── exceptions.py                  # Exception hierarchy
│
├── services/                      # Business logic layer
│   ├── __init__.py
│   ├── _base.py                   # BaseService (shared logic)
│   ├── chat.py                    # ChatService
│   ├── embeddings.py              # EmbeddingService
│   ├── images.py                  # ImageService
│   └── audio.py                   # AudioService (transcribe + synthesize)
│
├── tools/                         # Tool execution (separate concern)
│   ├── __init__.py
│   ├── registry.py                # ToolRegistry (stores tool definitions)
│   ├── executor.py                # ToolExecutor (executes tool calls)
│   ├── schema.py                  # JSON schema generation from functions
│   └── loop.py                    # AgentLoop (multi-turn orchestration)
│
├── transport/                     # HTTP communication
│   ├── __init__.py
│   ├── http.py                    # HTTPTransport
│   └── streaming.py               # SSE streaming iterator
│
├── models/                        # All data types
│   ├── __init__.py                # Re-exports
│   ├── requests.py                # Pydantic request models
│   ├── responses.py               # Dataclass response models
│   ├── messages.py                # Message types
│   ├── files.py                   # File class
│   ├── tools.py                   # ToolDefinition, ToolCall, ToolResult
│   └── common.py                  # CostInfo, RoutingInfo, TokenUsage
│
└── _compat.py                     # Python version compatibility
```

### 2.1 Line Count Targets

| Module | Target Lines | Responsibility |
|--------|--------------|----------------|
| `client.py` | ~100 | Facade, delegation only |
| `services/chat.py` | ~150 | Chat completion logic |
| `services/embeddings.py` | ~50 | Embedding creation |
| `services/images.py` | ~60 | Image generation |
| `services/audio.py` | ~80 | Transcription + TTS |
| `tools/registry.py` | ~80 | Tool storage, validation |
| `tools/executor.py` | ~60 | Tool call execution |
| `tools/loop.py` | ~100 | Multi-turn agent loop |
| `transport/http.py` | ~120 | HTTP transport |
| `models/*.py` | ~50 each | Data types |

---

## 3. Layer Specifications

### 3.1 Public API Layer (client.py)

```python
"""Thin facade - delegates all work to services."""

from __future__ import annotations

from mlpal_assistants.config import Config
from mlpal_assistants.models import ChatResponse, EmbeddingResponse, File
from mlpal_assistants.services import ChatService, EmbeddingService, AudioService, ImageService
from mlpal_assistants.transport import HTTPTransport


class Assistants:
    """
    MLPal Assistants API client.

    This is a thin facade that delegates to specialized services.
    Each service handles one domain (chat, embeddings, etc.).
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        config = Config.from_env(api_key=api_key, base_url=base_url, timeout=timeout)
        transport = HTTPTransport(config)

        # Initialize services with shared transport
        self._chat = ChatService(transport)
        self._embeddings = EmbeddingService(transport)
        self._audio = AudioService(transport)
        self._images = ImageService(transport)
        self._transport = transport

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str = "mlpal",
        response_format: type[BaseModel] | None = None,
        tools: ToolRegistry | None = None,
        max_turns: int = 10,
        **kwargs,
    ) -> ChatResponse:
        """Chat completion - delegates to ChatService."""
        return await self._chat.complete(
            messages=messages,
            model=model,
            response_format=response_format,
            tools=tools,
            max_turns=max_turns,
            **kwargs,
        )

    async def embed(self, input: str | list[str], **kwargs) -> EmbeddingResponse:
        """Create embeddings - delegates to EmbeddingService."""
        return await self._embeddings.create(input, **kwargs)

    async def transcribe(self, file: File, **kwargs) -> TranscriptionResponse:
        """Transcribe audio - delegates to AudioService."""
        return await self._audio.transcribe(file, **kwargs)

    @property
    def generate(self) -> _GenerateNamespace:
        """Generation methods (image, speech)."""
        return _GenerateNamespace(self._images, self._audio)

    async def close(self) -> None:
        """Close HTTP connections."""
        await self._transport.close()

    async def __aenter__(self) -> Assistants:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()


class _GenerateNamespace:
    """Namespace for generation methods."""

    def __init__(self, images: ImageService, audio: AudioService) -> None:
        self._images = images
        self._audio = audio

    async def image(self, prompt: str, **kwargs) -> ImageResponse:
        return await self._images.generate(prompt, **kwargs)

    async def speech(self, text: str, **kwargs) -> AudioResponse:
        return await self._audio.synthesize(text, **kwargs)
```

### 3.2 Service Layer

```python
# services/_base.py
"""Base service with shared functionality."""

from abc import ABC
from mlpal_assistants.transport import HTTPTransport


class BaseService(ABC):
    """Base class for all services."""

    def __init__(self, transport: HTTPTransport) -> None:
        self._transport = transport
```

```python
# services/chat.py
"""Chat completion service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlpal_assistants.models import ChatRequest, ChatResponse, Message
from mlpal_assistants.services._base import BaseService
from mlpal_assistants.tools import AgentLoop, ToolRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel


class ChatService(BaseService):
    """Handles chat completions."""

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str = "mlpal",
        response_format: type[BaseModel] | None = None,
        tools: ToolRegistry | None = None,
        max_turns: int = 10,
        **kwargs,
    ) -> ChatResponse:
        """
        Complete a chat conversation.

        If tools are provided, runs an agent loop until completion
        or max_turns is reached.
        """
        if tools is not None:
            return await self._complete_with_tools(
                messages, model, response_format, tools, max_turns, **kwargs
            )
        return await self._complete_single(messages, model, response_format, **kwargs)

    async def _complete_single(
        self,
        messages: list[Message],
        model: str,
        response_format: type[BaseModel] | None,
        **kwargs,
    ) -> ChatResponse:
        """Single completion without tool execution."""
        request = ChatRequest(
            model=model,
            messages=messages,
            response_format=self._build_response_format(response_format),
            **kwargs,
        )

        data = await self._transport.post("/v1/chat/completions", body=request.model_dump())
        return ChatResponse.from_api(data, response_format=response_format)

    async def _complete_with_tools(
        self,
        messages: list[Message],
        model: str,
        response_format: type[BaseModel] | None,
        tools: ToolRegistry,
        max_turns: int,
        **kwargs,
    ) -> ChatResponse:
        """Completion with automatic tool execution loop."""
        loop = AgentLoop(
            transport=self._transport,
            tools=tools,
            max_turns=max_turns,
        )
        return await loop.run(messages, model, response_format, **kwargs)

    def _build_response_format(self, model: type[BaseModel] | None) -> dict | None:
        """Build response_format parameter for structured output."""
        if model is None:
            return None

        schema = model.model_json_schema()
        schema = self._make_strict_schema(schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    @staticmethod
    def _make_strict_schema(schema: dict) -> dict:
        """Add additionalProperties: false recursively for OpenAI compatibility."""
        schema = schema.copy()

        if schema.get("type") == "object":
            schema["additionalProperties"] = False

        if "properties" in schema:
            schema["properties"] = {
                k: ChatService._make_strict_schema(v) if isinstance(v, dict) else v
                for k, v in schema["properties"].items()
            }

        if "$defs" in schema:
            schema["$defs"] = {
                k: ChatService._make_strict_schema(v) if isinstance(v, dict) else v
                for k, v in schema["$defs"].items()
            }

        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = ChatService._make_strict_schema(schema["items"])

        return schema
```

### 3.3 Tool Execution Module

```python
# tools/registry.py
"""Tool registration and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mlpal_assistants.tools.schema import generate_tool_schema


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A registered tool."""
    name: str
    func: Callable[..., Any]
    description: str
    parameters: dict

    def to_api_format(self) -> dict:
        """Convert to API request format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    Registry of tools available for chat completion.

    Validates tool functions and generates JSON schemas.

    Example:
        tools = ToolRegistry()
        tools.register("get_weather", get_weather_func)
        tools.register("search", search_func, description="Search the database")

        response = await assistant.chat(messages, tools=tools)
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        *,
        description: str | None = None,
    ) -> ToolRegistry:
        """
        Register a tool function.

        Returns self for method chaining.
        """
        schema = generate_tool_schema(func)

        self._tools[name] = ToolDefinition(
            name=name,
            func=func,
            description=description or schema.get("description", ""),
            parameters=schema["parameters"],
        )

        return self

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def to_api_format(self) -> list[dict]:
        """Convert all tools to API request format."""
        return [tool.to_api_format() for tool in self._tools.values()]

    @classmethod
    def from_dict(cls, tools: dict[str, Callable]) -> ToolRegistry:
        """Create registry from dict of name → function."""
        registry = cls()
        for name, func in tools.items():
            registry.register(name, func)
        return registry
```

```python
# tools/executor.py
"""Tool call execution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mlpal_assistants.exceptions import ToolExecutionError
from mlpal_assistants.models import ToolCall
from mlpal_assistants.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of executing a tool call."""
    tool_call_id: str
    name: str
    result: str
    success: bool
    error: str | None = None


class ToolExecutor:
    """Executes tool calls against a registry."""

    def execute(
        self,
        tool_calls: list[ToolCall],
        registry: ToolRegistry,
    ) -> list[ToolResult]:
        """
        Execute a list of tool calls.

        Returns results for all calls, including errors.
        """
        return [self._execute_one(call, registry) for call in tool_calls]

    def _execute_one(self, call: ToolCall, registry: ToolRegistry) -> ToolResult:
        """Execute a single tool call."""
        tool = registry.get(call.function.name)

        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.function.name,
                result="",
                success=False,
                error=f"Unknown tool: {call.function.name}",
            )

        try:
            args = json.loads(call.function.arguments)
            result = tool.func(**args)

            # Ensure string result
            if not isinstance(result, str):
                result = json.dumps(result)

            return ToolResult(
                tool_call_id=call.id,
                name=call.function.name,
                result=result,
                success=True,
            )

        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                name=call.function.name,
                result="",
                success=False,
                error=str(e),
            )
```

```python
# tools/loop.py
"""Agent loop for multi-turn tool execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlpal_assistants.exceptions import MaxTurnsExceededError
from mlpal_assistants.models import ChatResponse, Message, Turn
from mlpal_assistants.tools.executor import ToolExecutor
from mlpal_assistants.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from pydantic import BaseModel
    from mlpal_assistants.transport import HTTPTransport


class AgentLoop:
    """
    Orchestrates multi-turn chat with tool execution.

    Continues until:
    - Model returns without tool_calls (finish_reason != "tool_calls")
    - max_turns is reached
    """

    def __init__(
        self,
        transport: HTTPTransport,
        tools: ToolRegistry,
        max_turns: int = 10,
    ) -> None:
        self._transport = transport
        self._tools = tools
        self._max_turns = max_turns
        self._executor = ToolExecutor()

    async def run(
        self,
        messages: list[Message],
        model: str,
        response_format: type[BaseModel] | None,
        **kwargs,
    ) -> ChatResponse:
        """Run the agent loop until completion."""
        current_messages = list(messages)
        turns: list[Turn] = []

        for turn_num in range(self._max_turns):
            # Make API call
            response = await self._call_api(current_messages, model, **kwargs)

            # Check if done
            if response.finish_reason != "tool_calls" or not response.tool_calls:
                return self._finalize_response(response, turns, response_format)

            # Execute tools
            results = self._executor.execute(response.tool_calls, self._tools)

            # Append to messages
            current_messages.append(self._assistant_message(response))
            for result in results:
                current_messages.append(self._tool_message(result))

            # Record turn
            turns.append(Turn(
                messages=[self._assistant_message(response)],
                response=response,
                tool_results=[r.__dict__ for r in results],
            ))

        raise MaxTurnsExceededError(
            f"Agent loop exceeded {self._max_turns} turns",
            turns_completed=self._max_turns,
            max_turns=self._max_turns,
        )
```

### 3.4 Transport Layer

```python
# transport/http.py
"""HTTP transport - pure communication, no business logic."""

from __future__ import annotations

import httpx

from mlpal_assistants.config import Config
from mlpal_assistants.exceptions import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
)


class HTTPTransport:
    """
    HTTP transport layer.

    Responsibilities:
    - Connection pooling
    - Auth headers
    - HTTP status → exception mapping

    NOT responsible for:
    - Response parsing (beyond JSON)
    - Business logic
    - Retries (service handles this)
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.timeout),
        )

    async def post(
        self,
        path: str,
        *,
        body: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        """POST request, returns parsed JSON."""
        return await self._request("POST", path, json=body, timeout=timeout)

    async def post_multipart(
        self,
        path: str,
        *,
        data: dict | None = None,
        files: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        """POST multipart request."""
        return await self._request(
            "POST", path, data=data, files=files, timeout=timeout
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> dict:
        """Execute request and handle errors."""
        try:
            response = await self._client.request(
                method,
                path,
                timeout=timeout or self._config.timeout,
                **kwargs,
            )
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}") from e

        self._raise_for_status(response)
        return response.json()

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status codes to exceptions."""
        if response.is_success:
            return

        status = response.status_code

        try:
            data = response.json()
            message = data.get("detail", data.get("error", {}).get("message", ""))
        except Exception:
            message = response.text[:200]

        error_kwargs = {
            "status_code": status,
            "response": data if "data" in dir() else None,
        }

        if status == 401:
            raise AuthenticationError(message or "Invalid API key", **error_kwargs)
        elif status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message or "Rate limit exceeded",
                retry_after=float(retry_after) if retry_after else None,
                **error_kwargs,
            )
        elif status == 400:
            raise BadRequestError(message or "Bad request", **error_kwargs)
        elif status == 502:
            raise ProviderError(message or "Provider error", **error_kwargs)
        elif status >= 500:
            raise ServerError(message or "Server error", **error_kwargs)
        else:
            raise ServerError(f"HTTP {status}: {message}", **error_kwargs)

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
```

### 3.5 Models Layer

```python
# models/requests.py
"""Request models with Pydantic validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """Validated chat completion request."""

    model_config = ConfigDict(extra="forbid")  # Catch typos

    model: str
    messages: list[dict]
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1)
    top_p: float | None = Field(None, ge=0, le=1)
    stop: list[str] | None = None
    response_format: dict | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None


class EmbedRequest(BaseModel):
    """Validated embedding request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input: list[str]


class ImageRequest(BaseModel):
    """Validated image generation request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str
    size: str = "1024x1024"
    quality: str = "standard"
    n: int = Field(1, ge=1, le=10)
    style: str | None = None
```

```python
# models/responses.py
"""Response models as frozen dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mlpal_assistants.models.common import CostInfo, RoutingInfo
from mlpal_assistants.models.tools import ToolCall

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Immutable chat response."""

    content: str | None
    cost: CostInfo
    routing: RoutingInfo
    finish_reason: str = "stop"
    data: BaseModel | None = None  # Parsed structured output
    tool_calls: tuple[ToolCall, ...] | None = None
    turns: tuple[Turn, ...] | None = None

    @classmethod
    def from_api(
        cls,
        data: dict,
        *,
        response_format: type[BaseModel] | None = None,
        turns: list[Turn] | None = None,
    ) -> ChatResponse:
        """Parse API response into ChatResponse."""
        # Parse structured output
        parsed_data = None
        if response_format and (api_data := data.get("data")):
            parsed_data = response_format.model_validate(api_data)

        # Parse tool calls
        tool_calls = None
        if raw_calls := data.get("tool_calls"):
            tool_calls = tuple(ToolCall.from_api(tc) for tc in raw_calls)

        return cls(
            content=data.get("content"),
            cost=CostInfo.from_api(data.get("cost") or {}),
            routing=RoutingInfo.from_api(data.get("routing") or {}),
            finish_reason=data.get("finish_reason", "stop"),
            data=parsed_data,
            tool_calls=tool_calls,
            turns=tuple(turns) if turns else None,
        )


@dataclass(frozen=True, slots=True)
class Turn:
    """A single turn in an agentic conversation."""
    messages: tuple[dict, ...]
    response: ChatResponse
    tool_results: tuple[dict, ...] | None = None
```

```python
# models/files.py
"""File handling with factory methods."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FileType = Literal["image", "audio", "document"]


@dataclass(frozen=True, slots=True)
class FileData:
    """File content - either URL or base64."""
    url: str | None = None
    base64: str | None = None


@dataclass(frozen=True, slots=True)
class File:
    """
    Immutable file attachment.

    Use factory methods to create:
        File.from_path("image.jpg")
        File.from_bytes(data, mime_type="image/png")
        File.from_url("https://...")
    """

    type: FileType
    data: FileData
    mime_type: str | None

    @classmethod
    def from_path(cls, path: str | Path) -> File:
        """Create from local file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = base64.b64encode(path.read_bytes()).decode()
        mime = mimetypes.guess_type(str(path))[0]

        return cls(
            type=cls._infer_type(mime),
            data=FileData(base64=content),
            mime_type=mime,
        )

    @classmethod
    def from_bytes(cls, content: bytes, *, mime_type: str) -> File:
        """Create from bytes with explicit MIME type."""
        encoded = base64.b64encode(content).decode()
        return cls(
            type=cls._infer_type(mime_type),
            data=FileData(base64=encoded),
            mime_type=mime_type,
        )

    @classmethod
    def from_url(cls, url: str, *, mime_type: str | None = None) -> File:
        """Create from URL (passed through to API)."""
        mime = mime_type or mimetypes.guess_type(url)[0]
        return cls(
            type=cls._infer_type(mime),
            data=FileData(url=url),
            mime_type=mime,
        )

    @staticmethod
    def _infer_type(mime: str | None) -> FileType:
        """Infer file type from MIME type."""
        if not mime:
            return "document"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        return "document"

    def to_api_format(self) -> dict:
        """Serialize for API request."""
        result: dict = {"type": self.type}

        if self.data.url:
            result["url"] = self.data.url
        elif self.data.base64:
            result["base64"] = self.data.base64

        if self.mime_type:
            result["mime_type"] = self.mime_type

        return result
```

---

## 4. Public API

### 4.1 Initialization

```python
from mlpal_assistants import Assistants

# From environment
assistant = Assistants()  # Uses MLPAL_API_KEY

# Explicit
assistant = Assistants(
    api_key="mlpal_sk_...",
    base_url="https://models.mlpal.ai",
    timeout=120.0,
)

# Context manager
async with Assistants() as assistant:
    response = await assistant.chat(...)
```

### 4.2 Chat

```python
# Simple
response = await assistant.chat(
    messages=[{"role": "user", "content": "Hello"}]
)

# With files
response = await assistant.chat(
    messages=[{
        "role": "user",
        "content": "Describe this image",
        "files": [File.from_path("photo.jpg")]
    }]
)

# Structured output
class Person(BaseModel):
    name: str
    age: int

response = await assistant.chat(
    messages=[{"role": "user", "content": "John is 30"}],
    response_format=Person,
)
person = response.data  # Person instance
```

### 4.3 Tools

```python
from mlpal_assistants import ToolRegistry

# Create registry
tools = ToolRegistry()
tools.register("get_weather", get_weather_func)
tools.register("search", search_func, description="Search database")

# Or from dict
tools = ToolRegistry.from_dict({
    "get_weather": get_weather_func,
    "search": search_func,
})

# Use in chat
response = await assistant.chat(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    max_turns=5,
)

# Access turn history
for turn in response.turns:
    print(f"Tool calls: {turn.tool_results}")
```

### 4.4 Other Operations

```python
# Embeddings
response = await assistant.embed(["Hello", "World"])

# Image generation
response = await assistant.generate.image(
    prompt="A sunset over mountains",
    size="1024x1024",
)

# Text-to-speech
response = await assistant.generate.speech(
    text="Hello, world!",
    voice="alloy",
)
response.save("output.mp3")

# Transcription
response = await assistant.transcribe(File.from_path("audio.mp3"))
print(response.text)
```

---

## 5. Models

### 5.1 Response Types

| Type | Fields | Usage |
|------|--------|-------|
| `ChatResponse` | content, cost, routing, data, tool_calls, turns | `chat()` result |
| `EmbeddingResponse` | data (list[Embedding]), model, cost | `embed()` result |
| `ImageResponse` | images (list[GeneratedImage]), cost | `generate.image()` result |
| `AudioResponse` | audio, cost, save() | `generate.speech()` result |
| `TranscriptionResponse` | text, language, cost | `transcribe()` result |

### 5.2 Common Types

| Type | Fields |
|------|--------|
| `CostInfo` | model_name, provider, tokens, latency_ms, compute_units |
| `RoutingInfo` | requested_model, resolved_model, provider, strategy |
| `TokenUsage` | input_tokens, output_tokens, total_tokens |

### 5.3 Tool Types

| Type | Fields |
|------|--------|
| `ToolRegistry` | Container for tool definitions |
| `ToolDefinition` | name, func, description, parameters |
| `ToolCall` | id, function (name, arguments) |
| `ToolResult` | tool_call_id, name, result, success, error |

---

## 6. Exception Hierarchy

```
AssistantsError (base)
├── AuthenticationError (401)
├── PermissionDeniedError (403)
├── RateLimitError (429, has retry_after)
├── BadRequestError (400)
│   └── ValidationError (has errors list)
├── NotFoundError (404)
│   └── ModelNotAvailableError
├── ServerError (5xx)
│   ├── ServiceUnavailableError (503)
│   └── ProviderError (502)
├── TimeoutError
├── ConnectionError
├── ToolExecutionError (has tool_name, original_error)
├── MaxTurnsExceededError (has turns_completed, max_turns)
└── MCPError
    ├── MCPConnectionError
    └── MCPToolNotFoundError
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Refactor) ✅ COMPLETE

**Goal:** Establish clean architecture with layered design.

**Status:** Completed on 2026-02-07

**Implemented:**
1. ✅ Created module structure (services/, transport/, models/)
2. ✅ Implemented `HTTPTransport` (transport/http.py) - ~180 lines
3. ✅ Implemented `Config` (config.py) - immutable dataclass
4. ✅ Created model types:
   - `models/common.py` - TokenUsage, CostInfo, RoutingInfo
   - `models/files.py` - File with factory methods
   - `models/messages.py` - Message types
   - `models/requests.py` - Pydantic request validation
   - `models/responses.py` - Dataclass responses
   - `models/tools.py` - ToolCall, ToolResult
5. ✅ Implemented `ChatService` (services/chat.py) - ~150 lines
6. ✅ Implemented `EmbeddingService` (services/embeddings.py) - ~45 lines
7. ✅ Implemented `ImageService` (services/images.py) - ~60 lines
8. ✅ Implemented `AudioService` (services/audio.py) - ~80 lines
9. ✅ Created thin `Assistants` facade (facade.py) - ~230 lines
10. ✅ Cleaned up old monolithic code

**Files created:**
- `src/mlpal_assistants/config.py`
- `src/mlpal_assistants/facade.py`
- `src/mlpal_assistants/transport/__init__.py`
- `src/mlpal_assistants/transport/http.py`
- `src/mlpal_assistants/services/__init__.py`
- `src/mlpal_assistants/services/_base.py`
- `src/mlpal_assistants/services/chat.py`
- `src/mlpal_assistants/services/embeddings.py`
- `src/mlpal_assistants/services/images.py`
- `src/mlpal_assistants/services/audio.py`
- `src/mlpal_assistants/models/__init__.py`
- `src/mlpal_assistants/models/common.py`
- `src/mlpal_assistants/models/files.py`
- `src/mlpal_assistants/models/messages.py`
- `src/mlpal_assistants/models/requests.py`
- `src/mlpal_assistants/models/responses.py`
- `src/mlpal_assistants/models/tools.py`

**Verified working:**
- Chat completions (with routing info, cost tracking)
- Structured output (Pydantic models → JSON Schema → parsed response)
- Embeddings (single and batch)

### Phase 2: Tool Execution ✅ COMPLETE

**Goal:** Clean tool execution with proper separation.

**Status:** Completed on 2026-02-08

**Implemented:**
1. ✅ `ToolRegistry` (tools/registry.py) - ~200 lines
   - Register tools with `register()` or `@tools.tool` decorator
   - Create from dict with `ToolRegistry.from_dict()`
   - Auto-generate JSON Schema from Python functions
   - Convert to OpenAI-format tool definitions

2. ✅ `ToolExecutor` (tools/executor.py) - ~150 lines
   - Execute tool calls with validation
   - Support sync and async functions
   - Timeout protection
   - Parallel execution support

3. ✅ `AgentLoop` (tools/loop.py) - ~180 lines
   - Multi-turn orchestration
   - Automatic tool execution
   - Turn history tracking
   - Max turns protection

4. ✅ Schema generation (tools/schema.py) - ~130 lines
   - Generate JSON Schema from Python type hints
   - Parse docstrings for parameter descriptions
   - Support Optional, Union, list, dict types

5. ✅ `MCPBridge` (tools/mcp_bridge.py) - ~280 lines
   - Connect to MCP servers (stdio and HTTP transports)
   - Load tools from MCP servers
   - Execute MCP tools via bridge
   - Convert to ToolRegistry for use with ChatService

6. ✅ Integration with ChatService
   - Accept `ToolRegistry`, `dict[str, Callable]`, or raw tool definitions
   - Automatic tool execution with registry
   - Pass-through mode for raw definitions

**Files created:**
- `src/mlpal_assistants/tools/__init__.py`
- `src/mlpal_assistants/tools/registry.py`
- `src/mlpal_assistants/tools/executor.py`
- `src/mlpal_assistants/tools/loop.py`
- `src/mlpal_assistants/tools/schema.py`
- `src/mlpal_assistants/tools/result.py`
- `src/mlpal_assistants/tools/mcp_bridge.py`

**Verified working:**
- Tool registration and schema generation
- Tool execution with sync/async functions
- Multi-turn agentic loops
- Integration with chat completions

### Phase 3: Additional Services ✅ COMPLETE

**Goal:** Complete all modalities.

**Status:** Completed on 2026-02-08

**Implemented:**
1. ✅ `ImageService` (services/images.py) - completed in Phase 1
2. ✅ `AudioService` (services/audio.py) - completed in Phase 1
3. ✅ Streaming support (transport/streaming.py) - ~180 lines
   - SSE parsing with proper event handling
   - ChatStream async iterator for streaming responses
   - ChatStreamDelta for incremental content
   - Integration with HTTPTransport and ChatService
   - Facade method `stream_chat()` for streaming chat

**Files created/modified:**
- `src/mlpal_assistants/transport/streaming.py` (new)
- `src/mlpal_assistants/transport/http.py` (added stream_post method)
- `src/mlpal_assistants/transport/__init__.py` (exports streaming types)
- `src/mlpal_assistants/services/chat.py` (added stream method)
- `src/mlpal_assistants/facade.py` (added stream_chat method)
- `src/mlpal_assistants/__init__.py` (exports ChatStream, ChatStreamDelta)

### Phase 4: MCP Integration ✅ COMPLETE

**Goal:** MCP server support.

**Status:** Completed in Phase 2 (2026-02-08)

**Implemented:**
1. ✅ `MCPBridge` class (tools/mcp_bridge.py)
2. ✅ Stdio transport via `connect_stdio()`
3. ✅ HTTP transport via `connect_http()`
4. ✅ Tool discovery with `list_tools()` and refresh
5. ✅ Integration with ToolRegistry via `get_tool_registry()`
6. ✅ Async context manager support
7. ✅ Multi-server connections

### Phase 5: Production Hardening ✅ COMPLETE

**Goal:** Production-ready release.

**Status:** Completed on 2026-02-08

**Implemented:**
1. ✅ Comprehensive test suite (57 tests passing)
   - test_config.py: Config class tests (8 tests)
   - test_exceptions.py: Exception hierarchy tests (7 tests)
   - test_files.py: File handling tests (14 tests)
   - test_tools.py: Tool system tests (17 tests)
   - test_streaming.py: Streaming support tests (11 tests)

2. ✅ Type checking with mypy --strict
   - All 32 source files pass mypy strict mode
   - Proper type parameters for all dict types
   - Complete return type annotations
   - Fixed import paths

3. Documentation (in code)
   - Comprehensive docstrings throughout
   - Type hints for all public APIs
   - Usage examples in docstrings

**Remaining (optional):**
- Performance profiling (not critical for initial release)
- Example applications (can be added later)
- Integration tests with live API

---

## Dependencies

```toml
[project]
name = "mlpal-assistants"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
mcp = ["mcp>=1.0.0"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-httpx>=0.30.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "python-dotenv>=1.0.0",
]
```

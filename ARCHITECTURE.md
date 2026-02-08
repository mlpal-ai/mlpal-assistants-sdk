# MLPal Assistants SDK - Architecture

## Design Principles

1. **Single Responsibility** - Each module does one thing well
2. **Dependency Inversion** - Depend on abstractions, not concretions
3. **Open/Closed** - Easy to extend without modifying core
4. **Explicit over Implicit** - No magic, clear data flow
5. **Type Safety** - Strong typing at all boundaries

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Public API Layer                        │
│  Assistants (facade) - thin orchestration, delegates to     │
│  services                                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                           │
│  ChatService, EmbeddingService, ImageService, AudioService  │
│  - Business logic                                            │
│  - Request building                                          │
│  - Response parsing                                          │
│  - Tool execution (ChatService only)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Transport Layer                          │
│  HTTPTransport - sends requests, receives responses          │
│  - Connection pooling                                        │
│  - Auth headers                                              │
│  - Raw error handling (HTTP status → exception)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Models Layer                            │
│  Request models (Pydantic) - validated, typed                │
│  Response models (dataclasses) - immutable, typed            │
│  Domain models - File, Tool, Message                         │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/mlpal_assistants/
├── __init__.py              # Public exports only
├── _version.py
│
├── client.py                # Assistants facade (thin, ~100 lines)
│
├── services/                # Business logic layer
│   ├── __init__.py
│   ├── base.py              # BaseService with shared logic
│   ├── chat.py              # ChatService (~150 lines)
│   ├── embeddings.py        # EmbeddingService (~50 lines)
│   ├── images.py            # ImageService (~50 lines)
│   └── audio.py             # AudioService (~80 lines)
│
├── tools/                   # Tool execution (separate concern)
│   ├── __init__.py
│   ├── executor.py          # ToolExecutor
│   ├── registry.py          # ToolRegistry (validates, stores tools)
│   └── schema.py            # Tool schema generation
│
├── transport/               # HTTP layer
│   ├── __init__.py
│   ├── http.py              # HTTPTransport
│   └── streaming.py         # SSE streaming
│
├── models/                  # All data types
│   ├── __init__.py
│   ├── requests.py          # Request models (Pydantic for validation)
│   ├── responses.py         # Response models (dataclasses)
│   ├── messages.py          # Message types
│   ├── files.py             # File class
│   └── common.py            # Shared types (CostInfo, etc.)
│
├── config.py                # Configuration
└── exceptions.py            # Exception hierarchy
```

## Key Design Decisions

### 1. Facade Pattern for Client

The `Assistants` class is a thin facade that delegates to services:

```python
class Assistants:
    """Facade - delegates to specialized services."""

    def __init__(self, ...):
        self._transport = HTTPTransport(config)
        self._chat = ChatService(self._transport)
        self._embeddings = EmbeddingService(self._transport)
        self._images = ImageService(self._transport)
        self._audio = AudioService(self._transport)

    async def chat(self, messages, **kwargs) -> ChatResponse:
        return await self._chat.complete(messages, **kwargs)

    async def embed(self, input, **kwargs) -> EmbeddingResponse:
        return await self._embeddings.create(input, **kwargs)
```

### 2. Service Layer with Clear Responsibilities

Each service handles one domain:

```python
class ChatService:
    """Handles chat completions."""

    def __init__(self, transport: HTTPTransport):
        self._transport = transport
        self._tool_executor = ToolExecutor()

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str = "mlpal",
        response_format: type[BaseModel] | None = None,
        tools: ToolRegistry | None = None,
        **kwargs,
    ) -> ChatResponse:
        # 1. Build request
        request = self._build_request(messages, model, response_format, tools, **kwargs)

        # 2. Execute (with tool loop if needed)
        if tools:
            return await self._execute_with_tools(request, tools)
        return await self._execute_single(request, response_format)
```

### 3. Typed Request/Response Models

Use Pydantic for requests (validation) and dataclasses for responses (immutability):

```python
# requests.py
class ChatRequest(BaseModel):
    """Validated chat request."""
    model: str
    messages: list[MessageParam]
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: ResponseFormatParam | None = None
    tools: list[ToolParam] | None = None

    model_config = ConfigDict(extra="forbid")  # Catch typos

# responses.py
@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Immutable chat response."""
    content: str | None
    cost: CostInfo
    routing: RoutingInfo
    ...
```

### 4. Tool Execution as Separate Concern

Tools are handled by dedicated classes, not embedded in ChatService:

```python
class ToolRegistry:
    """Validates and stores tool definitions."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, func: Callable, description: str | None = None):
        """Register a tool with validation."""
        schema = self._generate_schema(func)
        self._tools[name] = ToolDefinition(name=name, func=func, schema=schema)

    def to_api_format(self) -> list[dict]:
        """Convert to API request format."""
        return [t.to_api_format() for t in self._tools.values()]


class ToolExecutor:
    """Executes tool calls."""

    async def execute(
        self,
        tool_calls: list[ToolCall],
        registry: ToolRegistry,
    ) -> list[ToolResult]:
        """Execute tool calls and return results."""
        results = []
        for call in tool_calls:
            result = await self._execute_one(call, registry)
            results.append(result)
        return results
```

### 5. Transport Layer Focuses Only on HTTP

```python
class HTTPTransport:
    """Handles HTTP communication only."""

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Send request and return parsed JSON."""
        response = await self._client.request(...)
        self._raise_for_status(response)  # Maps HTTP errors to exceptions
        return response.json()
```

### 6. File Class with Factory Methods

```python
@dataclass
class File:
    """Immutable file attachment."""
    type: FileType
    data: FileData  # Union of UrlData, Base64Data
    mime_type: str | None

    @classmethod
    def from_path(cls, path: Path | str) -> File:
        """Create from local file path."""
        ...

    @classmethod
    def from_bytes(cls, data: bytes, *, mime_type: str) -> File:
        """Create from bytes."""
        ...

    @classmethod
    def from_url(cls, url: str) -> File:
        """Create from URL (passed through to API)."""
        ...

    def to_api_format(self) -> dict:
        """Serialize for API request."""
        ...
```

## Anti-Patterns Avoided

| Anti-Pattern | Solution |
|--------------|----------|
| God Object | Facade + Services split responsibilities |
| Mixed concerns | Layered architecture |
| Weak typing | Pydantic models with validation |
| Magic strings | Enums and constants |
| Inline imports | Proper imports at module level |
| Dict manipulation | Typed models throughout |
| Tight coupling | Dependency injection via constructor |

## Testing Strategy

1. **Unit Tests** - Test each layer in isolation with mocks
2. **Integration Tests** - Test service layer with real transport
3. **Contract Tests** - Validate request/response shapes match API

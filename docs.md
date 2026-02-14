# MLPal Assistants SDK Documentation

Complete guide to the MLPal Assistants Python SDK.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Chat Completions](#chat-completions)
4. [Structured Output](#structured-output)
5. [Tool Use](#tool-use)
6. [Streaming](#streaming)
7. [File Attachments](#file-attachments)
8. [Embeddings](#embeddings)
9. [Image Generation](#image-generation)
10. [Text-to-Speech](#text-to-speech)
11. [Transcription](#transcription)
12. [MCP Integration](#mcp-integration)
13. [Error Handling](#error-handling)
14. [Configuration](#configuration)
15. [Type Reference](#type-reference)

---

## Installation

```bash
# Using pip (from MLPal private PyPI)
pip install mlpal-assistants --index-url https://pypi.mlpal.ai/simple/

# Using uv
uv add mlpal-assistants --index-url https://pypi.mlpal.ai/simple/

# With MCP support
pip install "mlpal-assistants[mcp]" --index-url https://pypi.mlpal.ai/simple/
```

**Requirements:** Python 3.12+

---

## Quick Start

```python
import asyncio
from mlpal_assistants import Assistant

async def main():
    # Create client (uses MLPAL_API_KEY environment variable)
    async with Assistant() as assistant:
        # Simple string shorthand
        response = await assistant.chat("Hello!")
        print(response.content)

        # Or full messages format
        response = await assistant.chat(
            messages=[{"role": "user", "content": "Hello!"}]
        )

asyncio.run(main())
```

---

## Chat Completions

### Basic Chat

```python
# Simple string shorthand - easiest way!
response = await assistant.chat("What is Python?")
print(response.content)

# Full messages format for more control
response = await assistant.chat(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
)

print(response.content)          # Response text
print(response.finish_reason)    # "stop", "tool_calls", etc.
print(response.cost.tokens)      # Token usage
print(response.routing)          # Model routing info
```

### Chat Parameters

```python
response = await assistant.chat(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o",              # Model to use (default: "mlpal")
    temperature=0.7,             # Sampling temperature (0-2)
    max_tokens=1000,             # Maximum tokens to generate
    top_p=0.9,                   # Nucleus sampling
    stop=["END"],                # Stop sequences
)
```

### Multi-turn Conversation

```python
messages = [
    {"role": "user", "content": "My name is Alice."},
]

response = await assistant.chat(messages=messages)

# Use .message property to add response to history
messages.append(response.message)

messages.append({"role": "user", "content": "What's my name?"})
response = await assistant.chat(messages=messages)
print(response.content)  # "Your name is Alice."
```

---

## Structured Output

Get typed responses using Pydantic models.

### Basic Usage

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

response = await assistant.chat(
    messages=[{"role": "user", "content": "John is 30 years old"}],
    response_format=Person,
)

# Access typed data
person = response.data
print(person.name)  # "John"
print(person.age)   # 30
```

### Nested Models

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    address: Address
    employee_count: int

response = await assistant.chat(
    messages=[{"role": "user", "content": "Tell me about Acme Corp in NYC"}],
    response_format=Company,
)

print(response.data.address.city)  # "New York"
```

### Lists and Optional Fields

```python
class Product(BaseModel):
    name: str
    price: float
    tags: list[str] = []
    description: str | None = None

class Catalog(BaseModel):
    products: list[Product]

response = await assistant.chat(
    messages=[{"role": "user", "content": "List 3 tech products"}],
    response_format=Catalog,
)

for product in response.data.products:
    print(f"{product.name}: ${product.price}")
```

---

## Tool Use

The SDK automatically executes tools and handles multi-turn conversations.

### Using ToolRegistry

```python
from mlpal_assistants import ToolRegistry

tools = ToolRegistry()

# Register with decorator
@tools.tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get weather for a city.

    Args:
        city: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    return f"Weather in {city}: 22°C, sunny"

# Register directly
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

tools.register("calculate", calculate)

# Use in chat
response = await assistant.chat(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    max_turns=5,  # Maximum tool execution rounds
)
```

### Quick Registration

```python
# From dict of functions
response = await assistant.chat(
    messages=[{"role": "user", "content": "Calculate 15 * 7"}],
    tools={"calculate": lambda expr: str(eval(expr))},
)
```

### Async Tools

```python
@tools.tool
async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text
```

### Tool Execution Control

```python
response = await assistant.chat(
    messages=messages,
    tools=tools,
    max_turns=10,                    # Max tool execution rounds
    tool_choice="auto",              # "auto", "none", or specific tool
    parallel_tool_calls=True,        # Allow parallel execution
)

# Access turn history
if response.turns:
    for turn in response.turns:
        print(f"Tool calls: {len(turn.tool_results or [])}")
```

---

## Streaming

Stream responses in real-time.

### Basic Streaming

```python
async with assistant.stream_chat(
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    async for delta in stream:
        if delta.content:
            print(delta.content, end="", flush=True)
    print()  # Newline at end
```

### Stream with Metadata

```python
async with assistant.stream_chat(messages=messages) as stream:
    async for delta in stream:
        if delta.content:
            print(delta.content, end="")

        if delta.finish_reason:
            print(f"\nFinished: {delta.finish_reason}")

    # After streaming
    full_content = stream.collected_content
    print(f"Total length: {len(full_content)}")
```

### Collect All Content

```python
async with assistant.stream_chat(messages=messages) as stream:
    content = await stream.collect()  # Consumes stream, returns full content
    print(content)
```

---

## File Attachments

Attach images, audio, or documents to messages.

### From Local File

```python
from mlpal_assistants import File

response = await assistant.chat(
    messages=[{
        "role": "user",
        "content": "Describe this image",
        "files": [File.from_path("photo.jpg")]
    }]
)
```

### From URL

```python
# URL is passed to API (not downloaded locally)
image = File.from_url("https://example.com/image.jpg")

response = await assistant.chat(
    messages=[{
        "role": "user",
        "content": "What's in this image?",
        "files": [image]
    }]
)
```

### From Bytes

```python
# For in-memory data
image_bytes = open("photo.png", "rb").read()
image = File.from_bytes(image_bytes, mime_type="image/png")

response = await assistant.chat(
    messages=[{
        "role": "user",
        "content": "Analyze this image",
        "files": [image]
    }]
)
```

### Convenience Factories

```python
# Explicit type
image = File.image("photo.jpg")
audio = File.audio("recording.mp3")
document = File.document("report.pdf")
```

### Multiple Files

```python
response = await assistant.chat(
    messages=[{
        "role": "user",
        "content": "Compare these images",
        "files": [
            File.from_path("image1.jpg"),
            File.from_path("image2.jpg"),
        ]
    }]
)
```

---

## Embeddings

Generate text embeddings.

### Single Text

```python
response = await assistant.embed("Hello, world!")

embedding = response.data[0].embedding  # list[float]
print(f"Dimensions: {len(embedding)}")
```

### Batch Embeddings

```python
response = await assistant.embed([
    "First sentence",
    "Second sentence",
    "Third sentence",
])

for item in response.data:
    print(f"Index {item.index}: {len(item.embedding)} dimensions")
```

### With Model Selection

```python
response = await assistant.embed(
    input=["Hello"],
    model="text-embedding-3-small",
)
```

---

## Image Generation

Generate images from text prompts.

```python
response = await assistant.generate_image(
    prompt="A sunset over mountains, oil painting style",
    size="1024x1024",      # Image size
    quality="standard",     # "standard" or "hd"
    n=1,                   # Number of images
    style="vivid",         # Style hint
)

for image in response.images:
    print(f"URL: {image.url}")
    if image.revised_prompt:
        print(f"Revised prompt: {image.revised_prompt}")
```

---

## Text-to-Speech

Convert text to speech audio.

```python
response = await assistant.generate_speech(
    input="Hello, how are you today?",
    voice="alloy",              # Voice selection
    response_format="mp3",      # Audio format
    speed=1.0,                  # Speed (0.25 - 4.0)
)

# Save to file
response.save("greeting.mp3")

# Or access bytes directly
audio_bytes = response.audio.content
```

### Available Voices

- `alloy` - Neutral
- `echo` - Male
- `fable` - British
- `onyx` - Deep male
- `nova` - Female
- `shimmer` - Soft female

---

## Transcription

Transcribe audio to text.

```python
response = await assistant.transcribe(
    "audio.mp3",                # Path, bytes, or file object
    language="en",              # Optional language hint
    prompt="Technical podcast", # Optional context
)

print(response.text)
print(response.language)  # Detected language
```

### From Different Sources

```python
# From path
response = await assistant.transcribe("audio.mp3")

# From bytes
audio_bytes = open("audio.wav", "rb").read()
response = await assistant.transcribe(audio_bytes)

# From file object
with open("audio.mp3", "rb") as f:
    response = await assistant.transcribe(f)
```

---

## MCP Integration

Connect to Model Context Protocol servers for extended tool capabilities.

### Setup

```bash
pip install "mlpal-assistants[mcp]" --index-url https://pypi.mlpal.ai/simple/
```

### Connect to MCP Server

```python
from mlpal_assistants.tools import MCPBridge

async with MCPBridge() as bridge:
    # Connect via stdio
    await bridge.connect_stdio(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    # Or via HTTP
    await bridge.connect_http(
        name="api-server",
        url="http://localhost:8080/mcp",
    )

    # Get tools as ToolRegistry
    tools = bridge.get_tool_registry()

    # Use with assistant
    response = await assistant.chat(
        messages=[{"role": "user", "content": "List files in /tmp"}],
        tools=tools,
    )
```

### Multi-Server Setup

```python
async with MCPBridge() as bridge:
    await bridge.connect_stdio("fs", "npx", ["-y", "@mcp/server-filesystem", "/"])
    await bridge.connect_stdio("git", "npx", ["-y", "@mcp/server-git"])

    print(f"Connected: {bridge.connected_servers}")
    print(f"Available tools: {bridge.available_tools}")

    tools = bridge.get_tool_registry()
    # Use combined tools...
```

### Config File (mlpal.json)

Auto-discover and connect MCP servers from a `mlpal.json` config file:

```json
{
  "mcpServers": {
    "perplexity": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "server-perplexity-ask"],
      "env": {
        "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"
      }
    },
    "api-server": {
      "transport": "http",
      "url": "http://localhost:8080/mcp",
      "headers": {
        "Authorization": "Bearer ${API_TOKEN}"
      }
    }
  }
}
```

Use the config in code:

```python
async with MCPBridge() as bridge:
    # Auto-discover mlpal.json in current directory or parents
    await bridge.connect_from_config()

    # Or connect specific servers only
    await bridge.connect_from_config(server_names=["perplexity"])

    # Or load from explicit path
    await bridge.connect_from_config(config_path="/path/to/mlpal.json")

    tools = bridge.get_tool_registry()
```

Environment variables are interpolated using `${VAR_NAME}` syntax.

### Merging Tools

Combine MCP tools with regular tools:

```python
from mlpal_assistants import ToolRegistry
from mlpal_assistants.tools import MCPBridge

# Create regular tools
tools = ToolRegistry()

@tools.tool
def search(query: str) -> str:
    """Search locally."""
    return "local results"

# Add MCP tools
async with MCPBridge() as bridge:
    await bridge.connect_from_config()
    mcp_tools = bridge.get_tool_registry()

    # Merge MCP tools into regular tools
    tools.merge(mcp_tools)

    # Or create a new combined registry
    combined = ToolRegistry.merged(tools, mcp_tools)
```

---

## Error Handling

The SDK provides a detailed exception hierarchy.

### Exception Types

```python
from mlpal_assistants import (
    AssistantsError,        # Base exception
    AuthenticationError,    # Invalid API key (401)
    PermissionDeniedError,  # Access denied (403)
    RateLimitError,         # Rate limited (429)
    BadRequestError,        # Invalid request (400)
    ValidationError,        # Validation failed
    NotFoundError,          # Resource not found (404)
    ModelNotAvailableError, # Model unavailable
    ServerError,            # Server error (5xx)
    TimeoutError,           # Request timeout
    ConnectionError,        # Connection failed
    ToolExecutionError,     # Tool execution failed
    MaxTurnsExceededError,  # Agentic loop limit
    MCPError,               # MCP-related error
)
```

### Handling Errors

```python
from mlpal_assistants import AssistantsError, RateLimitError

try:
    response = await assistant.chat(messages=messages)
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except AssistantsError as e:
    print(f"Error: {e}")
    print(f"Status code: {e.status_code}")
    print(f"Response: {e.response}")
```

### Tool Execution Errors

```python
from mlpal_assistants import ToolExecutionError

try:
    response = await assistant.chat(messages=messages, tools=tools)
except ToolExecutionError as e:
    print(f"Tool '{e.tool_name}' failed: {e.original_error}")
```

---

## Configuration

### Environment Variables

```bash
export MLPAL_API_KEY="mlpal_sk_..."
export MLPAL_BASE_URL="https://models.mlpal.ai"  # Optional
export MLPAL_TIMEOUT="120"  # Optional, in seconds
```

### Explicit Configuration

```python
from mlpal_assistants import Assistant

assistant = Assistant(
    api_key="mlpal_sk_...",
    base_url="https://models.mlpal.ai",
    timeout=120.0,
)
```

### Context Manager

```python
# Recommended - ensures proper cleanup
async with Assistant() as assistant:
    response = await assistant.chat(messages=messages)

# Manual cleanup
assistant = Assistant()
try:
    response = await assistant.chat(messages=messages)
finally:
    await assistant.close()
```

---

## Type Reference

### Response Types

| Type | Description |
|------|-------------|
| `ChatResponse` | Chat completion result |
| `EmbeddingResponse` | Embedding result |
| `ImageResponse` | Image generation result |
| `AudioResponse` | TTS result |
| `TranscriptionResponse` | Transcription result |

### ChatResponse Fields

```python
response.content        # str | None - Response text
response.role           # "assistant"
response.finish_reason  # "stop", "tool_calls", "length"
response.data           # BaseModel | None - Parsed structured output
response.tool_calls     # list[ToolCall] | None
response.turns          # list[Turn] | None - Agentic turn history
response.cost           # CostInfo - Token/cost info
response.routing        # RoutingInfo - Model routing info
```

### CostInfo Fields

```python
cost.model_name     # str - Model used
cost.provider       # str - Provider name
cost.tokens         # TokenUsage - Token counts
cost.latency_ms     # int - Request latency
cost.compute_units  # float - Compute units used
```

### TokenUsage Fields

```python
tokens.input_tokens   # int
tokens.output_tokens  # int
tokens.total_tokens   # int
tokens.cached_tokens  # int | None
```

### ToolCall Fields

```python
tool_call.id                # str - Unique ID
tool_call.type              # "function"
tool_call.function.name     # str - Function name
tool_call.function.arguments  # str - JSON arguments
```

---

## Best Practices

1. **Use context managers** for automatic cleanup
2. **Handle errors** with specific exception types
3. **Use structured output** for reliable data extraction
4. **Set appropriate timeouts** for your use case
5. **Use streaming** for long responses to improve UX
6. **Batch embeddings** when possible for efficiency

---

## More Examples

See the [examples/](examples/) directory for runnable scripts:

- `basic_chat.py` - Simple chat completion
- `structured_output.py` - Pydantic model extraction
- `tool_use.py` - Tool registration and execution
- `streaming.py` - Real-time streaming
- `file_attachments.py` - Image and file handling

# MLPal Assistants SDK

Official Python SDK for the MLPal Assistants API.

## Installation

```bash
pip install mlpal-assistants

# Or with uv
uv add mlpal-assistants

# With MCP support
pip install mlpal-assistants[mcp]
```

## Quick Start

```python
import asyncio
from mlpal_assistants import Assistants, File

async def main():
    async with Assistants() as assistant:  # Uses MLPAL_API_KEY env var
        # Simple chat
        response = await assistant.chat(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.content)

asyncio.run(main())
```

## Features

- **Unified chat interface** for text, files, tools, and structured output
- **Automatic tool execution** - SDK runs agentic loops when tools are provided
- **Structured output** - Pass a Pydantic model, get a typed instance back
- **Streaming** - Real-time response streaming with async iterators
- **File handling** - `File` class handles paths, bytes, and URLs
- **MCP Integration** - Connect to Model Context Protocol servers
- **Type-safe** - Full type annotations, mypy --strict compliant
- **Async-first** - Built for high-performance concurrent workloads

## Examples

### Chat with Files

```python
from mlpal_assistants import Assistants, File

async with Assistants() as assistant:
    response = await assistant.chat(
        messages=[{
            "role": "user",
            "content": "What's in this image?",
            "files": [File.from_path("photo.jpg")]
        }]
    )
```

### Structured Output

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

response = await assistant.chat(
    messages=[{"role": "user", "content": "John is 30 years old"}],
    response_format=Person,
)
print(response.data.name)  # "John"
print(response.data.age)   # 30
```

### Tool Use

```python
from mlpal_assistants import ToolRegistry

tools = ToolRegistry()

@tools.tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 22°C, sunny"

response = await assistant.chat(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
)
# SDK automatically executes tools and returns final response
```

### Streaming

```python
async with assistant.stream_chat(
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    async for delta in stream:
        if delta.content:
            print(delta.content, end="", flush=True)
```

### Other Capabilities

```python
# Embeddings
response = await assistant.embed(["Hello", "World"])

# Image generation
response = await assistant.generate_image(prompt="A sunset over mountains")

# Text-to-speech
response = await assistant.generate_speech(text="Hello!", voice="alloy")
response.save("greeting.mp3")

# Transcription
response = await assistant.transcribe(File.from_path("audio.mp3"))
print(response.text)
```

## Configuration

```python
from mlpal_assistants import Assistants

# From environment (recommended)
assistant = Assistants()  # Uses MLPAL_API_KEY

# Explicit configuration
assistant = Assistants(
    api_key="mlpal_sk_...",
    base_url="https://models.mlpal.ai",
    timeout=120.0,
)
```

Environment variables:
- `MLPAL_API_KEY` - API key (required)
- `MLPAL_BASE_URL` - Base URL (optional)
- `MLPAL_TIMEOUT` - Request timeout in seconds (optional)

## Documentation

See [docs.md](docs.md) for comprehensive documentation.

## License

MIT - see [LICENSE](LICENSE)

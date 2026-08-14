# mlpal-assistants

[![PyPI](https://img.shields.io/pypi/v/mlpal-assistants)](https://pypi.org/project/mlpal-assistants/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Python SDK for the MLPal Gateway — the managed service at `models.mlpal.ai` or
a [self-hosted gateway](https://github.com/mlpal-ai/mlpal-gateway). One client
for inference (Anthropic-wire messages, streaming, tools) and management (keys,
model policies, spend budgets, usage).

## Installation

```bash
pip install mlpal-assistants          # or: uv add mlpal-assistants
pip install "mlpal-assistants[mcp]"   # with MCP support
```

## Quick Start

Two clients ship in this package:

- **`MLPal` / `AsyncMLPal`** — the v2 client. One surface for **native inference**
  (the Anthropic Messages wire) and **management** (keys, model policy, spend
  budgets). This is what application teams standardize on. Point `base_url` at
  the managed service (default) or a self-hosted gateway — the surface is identical.
- **`Assistant`** — the v1 agentic convenience layer (automatic tool loops,
  structured output, file helpers). Kept for compatibility.

```python
from mlpal_assistants import MLPal

client = MLPal()  # uses MLPAL_API_KEY (+ optional MLPAL_BASE_URL)

# `mlpal` is a router tag — the gateway resolves it to the best model the
# deployment serves. Any concrete tag (claude-opus-5, gpt-5.6-terra, ...)
# works too; the response always comes back in this Anthropic shape.
msg = client.messages.create(
    model="mlpal",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(msg.text)

# Cost: the gateway reports compute units on every non-streaming response —
# the model's pass-through cost, no markup.
if msg.compute_units is not None:
    print(msg.compute_units)

# Streaming
with client.messages.stream(model="mlpal", max_tokens=512,
                            messages=[{"role": "user", "content": "Count to five."}]) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            print(event.data["delta"].get("text", ""), end="", flush=True)

# Management — issue a scoped key with a model policy and a monthly budget
key = client.admin.keys.create(
    name="team-web",
    permissions=["messages"],
    model_policy={"allow": ["claude-*", "mlpal*"], "deny": []},
    budgets=[{"unit": "usd", "amount": 100, "window": "month"}],
)
print(key.secret)  # shown once
```

Also on the client:

```python
# Discover routable models / curated tiers (route a subtask by complexity)
catalog = client.catalog.retrieve(profile="coding")
tier = catalog.tiers[catalog.routing_ladder[0]]   # cheapest served tier
client.models.list()                               # capability advertisement

# Close the curation loop — report how a delegated subtask turned out
client.feedback.create(model="gpt-5-nano", task_type="coding",
                       outcome="escalated", escalated_to="claude-opus-5")

# This account's own usage
client.usage.summary()
client.usage.daily(days=7)
```

`AsyncMLPal` mirrors the same surface with `await` and `async with`. See
[`examples/v2_quickstart.py`](examples/v2_quickstart.py).

### v1 agentic client

```python
import asyncio
from mlpal_assistants import Assistant

async def main():
    async with Assistant() as assistant:  # Uses MLPAL_API_KEY env var
        # Simple string shorthand
        response = await assistant.chat("Hello!")
        print(response.content)

        # Or full messages format
        response = await assistant.chat(
            messages=[{"role": "user", "content": "Hello!"}]
        )

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
from mlpal_assistants import Assistant, File

async with Assistant() as assistant:
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
response = await assistant.generate_speech(input="Hello!", voice="alloy")
response.save("greeting.mp3")

# Transcription
response = await assistant.transcribe("audio.mp3")
print(response.text)
```

## Configuration

```python
from mlpal_assistants import Assistant

# From environment (recommended)
assistant = Assistant()  # Uses MLPAL_API_KEY

# Explicit configuration
assistant = Assistant(
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

## License and contact

Apache-2.0 — see [LICENSE](LICENSE). Questions and issues: **contact@mlpal.ai**
or [GitHub issues](https://github.com/mlpal-ai/mlpal-assistants-sdk/issues).

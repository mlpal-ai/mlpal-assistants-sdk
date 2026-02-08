# MLPal Assistants SDK

Official Python SDK for the MLPal Assistants API.

## Installation

```bash
uv add mlpal-assistants
```

## Quick Start

```python
import asyncio
from mlpal_assistants import Assistants, File

async def main():
    assistant = Assistants()  # Uses MLPAL_API_KEY env var

    # Simple chat
    response = await assistant.chat(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.content)

    # With image
    response = await assistant.chat(
        messages=[{
            "role": "user",
            "content": "What's in this image?",
            "files": [File("photo.jpg")]
        }]
    )

asyncio.run(main())
```

## Features

- **Single `chat()` interface** for text, files, tools, and structured output
- **Automatic tool execution** - SDK runs agentic loops when tools are provided
- **Structured output** - Pass a Pydantic model, get an instance back
- **File handling** - `File` class handles paths, bytes, and URLs
- **Async-first** - Built for high-performance concurrent workloads
- **Resilient** - Retries, circuit breakers, and comprehensive error handling

## Structured Output

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
```

## Tool Use

```python
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: 22°C, sunny"

response = await assistant.chat(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools={"get_weather": get_weather},
)
# SDK automatically executes tools and returns final response
print(response.content)
```

## Additional Capabilities

```python
# Embeddings
response = await assistant.embed(["Hello", "World"])

# Image generation
response = await assistant.generate.image(prompt="A sunset over mountains")

# Text-to-speech
response = await assistant.generate.speech(text="Hello!", voice="alloy")
response.save("greeting.mp3")

# Transcription
response = await assistant.transcribe(File("audio.mp3"))
print(response.text)
```

## License

MIT

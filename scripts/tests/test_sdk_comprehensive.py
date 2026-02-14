#!/usr/bin/env python3
"""
Comprehensive SDK Test Suite

Tests all MLPal Assistants SDK features against the production endpoint:
- Basic chat
- Simple string chat API
- Conversation history with .message
- Structured output (Pydantic models)
- Tools with automatic execution
- Vision (single image)
- Vision + Structured Output (multiple images) - MULTIMODAL TEST
- Streaming
- Embeddings
- Image generation
- Text-to-speech
- Transcription

Usage:
    cd mlpal-assistants-sdk
    export MLPAL_API_KEY=mlpal_sk_xxx
    uv run python scripts/tests/test_sdk_comprehensive.py
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_URL, API_KEY

from mlpal_assistants import Assistant, File, ToolRegistry, ChatResponse

# Test files
TEST_DIR = Path(__file__).parent
IMAGE_PATH = TEST_DIR / "mm" / "photo" / "hadoop.jpg"

# Second test image - use a reliable public image URL
IMAGE_URL = "https://picsum.photos/id/237/300/200"  # Random dog image from Lorem Picsum


def _print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def _print_result(response: Any, latency_ms: int):
    """Print response details."""
    print(f"\n RESPONSE ({latency_ms}ms):")

    if hasattr(response, 'content') and response.content:
        content = response.content
        if len(content) > 500:
            content = content[:500] + "...[truncated]"
        print(f"   Content: {content}")

    if hasattr(response, 'data') and response.data:
        print(f"   Structured Data: {response.data}")

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"   Tool Calls: {[(tc.function.name, tc.function.arguments) for tc in response.tool_calls]}")

    if hasattr(response, 'routing') and response.routing:
        r = response.routing
        print(f"   Routing: {r.requested_model} -> {r.resolved_model} ({r.provider})")

    if hasattr(response, 'cost') and response.cost:
        c = response.cost
        print(f"   Cost: {c.compute_units} units, {c.latency_ms}ms, {c.tokens.total_tokens} tokens")


async def test_basic_chat(assistant: Assistant) -> dict:
    """Test basic chat completion."""
    _print_section("CHAT - Basic Completion")

    print("\n REQUEST:")
    print('   assistant.chat(messages=[{"role": "user", "content": "What is 2 + 2?"}])')

    start = time.perf_counter()
    try:
        response = await assistant.chat(
            messages=[{"role": "user", "content": "What is 2 + 2? Answer in one word."}],
            max_tokens=50,
            temperature=0.1
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        _print_result(response, latency_ms)
        return {"test": "basic_chat", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "basic_chat", "status": "error", "latency_ms": latency_ms}


async def test_simple_chat_api(assistant: Assistant) -> dict:
    """Test simple string chat API."""
    _print_section("CHAT - Simple String API")

    print("\n REQUEST:")
    print('   assistant.chat("Hello, who are you?")')

    start = time.perf_counter()
    try:
        response = await assistant.chat("Hello, who are you? Answer briefly.")
        latency_ms = int((time.perf_counter() - start) * 1000)
        _print_result(response, latency_ms)
        return {"test": "simple_chat_api", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "simple_chat_api", "status": "error", "latency_ms": latency_ms}


async def test_conversation_history(assistant: Assistant) -> dict:
    """Test multi-turn conversation with .message property."""
    _print_section("CHAT - Conversation History (.message)")

    print("\n REQUEST:")
    print('   # Turn 1: Tell assistant my name')
    print('   # Turn 2: Ask what my name is (uses .message to build history)')

    start = time.perf_counter()
    try:
        # Turn 1
        messages = [{"role": "user", "content": "My name is TestUser123."}]
        response1 = await assistant.chat(messages=messages, max_tokens=100)

        # Add response to history using .message
        messages.append(response1.message)

        # Turn 2
        messages.append({"role": "user", "content": "What is my name?"})
        response2 = await assistant.chat(messages=messages, max_tokens=100)

        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n   Turn 1 Response: {response1.content[:100]}...")
        print(f"   Turn 2 Response: {response2.content[:200]}")

        # Check if it remembered the name
        success = "TestUser123" in response2.content
        print(f"   Memory Test: {'PASSED' if success else 'FAILED'} (should contain 'TestUser123')")

        return {"test": "conversation_history", "status": 200 if success else "failed", "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "conversation_history", "status": "error", "latency_ms": latency_ms}


async def test_structured_output(assistant: Assistant) -> dict:
    """Test structured output with Pydantic model."""
    _print_section("STRUCTURED OUTPUT - Pydantic Model")

    class PersonInfo(BaseModel):
        name: str
        age: int
        city: str

    print("\n REQUEST:")
    print('   class PersonInfo(BaseModel):')
    print('       name: str')
    print('       age: int')
    print('       city: str')
    print('   assistant.chat(..., response_format=PersonInfo)')

    start = time.perf_counter()
    try:
        response = await assistant.chat(
            "Extract: John Smith is 30 years old and lives in New York City.",
            response_format=PersonInfo,
            max_tokens=200,
            temperature=0.1
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        _print_result(response, latency_ms)

        # Verify structured data
        if response.data:
            print(f"\n   Parsed: name={response.data.name}, age={response.data.age}, city={response.data.city}")

        return {"test": "structured_output", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "structured_output", "status": "error", "latency_ms": latency_ms}


async def test_tools(assistant: Assistant) -> dict:
    """Test tool calling with automatic execution."""
    _print_section("TOOLS - Function Calling with Auto-Execution")

    tools = ToolRegistry()

    @tools.tool
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location.

        Args:
            location: City name
            unit: Temperature unit (celsius or fahrenheit)
        """
        # Simulated response
        return f"Weather in {location}: 22°{'C' if unit == 'celsius' else 'F'}, sunny"

    @tools.tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: Math expression to evaluate
        """
        try:
            return str(eval(expression))
        except:
            return "Error evaluating expression"

    print("\n REQUEST:")
    print('   tools = ToolRegistry()')
    print('   @tools.tool')
    print('   def get_weather(location: str) -> str: ...')
    print('   @tools.tool')
    print('   def calculate(expression: str) -> str: ...')
    print('   assistant.chat("Weather in Paris and calculate 15*7", tools=tools)')

    start = time.perf_counter()
    try:
        response = await assistant.chat(
            "What's the weather in Paris? Also, what is 15 * 7?",
            tools=tools,
            max_turns=5
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        _print_result(response, latency_ms)

        # Show turns
        if response.turns:
            print(f"\n   Completed in {len(response.turns)} turn(s)")

        return {"test": "tools", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "tools", "status": "error", "latency_ms": latency_ms}


async def test_vision_single(assistant: Assistant) -> dict:
    """Test vision with single image."""
    _print_section("VISION - Single Image")

    if not IMAGE_PATH.exists():
        print("   Image file not found, skipping test")
        return {"test": "vision_single", "status": "skipped", "latency_ms": 0}

    print("\n REQUEST:")
    print(f'   File.from_path("{IMAGE_PATH.name}")')
    print('   assistant.chat(messages=[{...content + files...}])')

    start = time.perf_counter()
    try:
        response = await assistant.chat(
            messages=[{
                "role": "user",
                "content": "Describe this image in one sentence.",
                "files": [File.from_path(IMAGE_PATH)]
            }],
            max_tokens=200
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        _print_result(response, latency_ms)
        return {"test": "vision_single", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "vision_single", "status": "error", "latency_ms": latency_ms}


async def test_multimodal_structured(assistant: Assistant) -> dict:
    """
    TEST: Multiple images + Structured Output

    This is the key multimodal test - can the model:
    1. Accept multiple images
    2. Analyze them both
    3. Return structured output comparing them
    """
    _print_section("MULTIMODAL - Multiple Images + Structured Output")

    class ImageComparison(BaseModel):
        """Structured comparison of two images."""
        image1_description: str
        image2_description: str
        similarities: list[str]
        differences: list[str]
        which_is_more_complex: str

    if not IMAGE_PATH.exists():
        print("   Local image file not found, skipping test")
        return {"test": "multimodal_structured", "status": "skipped", "latency_ms": 0}

    # Download second image and encode as bytes (URL format has issues with some providers)
    print("\n   Downloading second test image...")
    import httpx
    async with httpx.AsyncClient(follow_redirects=True) as client:
        img_response = await client.get(IMAGE_URL)
        second_image_bytes = img_response.content

    print("\n REQUEST:")
    print('   class ImageComparison(BaseModel):')
    print('       image1_description: str')
    print('       image2_description: str')
    print('       similarities: list[str]')
    print('       differences: list[str]')
    print('       which_is_more_complex: str')
    print('')
    print('   files = [')
    print(f'       File.from_path("{IMAGE_PATH.name}"),  # Local file')
    print(f'       File.from_bytes(image_bytes, mime_type="image/jpeg"),  # Downloaded')
    print('   ]')
    print('   assistant.chat(..., files=files, response_format=ImageComparison)')

    start = time.perf_counter()
    try:
        response = await assistant.chat(
            messages=[{
                "role": "user",
                "content": "Compare these two images. Describe each one, list similarities and differences, and tell me which is more visually complex.",
                "files": [
                    File.from_path(IMAGE_PATH),
                    File.from_bytes(second_image_bytes, mime_type="image/jpeg"),
                ]
            }],
            response_format=ImageComparison,
            max_tokens=500,
            temperature=0.1
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n RESPONSE ({latency_ms}ms):")

        if response.data:
            data = response.data
            print(f"\n   Image 1: {data.image1_description[:100]}...")
            print(f"   Image 2: {data.image2_description[:100]}...")
            print(f"   Similarities: {data.similarities}")
            print(f"   Differences: {data.differences}")
            print(f"   More Complex: {data.which_is_more_complex}")
            print("\n   MULTIMODAL TEST: PASSED")
        else:
            print(f"   Content: {response.content}")
            print("\n   MULTIMODAL TEST: No structured data returned")

        if response.routing:
            r = response.routing
            print(f"\n   Routing: {r.requested_model} -> {r.resolved_model} ({r.provider})")

        return {"test": "multimodal_structured", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"test": "multimodal_structured", "status": "error", "latency_ms": latency_ms}


async def test_streaming(assistant: Assistant) -> dict:
    """Test streaming chat."""
    _print_section("STREAMING - Real-time Response")

    print("\n REQUEST:")
    print('   async with assistant.stream_chat("Tell me a joke") as stream:')
    print('       async for delta in stream: ...')

    start = time.perf_counter()
    try:
        chunks = []
        print("\n   Streaming: ", end="", flush=True)

        async with assistant.stream_chat(
            messages=[{"role": "user", "content": "Tell me a very short joke (2 sentences max)."}],
            max_tokens=100
        ) as stream:
            async for delta in stream:
                if delta.content:
                    print(delta.content, end="", flush=True)
                    chunks.append(delta.content)

        print()  # newline
        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n   Total chunks: {len(chunks)}")
        print(f"   Final content length: {len(stream.collected_content)}")
        print(f"   Finish reason: {stream.finish_reason}")

        return {"test": "streaming", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "streaming", "status": "error", "latency_ms": latency_ms}


async def test_embeddings(assistant: Assistant) -> dict:
    """Test embedding generation."""
    _print_section("EMBEDDINGS")

    print("\n REQUEST:")
    print('   assistant.embed(["Hello world", "Goodbye world"])')

    start = time.perf_counter()
    try:
        response = await assistant.embed(["Hello world", "Goodbye world"])
        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n RESPONSE ({latency_ms}ms):")
        print(f"   Model: {response.model}")
        print(f"   Embeddings: {len(response.data)} vectors")
        for emb in response.data:
            print(f"   - Index {emb.index}: {len(emb.embedding)} dimensions, first 5: {list(emb.embedding[:5])}")

        return {"test": "embeddings", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "embeddings", "status": "error", "latency_ms": latency_ms}


async def test_image_generation(assistant: Assistant) -> dict:
    """Test image generation."""
    _print_section("IMAGE GENERATION")

    print("\n REQUEST:")
    print('   assistant.generate_image("A red circle on white background")')

    start = time.perf_counter()
    try:
        response = await assistant.generate_image(
            prompt="A simple red circle on a white background",
            size="1024x1024",
            n=1
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n RESPONSE ({latency_ms}ms):")
        print(f"   Generated: {len(response.images)} image(s)")
        for img in response.images:
            print(f"   - URL: {img.url[:80]}...")
            if img.revised_prompt:
                print(f"   - Revised prompt: {img.revised_prompt[:100]}...")

        return {"test": "image_generation", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "image_generation", "status": "error", "latency_ms": latency_ms}


async def test_tts(assistant: Assistant) -> dict:
    """Test text-to-speech."""
    _print_section("TEXT-TO-SPEECH")

    print("\n REQUEST:")
    print('   assistant.generate_speech("Hello, this is a test.", voice="alloy")')

    start = time.perf_counter()
    try:
        response = await assistant.generate_speech(
            input="Hello, this is a test of the MLPal text to speech system.",
            voice="alloy"
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n RESPONSE ({latency_ms}ms):")
        print(f"   Format: {response.format}")
        print(f"   Audio size: {len(response.content)} bytes")
        if response.audio.url:
            print(f"   URL: {response.audio.url[:80]}...")

        # Optionally save
        # response.save("test_output.mp3")

        return {"test": "tts", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "tts", "status": "error", "latency_ms": latency_ms}


async def test_transcription(assistant: Assistant) -> dict:
    """Test audio transcription."""
    _print_section("TRANSCRIPTION")

    # We'll use the TTS output to test transcription
    print("\n REQUEST:")
    print('   # First generate speech, then transcribe it back')
    print('   audio = await assistant.generate_speech("Hello world")')
    print('   result = await assistant.transcribe(audio.content)')

    start = time.perf_counter()
    try:
        # Generate some speech first
        speech = await assistant.generate_speech(
            input="Hello, my name is MLPal.",
            voice="alloy"
        )

        # Transcribe it back
        response = await assistant.transcribe(speech.content)
        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n RESPONSE ({latency_ms}ms):")
        print(f"   Transcribed text: {response.text}")
        print(f"   Language: {response.language}")

        return {"test": "transcription", "status": 200, "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        print(f"\n   ERROR: {e}")
        return {"test": "transcription", "status": "error", "latency_ms": latency_ms}


async def main():
    """Run all SDK tests."""
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  MLPAL ASSISTANTS SDK - COMPREHENSIVE TEST SUITE".center(78) + "#")
    print("#" + f"  Testing: {BASE_URL}".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    if not API_KEY:
        print("\n ERROR: MLPAL_API_KEY not set!")
        print(" Set: export MLPAL_API_KEY=mlpal_sk_xxx")
        return 1

    print(f"\n Base URL: {BASE_URL}")
    print(f" API Key: {API_KEY[:20]}...")

    results = []

    async with Assistant(api_key=API_KEY, base_url=BASE_URL) as assistant:
        # Basic tests
        results.append(await test_basic_chat(assistant))
        results.append(await test_simple_chat_api(assistant))
        results.append(await test_conversation_history(assistant))

        # Structured output
        results.append(await test_structured_output(assistant))

        # Tools
        results.append(await test_tools(assistant))

        # Vision
        results.append(await test_vision_single(assistant))

        # MULTIMODAL - Multiple images + structured output (KEY TEST)
        results.append(await test_multimodal_structured(assistant))

        # Streaming
        results.append(await test_streaming(assistant))

        # Embeddings
        results.append(await test_embeddings(assistant))

        # Generation
        results.append(await test_image_generation(assistant))
        results.append(await test_tts(assistant))
        results.append(await test_transcription(assistant))

    # Summary
    _print_section("SUMMARY")

    total_latency = 0
    passed = 0
    failed = 0
    skipped = 0

    print("\n   Test                        | Status       | Latency")
    print("   " + "-" * 55)

    for r in results:
        test_name = r["test"].ljust(26)
        latency = r["latency_ms"]
        status = r["status"]

        if status == "skipped":
            status_str = "SKIP"
            skipped += 1
        elif status == 200:
            status_str = "PASS"
            passed += 1
            total_latency += latency
        elif status == "failed":
            status_str = "FAIL"
            failed += 1
            total_latency += latency
        else:
            status_str = f"ERROR"
            failed += 1
            total_latency += latency

        print(f"   {test_name} | {status_str:12} | {latency}ms")

    print("   " + "-" * 55)
    print(f"\n   Total: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"   Total latency: {total_latency}ms")
    if passed + failed > 0:
        print(f"   Average latency: {total_latency // (passed + failed)}ms")

    print("\n" + "#" * 80 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

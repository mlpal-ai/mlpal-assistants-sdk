#!/usr/bin/env python3
"""Test tool execution functionality."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from mlpal_assistants import Assistants


def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is 22°C and sunny."


async def test_simple_tool():
    """Test simple tool execution."""
    print("=" * 60)
    print("Testing simple tool execution...")
    print("=" * 60)

    assistant = Assistants()

    response = await assistant.chat(
        messages=[{"role": "user", "content": "What time is it right now?"}],
        tools={"get_current_time": get_current_time},
    )

    print(f"Content: {response.content}")
    print(f"Turns: {len(response.turns) if response.turns else 0}")

    if response.turns:
        for i, turn in enumerate(response.turns):
            print(f"  Turn {i+1}: tool_results = {turn.tool_results}")

    print()
    await assistant.close()
    return True


async def test_tool_with_params():
    """Test tool with parameters."""
    print("=" * 60)
    print("Testing tool with parameters...")
    print("=" * 60)

    assistant = Assistants()

    response = await assistant.chat(
        messages=[{"role": "user", "content": "What is 42 + 17?"}],
        tools={"add_numbers": add_numbers},
    )

    print(f"Content: {response.content}")

    # The answer should mention 59
    assert "59" in response.content, f"Expected 59 in response, got: {response.content}"
    print("Verified: Answer contains 59")
    print()

    await assistant.close()
    return True


async def test_multiple_tools():
    """Test with multiple tools available."""
    print("=" * 60)
    print("Testing multiple tools...")
    print("=" * 60)

    assistant = Assistants()

    response = await assistant.chat(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools={
            "get_current_time": get_current_time,
            "add_numbers": add_numbers,
            "get_weather": get_weather,
        },
    )

    print(f"Content: {response.content}")
    print(f"Turns: {len(response.turns) if response.turns else 0}")
    print()

    await assistant.close()
    return True


async def main():
    """Run all tests."""
    try:
        await test_simple_tool()
        await test_tool_with_params()
        await test_multiple_tools()
        print("All tool tests passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))

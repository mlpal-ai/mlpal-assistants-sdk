#!/usr/bin/env python3
"""Test basic chat functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from mlpal_assistants import Assistants


async def test_simple_chat():
    """Test simple text chat."""
    print("=" * 60)
    print("Testing simple chat...")
    print("=" * 60)

    assistant = Assistants()

    response = await assistant.chat(
        messages=[{"role": "user", "content": "Say hello in exactly 5 words."}]
    )

    print(f"Content: {response.content}")
    print(f"Model: {response.routing.resolved_model}")
    print(f"Provider: {response.routing.provider}")
    print(f"Tokens: {response.cost.tokens.total_tokens}")
    print(f"Compute Units: {response.cost.compute_units}")
    print()

    await assistant.close()
    return True


async def test_system_message():
    """Test chat with system message."""
    print("=" * 60)
    print("Testing system message...")
    print("=" * 60)

    assistant = Assistants()

    response = await assistant.chat(
        messages=[
            {"role": "system", "content": "You are a pirate. Speak like one."},
            {"role": "user", "content": "Hello!"}
        ]
    )

    print(f"Content: {response.content}")
    print()

    await assistant.close()
    return True


async def test_conversation():
    """Test multi-turn conversation."""
    print("=" * 60)
    print("Testing multi-turn conversation...")
    print("=" * 60)

    assistant = Assistants()

    messages = [
        {"role": "user", "content": "My name is Alice."}
    ]

    response = await assistant.chat(messages=messages)
    print(f"Turn 1: {response.content}")

    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": "What is my name?"})

    response = await assistant.chat(messages=messages)
    print(f"Turn 2: {response.content}")
    print()

    await assistant.close()
    return True


async def main():
    """Run all tests."""
    try:
        await test_simple_chat()
        await test_system_message()
        await test_conversation()
        print("All chat tests passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))

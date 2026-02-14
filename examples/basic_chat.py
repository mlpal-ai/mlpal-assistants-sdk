"""Basic chat example."""

import asyncio

from mlpal_assistants import Assistant


async def main():
    async with Assistant() as assistant:
        # Simple string shorthand - the easiest way!
        response = await assistant.chat("What is the capital of France?")
        print(f"Response: {response.content}")
        print(f"Model: {response.routing.resolved_model}")
        print(f"Tokens: {response.cost.tokens.total_tokens}")

        # Multi-turn conversation using response.message
        messages = [{"role": "user", "content": "My name is Alice."}]
        response = await assistant.chat(messages=messages)
        print(f"\nAssistant: {response.content}")

        # Add response to history and continue
        messages.append(response.message)
        messages.append({"role": "user", "content": "What's my name?"})
        response = await assistant.chat(messages=messages)
        print(f"Assistant: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())

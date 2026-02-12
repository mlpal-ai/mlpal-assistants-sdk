"""Basic chat example."""

import asyncio

from mlpal_assistants import Assistants


async def main():
    async with Assistants() as assistant:
        # Simple chat
        response = await assistant.chat(
            messages=[{"role": "user", "content": "What is the capital of France?"}]
        )
        print(f"Response: {response.content}")
        print(f"Model: {response.routing.resolved_model}")
        print(f"Tokens: {response.cost.tokens.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())

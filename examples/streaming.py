"""Streaming chat example."""

import asyncio

from mlpal_assistants import Assistant


async def main():
    async with Assistant() as assistant:
        print("Streaming response: ", end="", flush=True)

        async with assistant.stream_chat(
            messages=[{"role": "user", "content": "Tell me a short story about a robot."}]
        ) as stream:
            async for delta in stream:
                if delta.content:
                    print(delta.content, end="", flush=True)

        print("\n")
        print(f"Finish reason: {stream.finish_reason}")
        print(f"Total content length: {len(stream.collected_content)}")


if __name__ == "__main__":
    asyncio.run(main())

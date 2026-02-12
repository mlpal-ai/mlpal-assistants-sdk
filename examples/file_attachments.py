"""File attachment example."""

import asyncio

from mlpal_assistants import Assistants, File


async def main():
    async with Assistants() as assistant:
        # From local file path
        # image = File.from_path("photo.jpg")

        # From URL (not downloaded, passed to API)
        image = File.from_url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
        )

        # From bytes
        # image = File.from_bytes(image_bytes, mime_type="image/png")

        response = await assistant.chat(
            messages=[
                {
                    "role": "user",
                    "content": "What do you see in this image?",
                    "files": [image],
                }
            ]
        )

        print(f"Response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())

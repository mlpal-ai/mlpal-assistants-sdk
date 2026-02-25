#!/usr/bin/env python3
"""
Test image generation with reference images across providers.

This script tests the image-to-image generation capability by:
1. Loading two reference images
2. Sending them to the API with a prompt
3. Saving the generated image to an output folder
"""

import asyncio
import base64
import mimetypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from mlpal_assistants import Assistants


# Configuration
IMAGE_DIR = "/Users/saivivekpeddi/Downloads/Coding/mlpal/code/mlpal/usage_examples/mm/image_gen"
OUTPUT_DIR = "/Users/saivivekpeddi/Downloads/Coding/mlpal/code/mlpal-assistants-sdk/scripts/output"

REFERENCE_IMAGES = [
    os.path.join(IMAGE_DIR, "6F67EF2B-5B1D-4050-985C-64BE98EDA658_1_105_c.jpeg"),
    os.path.join(IMAGE_DIR, "r2.png"),
]

PROMPT = """Can you imagine me and my girlfriend in a whitesand beach in croatia.
It should be like a professional shoot with proper swimwear."""


def file_to_base64(file_path: str) -> tuple[str, str]:
    """Read a file and return base64 encoded data with MIME type."""
    with open(file_path, "rb") as f:
        data = f.read()

    b64_data = base64.b64encode(data).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    return b64_data, mime_type


async def test_image_generation(model: str):
    """Test image generation with reference images."""
    print(f"\n{'='*60}")
    print(f"Testing with model: {model}")
    print(f"{'='*60}")

    # Prepare reference images
    print("\nPreparing reference images...")
    reference_images = []
    for img_path in REFERENCE_IMAGES:
        if os.path.exists(img_path):
            b64_data, mime_type = file_to_base64(img_path)
            reference_images.append({
                "base64": b64_data,
                "mime_type": mime_type,
            })
            print(f"  + Loaded: {os.path.basename(img_path)} ({mime_type}, {len(b64_data)} chars)")
        else:
            print(f"  ! Not found: {img_path}")

    if not reference_images:
        print("No reference images found!")
        return None

    print(f"\nPrompt: {PROMPT[:80]}...")
    print(f"Reference images: {len(reference_images)}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Call SDK
    print("\nCalling SDK...")
    assistant = Assistants()

    try:
        response = await assistant.generate_image(
            prompt=PROMPT,
            model=model,
            reference_images=reference_images,
            size="1024x1024",
            quality="standard",
        )

        print(f"\nResponse received!")
        print(f"  Response type: {type(response)}")
        print(f"  Response attrs: {dir(response)}")
        print(f"  Images: {len(response.images)}")
        print(f"  Images raw: {response.images}")
        print(f"  Cost: {response.cost}")

        # Check for any other data
        if hasattr(response, 'data'):
            print(f"  Data: {response.data}")
        if hasattr(response, 'content'):
            print(f"  Content: {response.content}")
        if hasattr(response, 'metadata'):
            print(f"  Metadata: {response.metadata}")

        # Save generated images
        saved_files = []
        for i, img in enumerate(response.images):
            # Check what data we have
            print(f"\n  Image {i+1}:")
            print(f"    Has URL: {hasattr(img, 'url') and img.url is not None}")
            print(f"    Has data: {hasattr(img, 'data') and img.data is not None}")
            print(f"    Has b64_json: {hasattr(img, 'b64_json') and img.b64_json is not None}")

            # Try to save the image
            output_path = os.path.join(OUTPUT_DIR, f"{model.replace('/', '_')}_{i+1}.png")

            if hasattr(img, 'data') and img.data:
                with open(output_path, "wb") as f:
                    f.write(img.data)
                print(f"    Saved (data): {output_path}")
                saved_files.append(output_path)
            elif hasattr(img, 'b64_json') and img.b64_json:
                img_data = base64.b64decode(img.b64_json)
                with open(output_path, "wb") as f:
                    f.write(img_data)
                print(f"    Saved (b64): {output_path}")
                saved_files.append(output_path)
            elif hasattr(img, 'url') and img.url:
                print(f"    URL: {img.url}")
                # Download from URL
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(img.url)
                    with open(output_path, "wb") as f:
                        f.write(resp.content)
                print(f"    Saved (url): {output_path}")
                saved_files.append(output_path)
            else:
                print(f"    ! No image data found")

        return saved_files

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        await assistant.close()


async def main():
    print("="*60)
    print("Image Generation Test with Reference Images")
    print("="*60)

    # Test with mlpal first
    model = "mlpal"

    result = await test_image_generation(model)

    if result:
        print(f"\n\nSUCCESS! Generated {len(result)} image(s):")
        for f in result:
            print(f"  - {f}")
    else:
        print("\n\nFAILED to generate images")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())

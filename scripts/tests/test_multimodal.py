#!/usr/bin/env python3
"""
Multimodal Test - Multiple Images + Structured Output

Tests whether the SDK can:
1. Accept multiple images (from different sources: file + downloaded)
2. Analyze them together
3. Return structured output (Pydantic model)

This is a key test for true multimodal capabilities.

Usage:
    cd mlpal-assistants-sdk
    export MLPAL_API_KEY=mlpal_sk_xxx
    uv run python scripts/tests/test_multimodal.py
"""

import asyncio
import sys
from pathlib import Path

import httpx
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_URL, API_KEY

from mlpal_assistants import Assistant, File


# Test images
TEST_DIR = Path(__file__).parent
LOCAL_IMAGE = TEST_DIR / "mm" / "photo" / "hadoop.jpg"

# Test image URLs (using picsum.photos which allows API access)
IMAGE_URLS = [
    "https://picsum.photos/seed/test1/400/300",  # Random image 1
    "https://picsum.photos/seed/test2/400/300",  # Random image 2
    "https://picsum.photos/seed/test3/400/300",  # Random image 3
]


async def download_image(url: str) -> bytes:
    """Download image from URL and return bytes."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


class ImageAnalysis(BaseModel):
    """Analysis of a single image."""
    description: str
    main_colors: list[str]
    contains_text: bool
    complexity_score: int  # 1-10


class MultiImageComparison(BaseModel):
    """Structured comparison of multiple images."""
    image_analyses: list[ImageAnalysis]
    total_images: int
    common_themes: list[str]
    most_complex_image_index: int
    overall_summary: str


async def test_two_images_structured():
    """Test: 2 images (local + downloaded) with structured output."""
    print("\n" + "=" * 70)
    print(" TEST: Two Images + Structured Output")
    print("=" * 70)

    if not LOCAL_IMAGE.exists():
        print(f" ERROR: Local image not found: {LOCAL_IMAGE}")
        return False

    class TwoImageAnalysis(BaseModel):
        image1_description: str
        image2_description: str
        key_difference: str

    # Download second image
    print("\n   Downloading second test image...")
    second_image_bytes = await download_image(IMAGE_URLS[0])
    print(f"   Downloaded: {len(second_image_bytes)} bytes")

    print(f"\n Images:")
    print(f"   1. Local file: {LOCAL_IMAGE.name}")
    print(f"   2. Downloaded: {IMAGE_URLS[0][:50]}...")

    print(f"\n Pydantic Model: TwoImageAnalysis")
    print(f"   - image1_description: str")
    print(f"   - image2_description: str")
    print(f"   - key_difference: str")

    async with Assistant(api_key=API_KEY, base_url=BASE_URL) as assistant:
        response = await assistant.chat(
            messages=[{
                "role": "user",
                "content": "Describe each image briefly and tell me the key difference between them.",
                "files": [
                    File.from_path(LOCAL_IMAGE),
                    File.from_bytes(second_image_bytes, mime_type="image/jpeg"),
                ]
            }],
            response_format=TwoImageAnalysis,
            max_tokens=300,
        )

    print(f"\n Result:")
    if response.data:
        print(f"   Image 1: {response.data.image1_description}")
        print(f"   Image 2: {response.data.image2_description}")
        print(f"   Key Difference: {response.data.key_difference}")
        print(f"\n   STATUS: PASSED")
        return True
    else:
        print(f"   Content: {response.content}")
        print(f"\n   STATUS: FAILED (no structured data)")
        return False


async def test_three_images_downloaded():
    """Test: 3 images (all downloaded) with structured output."""
    print("\n" + "=" * 70)
    print(" TEST: Three Images (downloaded) + Structured Output")
    print("=" * 70)

    class ThreeImageSummary(BaseModel):
        descriptions: list[str]
        most_interesting: int  # index 0-2
        reason: str

    # Download all three images
    print("\n   Downloading test images...")
    images = []
    for i, url in enumerate(IMAGE_URLS):
        img_bytes = await download_image(url)
        images.append(img_bytes)
        print(f"   Image {i+1}: {len(img_bytes)} bytes")

    print(f"\n Pydantic Model: ThreeImageSummary")
    print(f"   - descriptions: list[str]")
    print(f"   - most_interesting: int")
    print(f"   - reason: str")

    async with Assistant(api_key=API_KEY, base_url=BASE_URL) as assistant:
        response = await assistant.chat(
            messages=[{
                "role": "user",
                "content": "Describe each of these 3 images briefly. Which is most interesting and why?",
                "files": [
                    File.from_bytes(img, mime_type="image/jpeg")
                    for img in images
                ]
            }],
            response_format=ThreeImageSummary,
            max_tokens=400,
        )

    print(f"\n Result:")
    if response.data:
        for i, desc in enumerate(response.data.descriptions):
            print(f"   Image {i+1}: {desc[:80]}...")
        print(f"   Most interesting: Image {response.data.most_interesting + 1}")
        print(f"   Reason: {response.data.reason}")
        print(f"\n   STATUS: PASSED")
        return True
    else:
        print(f"   Content: {response.content}")
        print(f"\n   STATUS: FAILED (no structured data)")
        return False


async def test_image_with_complex_schema():
    """Test: Single image with complex nested schema."""
    print("\n" + "=" * 70)
    print(" TEST: Single Image + Complex Nested Schema")
    print("=" * 70)

    class ColorInfo(BaseModel):
        name: str
        hex_approximate: str
        percentage: int

    class ObjectDetected(BaseModel):
        name: str
        confidence: str  # "high", "medium", "low"

    class DetailedImageAnalysis(BaseModel):
        description: str
        dominant_colors: list[ColorInfo]
        objects_detected: list[ObjectDetected]
        mood: str
        suggested_tags: list[str]

    # Download an image
    print("\n   Downloading test image...")
    image_bytes = await download_image(IMAGE_URLS[1])
    print(f"   Downloaded: {len(image_bytes)} bytes")

    print(f"\n Complex Pydantic Model with nested types:")
    print(f"   DetailedImageAnalysis:")
    print(f"     - description: str")
    print(f"     - dominant_colors: list[ColorInfo]")
    print(f"     - objects_detected: list[ObjectDetected]")
    print(f"     - mood: str")
    print(f"     - suggested_tags: list[str]")

    async with Assistant(api_key=API_KEY, base_url=BASE_URL) as assistant:
        response = await assistant.chat(
            messages=[{
                "role": "user",
                "content": "Analyze this image in detail. Identify colors (with approximate hex), objects, mood, and suggest tags.",
                "files": [File.from_bytes(image_bytes, mime_type="image/jpeg")]
            }],
            response_format=DetailedImageAnalysis,
            max_tokens=600,
        )

    print(f"\n Result:")
    if response.data:
        print(f"   Description: {response.data.description[:100]}...")
        print(f"   Colors: {[(c.name, c.hex_approximate) for c in response.data.dominant_colors]}")
        print(f"   Objects: {[(o.name, o.confidence) for o in response.data.objects_detected]}")
        print(f"   Mood: {response.data.mood}")
        print(f"   Tags: {response.data.suggested_tags}")
        print(f"\n   STATUS: PASSED")
        return True
    else:
        print(f"   Content: {response.content}")
        print(f"\n   STATUS: FAILED (no structured data)")
        return False


async def main():
    print("\n" + "#" * 70)
    print("#" + "  MULTIMODAL TEST SUITE".center(68) + "#")
    print("#" + "  Multiple Images + Structured Output".center(68) + "#")
    print("#" * 70)

    if not API_KEY:
        print("\n ERROR: MLPAL_API_KEY not set!")
        return 1

    print(f"\n Base URL: {BASE_URL}")
    print(f" API Key: {API_KEY[:20]}...")

    results = []

    try:
        results.append(("2 images (file+downloaded)", await test_two_images_structured()))
    except Exception as e:
        print(f"\n   ERROR: {e}")
        results.append(("2 images (file+downloaded)", False))

    try:
        results.append(("3 images (downloaded)", await test_three_images_downloaded()))
    except Exception as e:
        print(f"\n   ERROR: {e}")
        results.append(("3 images (downloaded)", False))

    try:
        results.append(("Complex schema", await test_image_with_complex_schema()))
    except Exception as e:
        print(f"\n   ERROR: {e}")
        results.append(("Complex schema", False))

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    failed = len(results) - passed

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"   {name.ljust(25)} : {status}")

    print(f"\n   Total: {passed} passed, {failed} failed")
    print("#" * 70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

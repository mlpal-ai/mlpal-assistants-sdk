#!/usr/bin/env python3
"""Test embeddings functionality."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from mlpal_assistants import Assistants


async def test_single_embedding():
    """Test single text embedding."""
    print("=" * 60)
    print("Testing single embedding...")
    print("=" * 60)

    assistant = Assistants()

    response = await assistant.embed("Hello world")

    print(f"Model: {response.model}")
    print(f"Dimensions: {len(response.data[0].embedding)}")
    print(f"First 5 values: {response.data[0].embedding[:5]}")
    print(f"Compute Units: {response.cost.compute_units}")
    print()

    assert len(response.data) == 1
    assert len(response.data[0].embedding) > 0

    await assistant.close()
    return True


async def test_batch_embeddings():
    """Test batch embeddings."""
    print("=" * 60)
    print("Testing batch embeddings...")
    print("=" * 60)

    assistant = Assistants()

    texts = ["Hello", "World", "How are you?"]
    response = await assistant.embed(texts)

    print(f"Number of embeddings: {len(response.data)}")
    print(f"Dimensions: {len(response.data[0].embedding)}")

    for i, emb in enumerate(response.data):
        print(f"  [{i}] First 3: {emb.embedding[:3]}")

    print()

    assert len(response.data) == 3

    await assistant.close()
    return True


async def test_similarity():
    """Test embedding similarity."""
    print("=" * 60)
    print("Testing embedding similarity...")
    print("=" * 60)

    assistant = Assistants()

    texts = [
        "The cat sat on the mat",
        "A feline rested on the rug",
        "The stock market crashed today",
    ]
    response = await assistant.embed(texts)

    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b)

    emb1, emb2, emb3 = [e.embedding for e in response.data]

    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)

    print(f"Similarity (cat/feline): {sim_12:.4f}")
    print(f"Similarity (cat/stock): {sim_13:.4f}")
    print(f"Similar texts should have higher similarity: {sim_12 > sim_13}")
    print()

    assert sim_12 > sim_13, "Similar texts should have higher similarity"

    await assistant.close()
    return True


async def main():
    """Run all tests."""
    try:
        await test_single_embedding()
        await test_batch_embeddings()
        await test_similarity()
        print("All embedding tests passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))

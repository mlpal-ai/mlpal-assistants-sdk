#!/usr/bin/env python3
"""
Test structured output functionality across all providers.

Tests complex Pydantic schemas with:
- Enums (creates $ref in JSON schema)
- Nested models
- Optional fields

Uses representative models from each provider (economy/standard tier).
"""

import asyncio
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from mlpal_assistants import Assistants


# ============================================================================
# Test Schemas - Complex types that exercise $ref handling
# ============================================================================

class Priority(str, Enum):
    """Task priority levels - creates $ref in JSON schema."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(str, Enum):
    """Task status - another enum for testing."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class Task(BaseModel):
    """Task with enum fields - tests $ref sibling handling."""
    title: str = Field(description="Task title")
    description: str = Field(description="Task description")
    priority: Priority = Field(description="Task priority level")
    status: Status = Field(description="Current task status")
    assignee: Optional[str] = Field(description="Person assigned, or null if unassigned")


class UserIntent(str, Enum):
    """User intent detection - matches the multimodal agent use case."""
    ANALYZE = "analyze"
    GENERATE = "generate"
    EDIT = "edit"
    CHAT = "chat"
    EXIT = "exit"


class IntentResult(BaseModel):
    """Intent detection result - similar to multimodal agent schema."""
    intent: UserIntent = Field(description="Detected user intent")
    reasoning: str = Field(description="Why this intent was detected")
    refined_prompt: str = Field(description="Enhanced version of user's message")


# ============================================================================
# Models to test - one representative from each provider
# ============================================================================

MODELS_TO_TEST = [
    # MLPal auto-routing
    ("mlpal", "MLPal Auto-routing"),
    # OpenAI
    ("gpt-4o-mini", "OpenAI GPT-4o Mini"),
    ("gpt-4o", "OpenAI GPT-4o"),
    # Anthropic
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ("claude-sonnet-4-20250514", "Claude Sonnet 4"),
    # Google
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    # Bedrock/Llama
    ("us.meta.llama3-2-11b-instruct", "Llama 3.2 11B"),
]


# ============================================================================
# Test Functions
# ============================================================================

async def test_enum_schema(model: str, display_name: str) -> tuple[bool, str]:
    """Test schema with enum fields - the main issue we're fixing."""
    assistant = Assistants()
    try:
        response = await assistant.chat(
            messages=[{
                "role": "user",
                "content": "Create a high priority task called 'Fix bug' that is in progress, assigned to Alice."
            }],
            model=model,
            response_format=Task,
        )

        if response.data:
            task = response.data
            if not isinstance(task.priority, Priority):
                return False, f"Expected Priority enum, got {type(task.priority)}"
            if not isinstance(task.status, Status):
                return False, f"Expected Status enum, got {type(task.status)}"
            return True, f"priority={task.priority.value}, status={task.status.value}"
        else:
            return False, "No structured data returned"

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 80:
            error_msg = error_msg[:80] + "..."
        return False, error_msg
    finally:
        await assistant.close()


async def test_intent_detection(model: str, display_name: str) -> tuple[bool, str]:
    """Test intent detection schema - matches multimodal agent use case."""
    assistant = Assistants()
    try:
        response = await assistant.chat(
            messages=[{
                "role": "user",
                "content": "Detect intent: 'Create an image of sunset'. Return intent, reasoning, refined prompt."
            }],
            model=model,
            response_format=IntentResult,
        )

        if response.data:
            result = response.data
            if not isinstance(result.intent, UserIntent):
                return False, f"Expected UserIntent enum, got {type(result.intent)}"
            return True, f"intent={result.intent.value}"
        else:
            return False, "No structured data returned"

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 80:
            error_msg = error_msg[:80] + "..."
        return False, error_msg
    finally:
        await assistant.close()


async def run_model_tests(model: str, display_name: str) -> dict:
    """Run all tests for a single model."""
    print(f"\n  {display_name} ({model})")

    results = {}

    # Test 1: Enum schema
    passed, msg = await test_enum_schema(model, display_name)
    status = "✓" if passed else "✗"
    print(f"    {status} enum_schema: {msg}")
    results['enum_schema'] = {'passed': passed, 'message': msg}

    # Test 2: Intent detection
    passed, msg = await test_intent_detection(model, display_name)
    status = "✓" if passed else "✗"
    print(f"    {status} intent_detection: {msg}")
    results['intent_detection'] = {'passed': passed, 'message': msg}

    return results


async def main():
    """Run tests across all providers."""
    print("=" * 70)
    print("Structured Output Cross-Provider Test Suite")
    print("=" * 70)
    print(f"\nTesting {len(MODELS_TO_TEST)} models...")

    all_results = {}

    for model, display_name in MODELS_TO_TEST:
        results = await run_model_tests(model, display_name)
        all_results[model] = {
            'display_name': display_name,
            'results': results,
        }

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_tests = 0

    for model, data in all_results.items():
        passed = sum(1 for r in data['results'].values() if r['passed'])
        total = len(data['results'])
        total_passed += passed
        total_tests += total
        status = "✓" if passed == total else "✗"
        print(f"  {status} {data['display_name']}: {passed}/{total}")

    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

    # List failures
    failures = []
    for model, data in all_results.items():
        for test_name, result in data['results'].items():
            if not result['passed']:
                failures.append((data['display_name'], test_name, result['message']))

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for display_name, test, msg in failures:
            print(f"  - {display_name} / {test}")
            print(f"    {msg}")

    return 0 if total_passed == total_tests else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))

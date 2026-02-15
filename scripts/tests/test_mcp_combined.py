#!/usr/bin/env python3
"""
Test MCP Tools + Regular Tools Combined

Tests whether the SDK can:
1. Connect to an MCP server (Perplexity)
2. Register regular tools alongside MCP tools
3. Use both types of tools in the same chat

Usage:
    cd mlpal-assistants-sdk
    export MLPAL_API_KEY=mlpal_sk_xxx
    export PERPLEXITY_API_KEY=pplx-xxx
    uv run python scripts/tests/test_mcp_combined.py
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_URL, API_KEY

from mlpal_assistants import Assistant, ToolRegistry
from mlpal_assistants.tools import MCPBridge


# Check for Perplexity API key
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")


class ResearchSummary(BaseModel):
    """Structured output for research results."""
    topic: str
    key_findings: list[str]
    current_date_used: str
    sources_mentioned: int
    local_calculation_result: str | None = None


async def test_mcp_with_regular_tools():
    """Test combining MCP tools with regular tools."""
    print("\n" + "=" * 70)
    print(" TEST: MCP Tools + Regular Tools Combined")
    print("=" * 70)

    if not PERPLEXITY_API_KEY:
        print("\n ERROR: PERPLEXITY_API_KEY not set!")
        print(" Set: export PERPLEXITY_API_KEY=pplx-xxx")
        return False

    # Create regular tools
    tools = ToolRegistry()

    @tools.tool
    def get_current_date() -> str:
        """Get the current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @tools.tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: A math expression like "2 + 2" or "15 * 7"
        """
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    @tools.tool
    def format_summary(title: str, points: list[str]) -> str:
        """Format a summary with bullet points.

        Args:
            title: The title for the summary
            points: List of bullet points
        """
        formatted = f"## {title}\n\n"
        for point in points:
            formatted += f"• {point}\n"
        return formatted

    print(f"\n Regular tools registered: {tools.names}")

    # Connect to Perplexity MCP server
    print("\n Connecting to Perplexity MCP server...")

    try:
        async with MCPBridge() as bridge:
            await bridge.connect_stdio(
                name="perplexity",
                command="npx",
                args=["-y", "perplexity-mcp"],
                env={"PERPLEXITY_API_KEY": PERPLEXITY_API_KEY},
            )

            print(f" Connected servers: {bridge.connected_servers}")
            print(f" MCP tools available: {bridge.available_tools}")

            # Get MCP tools and merge with regular tools
            mcp_tools = bridge.get_tool_registry()
            tools.merge(mcp_tools)

            print(f"\n Combined tools: {tools.names}")

            # Test 1: Use both MCP and regular tools
            print("\n" + "-" * 70)
            print(" Test 1: Combined tool usage")
            print("-" * 70)

            async with Assistant(api_key=API_KEY, base_url=BASE_URL) as assistant:
                response = await assistant.chat(
                    messages=[{
                        "role": "user",
                        "content": """I need you to:
1. First, get the current date using the get_current_date tool
2. Then use perplexity to research "latest developments in AI agents in 2024"
3. Finally, calculate how many days are in 52 weeks using the calculate tool

Summarize what you found."""
                    }],
                    tools=tools,
                    max_turns=10,
                )

                print(f"\n Response:")
                print(f"   {response.content[:500]}..." if len(response.content) > 500 else f"   {response.content}")

                if response.turns:
                    print(f"\n   Turns used: {len(response.turns)}")
                    for i, turn in enumerate(response.turns):
                        if turn.tool_results:
                            for tr in turn.tool_results:
                                print(f"   Turn {i+1}: Called '{tr.get('name', 'unknown')}'")

                print("\n   TEST 1: PASSED")

            # Test 2: Structured output with combined tools
            print("\n" + "-" * 70)
            print(" Test 2: Structured output with combined tools")
            print("-" * 70)

            async with Assistant(api_key=API_KEY, base_url=BASE_URL) as assistant:
                response = await assistant.chat(
                    messages=[{
                        "role": "user",
                        "content": """Do these three things:
1. Get today's date using the get_current_date tool
2. Use perplexity_search to find "What is Claude AI by Anthropic" (keep it brief)
3. Calculate 52 * 7 using the calculate tool

Then return structured JSON output with these exact fields."""
                    }],
                    tools=tools,
                    response_format=ResearchSummary,
                    max_turns=8,
                )

                print(f"\n Response:")
                if response.data:
                    print(f"   Topic: {response.data.topic}")
                    print(f"   Date used: {response.data.current_date_used}")
                    print(f"   Calculation: {response.data.local_calculation_result}")
                    print(f"   Key findings ({len(response.data.key_findings)}):")
                    for finding in response.data.key_findings[:3]:
                        print(f"     • {finding[:80]}...")
                    print(f"\n   TEST 2: PASSED")
                else:
                    print(f"   Content: {response.content}")
                    print(f"\n   TEST 2: FAILED (no structured data)")
                    return False

        return True

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mcp_tool_registry_merge():
    """Test the ToolRegistry.merge() functionality."""
    print("\n" + "=" * 70)
    print(" TEST: ToolRegistry Merge Methods")
    print("=" * 70)

    # Create two separate registries
    registry1 = ToolRegistry()
    registry2 = ToolRegistry()

    @registry1.tool
    def tool_a() -> str:
        """Tool A from registry 1."""
        return "A"

    @registry1.tool
    def tool_b() -> str:
        """Tool B from registry 1."""
        return "B"

    @registry2.tool
    def tool_c() -> str:
        """Tool C from registry 2."""
        return "C"

    @registry2.tool
    def tool_b() -> str:
        """Tool B from registry 2 (different implementation)."""
        return "B2"

    print(f"\n Registry 1 tools: {registry1.names}")
    print(f" Registry 2 tools: {registry2.names}")

    # Test merge with overwrite
    merged = ToolRegistry()
    merged.merge(registry1)
    merged.merge(registry2, overwrite=True)
    print(f"\n After merge (overwrite=True): {merged.names}")

    # Test merge without overwrite
    merged2 = ToolRegistry()
    merged2.merge(registry1)
    merged2.merge(registry2, overwrite=False)
    print(f" After merge (overwrite=False): {merged2.names}")

    # Test merge with prefix
    merged3 = ToolRegistry()
    merged3.merge(registry1)
    merged3.merge(registry2, prefix="r2_")
    print(f" After merge (prefix='r2_'): {merged3.names}")

    # Test ToolRegistry.merged() class method
    combined = ToolRegistry.merged(registry1, registry2)
    print(f" ToolRegistry.merged(): {combined.names}")

    print("\n TEST: PASSED")
    return True


async def main():
    print("\n" + "#" * 70)
    print("#" + "  MCP + REGULAR TOOLS COMBINED TEST".center(68) + "#")
    print("#" * 70)

    if not API_KEY:
        print("\n ERROR: MLPAL_API_KEY not set!")
        return 1

    print(f"\n Base URL: {BASE_URL}")
    print(f" API Key: {API_KEY[:20]}...")
    print(f" Perplexity Key: {PERPLEXITY_API_KEY[:15]}..." if PERPLEXITY_API_KEY else " Perplexity Key: NOT SET")

    results = []

    # Test merge functionality first (doesn't require MCP)
    try:
        results.append(("ToolRegistry.merge()", await test_mcp_tool_registry_merge()))
    except Exception as e:
        print(f"\n ERROR: {e}")
        results.append(("ToolRegistry.merge()", False))

    # Test combined MCP + regular tools
    if PERPLEXITY_API_KEY:
        try:
            results.append(("MCP + Regular Tools", await test_mcp_with_regular_tools()))
        except Exception as e:
            print(f"\n ERROR: {e}")
            results.append(("MCP + Regular Tools", False))
    else:
        print("\n SKIPPING MCP test (PERPLEXITY_API_KEY not set)")
        results.append(("MCP + Regular Tools", "SKIPPED"))

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    for name, result in results:
        if result == "SKIPPED":
            status = "SKIP"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"   {name.ljust(30)} : {status}")

    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r == "SKIPPED")

    print(f"\n   Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("#" * 70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

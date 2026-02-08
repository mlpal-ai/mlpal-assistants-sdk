#!/usr/bin/env python3
"""Run all integration tests."""

import asyncio
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parent


def run_test(script_name: str) -> bool:
    """Run a test script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'=' * 70}")
    print(f"Running: {script_name}")
    print('=' * 70)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPTS_DIR.parent),
    )

    return result.returncode == 0


def main():
    """Run all test scripts."""
    tests = [
        "test_chat.py",
        "test_structured.py",
        "test_embeddings.py",
        "test_tools.py",
    ]

    results = {}
    for test in tests:
        results[test] = run_test(test)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test}")
        if not passed:
            all_passed = False

    print("=" * 70)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

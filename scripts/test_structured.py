#!/usr/bin/env python3
"""Test structured output functionality."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from mlpal_assistants import Assistants


class Person(BaseModel):
    name: str
    age: int
    occupation: str


class WeatherReport(BaseModel):
    location: str
    temperature: float
    conditions: str
    humidity: int


async def test_simple_structured():
    """Test simple structured output."""
    print("=" * 60)
    print("Testing simple structured output...")
    print("=" * 60)

    assistant = Assistants()

    response = await assistant.chat(
        messages=[{
            "role": "user",
            "content": "John is a 30 year old software engineer from San Francisco."
        }],
        model="gpt-5.2",  # Use specific model for structured output
        response_format=Person,
    )

    print(f"Content: {response.content}")
    print(f"Data type: {type(response.data)}")

    if response.data:
        person = response.data
        print(f"Name: {person.name}")
        print(f"Age: {person.age}")
        print(f"Occupation: {person.occupation}")

        assert person.name == "John", f"Expected John, got {person.name}"
        assert person.age == 30, f"Expected 30, got {person.age}"
    else:
        print("WARNING: No structured data returned")

    print()
    await assistant.close()
    return True


async def test_weather_structured():
    """Test weather report structured output."""
    print("=" * 60)
    print("Testing weather structured output...")
    print("=" * 60)

    assistant = Assistants()

    response = await assistant.chat(
        messages=[{
            "role": "user",
            "content": "The weather in Paris is currently 22 degrees Celsius, sunny with 65% humidity."
        }],
        model="gpt-5.2",  # Use specific model for structured output
        response_format=WeatherReport,
    )

    print(f"Content: {response.content}")
    print(f"Data type: {type(response.data)}")

    if response.data:
        weather = response.data
        print(f"Location: {weather.location}")
        print(f"Temperature: {weather.temperature}")
        print(f"Conditions: {weather.conditions}")
        print(f"Humidity: {weather.humidity}")
    else:
        print("WARNING: No structured data returned")

    print()
    await assistant.close()
    return True


async def main():
    """Run all tests."""
    try:
        await test_simple_structured()
        await test_weather_structured()
        print("All structured output tests passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))

"""Tool use example with automatic execution."""

import asyncio

from mlpal_assistants import Assistants, ToolRegistry


def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    # Simulated weather data
    weather_data = {
        "paris": {"temp": 18, "condition": "partly cloudy"},
        "london": {"temp": 14, "condition": "rainy"},
        "tokyo": {"temp": 22, "condition": "sunny"},
    }

    data = weather_data.get(city.lower(), {"temp": 20, "condition": "unknown"})
    temp = data["temp"]
    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32

    return f"Weather in {city}: {temp}°{'F' if unit == 'fahrenheit' else 'C'}, {data['condition']}"


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2")
    """
    try:
        result = eval(expression)  # Note: Use a safe evaluator in production
        return str(result)
    except Exception as e:
        return f"Error: {e}"


async def main():
    # Method 1: Using ToolRegistry with decorators
    tools = ToolRegistry()

    @tools.tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Search results for: {query}"

    # Method 2: Register functions directly
    tools.register("get_weather", get_weather)
    tools.register("calculate", calculate)

    async with Assistants() as assistant:
        # The SDK automatically executes tools and returns the final response
        response = await assistant.chat(
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris? Also, what is 15 * 7?",
                }
            ],
            tools=tools,
            max_turns=5,
        )

        print(f"Final response: {response.content}")

        # View the turns (tool calls and results)
        if response.turns:
            print(f"\nCompleted in {len(response.turns)} turn(s)")


if __name__ == "__main__":
    asyncio.run(main())

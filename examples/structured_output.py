"""Structured output example using Pydantic models."""

import asyncio

from pydantic import BaseModel

from mlpal_assistants import Assistant


class Person(BaseModel):
    """A person with name and age."""

    name: str
    age: int
    occupation: str | None = None


class ExtractedPeople(BaseModel):
    """List of extracted people."""

    people: list[Person]


async def main():
    async with Assistant() as assistant:
        # Extract structured data from text using simple string API
        response = await assistant.chat(
            "John is a 30 year old software engineer. "
            "His colleague Maria is 28 and works as a designer.",
            response_format=ExtractedPeople,
        )

        # Access typed data
        if response.data:
            for person in response.data.people:
                print(f"Name: {person.name}, Age: {person.age}, Job: {person.occupation}")


if __name__ == "__main__":
    asyncio.run(main())

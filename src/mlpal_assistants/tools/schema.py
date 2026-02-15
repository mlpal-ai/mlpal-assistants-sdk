"""JSON Schema generation from Python functions."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin, get_type_hints

# Type to JSON Schema mapping
TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

# Get UnionType for Python 3.10+
if sys.version_info >= (3, 10):
    from types import UnionType
else:
    UnionType = None


def generate_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Generate JSON Schema for a function's parameters.

    Uses type hints and docstrings to build a complete schema.
    Supports:
    - Basic types (str, int, float, bool)
    - Optional types (Optional[T], T | None)
    - Lists and dicts
    - Default values
    - Docstring descriptions

    Args:
        func: Function to generate schema for

    Returns:
        Schema dict with name, description, and parameters
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    docstring = inspect.getdoc(func) or ""

    # Parse docstring for parameter descriptions
    param_docs = _parse_docstring_params(docstring)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        # Get type from hints
        param_type = hints.get(param_name, str)
        param_schema = _type_to_schema(param_type)

        # Add description from docstring
        if param_name in param_docs:
            param_schema["description"] = param_docs[param_name]

        # Handle default values
        if param.default is not inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = param_schema

    # Extract first line of docstring as description
    description = docstring.split("\n")[0] if docstring else ""

    return {
        "name": func.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _type_to_schema(python_type: type) -> dict[str, Any]:
    """Convert a Python type hint to JSON Schema."""
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle None type directly
    if python_type is type(None):
        return {"type": "null"}

    # Handle Union types (including Optional and Python 3.10+ X | Y syntax)
    is_union = origin is Union
    if UnionType is not None and isinstance(python_type, UnionType):
        is_union = True
        args = get_args(python_type)

    if is_union:
        # Filter out None to get the actual type
        non_none_types = [t for t in args if t is not type(None)]
        if len(non_none_types) == 1:
            return _type_to_schema(non_none_types[0])
        # Multiple types - use anyOf
        return {"anyOf": [_type_to_schema(t) for t in non_none_types]}

    # Handle list[T]
    if origin is list:
        item_type = args[0] if args else str
        return {
            "type": "array",
            "items": _type_to_schema(item_type),
        }

    # Handle dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Handle basic types
    if python_type in TYPE_MAP:
        return {"type": TYPE_MAP[python_type]}

    # Default to string for unknown types
    return {"type": "string"}


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """
    Parse parameter descriptions from docstring.

    Supports Google-style docstrings:
        Args:
            param_name: Description here
            other_param: Another description

    Returns:
        Dict mapping parameter names to descriptions
    """
    params: dict[str, str] = {}

    if not docstring:
        return params

    lines = docstring.split("\n")
    in_args_section = False
    current_param = ""
    current_desc = ""

    for line in lines:
        stripped = line.strip()

        # Detect Args section
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue

        # Detect end of Args section
        if in_args_section and stripped and not stripped.startswith(" ") and ":" in stripped:
            if stripped.lower().startswith(("returns:", "raises:", "yields:", "examples:")):
                # Save current param and exit
                if current_param:
                    params[current_param] = current_desc.strip()
                break

        if in_args_section:
            # Check for new parameter
            if ":" in stripped and not stripped.startswith(" "):
                # Save previous param
                if current_param:
                    params[current_param] = current_desc.strip()

                # Parse new param
                colon_idx = stripped.index(":")
                current_param = stripped[:colon_idx].strip()
                current_desc = stripped[colon_idx + 1:].strip()
            elif current_param:
                # Continuation of current param description
                current_desc += " " + stripped

    # Save last param
    if current_param:
        params[current_param] = current_desc.strip()

    return params

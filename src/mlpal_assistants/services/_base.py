"""Base service with shared functionality."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlpal_assistants.transport import HTTPTransport


class BaseService(ABC):
    """Base class for all services."""

    def __init__(self, transport: HTTPTransport) -> None:
        self._transport = transport

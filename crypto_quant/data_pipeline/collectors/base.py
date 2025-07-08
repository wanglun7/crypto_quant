"""Base abstractions for data collectors."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class AsyncProducer(ABC):
    """Abstract base class for async data producers."""

    @abstractmethod
    async def start(self) -> None:
        """Start the producer."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the producer gracefully."""
        pass

    @abstractmethod
    def produce(self) -> AsyncGenerator[dict[str, Any], None]:
        """Yield data as dict."""
        pass

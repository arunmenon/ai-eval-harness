"""
Base agent implementation with retry logic and rate limiting.

This module provides a base class for agents with common functionality
like retries, rate limiting, and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import abstractmethod
from typing import Any, AsyncIterator

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..core.agent import (
    Agent,
    AgentAPIError,
    AgentConfig,
    AgentRateLimitError,
    AgentTimeoutError,
)
from ..core.types import AgentResponse, Message, ToolDefinition

logger = logging.getLogger(__name__)


class BaseAgent(Agent):
    """
    Base agent with retry logic and rate limiting.

    This class extends the abstract Agent with:
    - Automatic retries with exponential backoff
    - Rate limiting to avoid API throttling
    - Request timing and logging
    - Common error handling

    Subclasses should implement _generate_impl() instead of generate().
    """

    def __init__(
        self,
        config: AgentConfig,
        min_request_interval: float = 0.0,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            config: Agent configuration
            min_request_interval: Minimum seconds between requests (rate limiting)
        """
        super().__init__(config)
        self.min_request_interval = min_request_interval
        self._last_request_time: float = 0.0
        self._request_count: int = 0
        self._total_tokens: int = 0

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Generate a response with automatic retries.

        Args:
            messages: Conversation history
            tools: Optional list of tools
            **kwargs: Additional parameters

        Returns:
            AgentResponse

        Raises:
            AgentAPIError: If all retries fail
        """
        # Apply rate limiting
        await self._apply_rate_limit()

        start_time = time.time()

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((AgentRateLimitError, AgentTimeoutError)),
                stop=stop_after_attempt(self.config.max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=60),
                reraise=True,
            ):
                with attempt:
                    response = await self._generate_impl(messages, tools, **kwargs)
                    self._record_request(response, time.time() - start_time)
                    return response

        except RetryError as e:
            raise AgentAPIError(
                f"All {self.config.max_retries} retries failed: {e.last_attempt.exception()}"
            ) from e

        # Should not reach here, but satisfy type checker
        raise AgentAPIError("Unexpected error in generate")

    @abstractmethod
    async def _generate_impl(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Implementation of generate to be overridden by subclasses.

        Args:
            messages: Conversation history
            tools: Optional list of tools
            **kwargs: Additional parameters

        Returns:
            AgentResponse
        """
        pass

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentResponse]:
        """
        Streaming generation with rate limiting.

        Default implementation calls _generate_stream_impl if available,
        otherwise falls back to non-streaming generate.

        Args:
            messages: Conversation history
            tools: Optional list of tools
            **kwargs: Additional parameters

        Yields:
            AgentResponse chunks
        """
        await self._apply_rate_limit()

        try:
            async for chunk in self._generate_stream_impl(messages, tools, **kwargs):
                yield chunk
        except NotImplementedError:
            # Fall back to non-streaming
            response = await self.generate(messages, tools, **kwargs)
            yield response

    async def _generate_stream_impl(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentResponse]:
        """
        Streaming implementation to be overridden by subclasses.

        Raises:
            NotImplementedError: If streaming is not supported
        """
        raise NotImplementedError("Streaming not implemented")

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self.min_request_interval > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _record_request(self, response: AgentResponse, duration: float) -> None:
        """Record request statistics."""
        self._request_count += 1
        if response.usage:
            self._total_tokens += response.usage.get("total_tokens", 0)

        logger.debug(
            f"Request {self._request_count}: {duration:.2f}s, "
            f"tokens: {response.usage.get('total_tokens', 'N/A')}"
        )

    def get_stats(self) -> dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary with request count and token usage
        """
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
        }

    def reset(self) -> None:
        """Reset agent state including statistics."""
        super().reset()
        # Optionally reset stats - keeping them for now for aggregate tracking


# Registry for agent implementations
_AGENT_REGISTRY: dict[str, type[Agent]] = {}


def register_agent(name: str) -> Any:
    """
    Decorator to register an agent implementation.

    Args:
        name: Name to register the agent under

    Returns:
        Decorator function
    """

    def decorator(cls: type[Agent]) -> type[Agent]:
        _AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_agent(config: AgentConfig | dict[str, Any]) -> Agent:
    """
    Get an agent instance by provider name.

    Args:
        config: AgentConfig or dict with 'provider' key

    Returns:
        Agent instance

    Raises:
        ValueError: If provider is not registered
    """
    if isinstance(config, dict):
        config = AgentConfig(**config)

    provider = config.provider.lower()

    if provider not in _AGENT_REGISTRY:
        available = ", ".join(_AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent provider: {provider}. Available: {available}")

    agent_cls = _AGENT_REGISTRY[provider]
    return agent_cls(config)


def list_agents() -> list[str]:
    """
    List all registered agent providers.

    Returns:
        List of provider names
    """
    return list(_AGENT_REGISTRY.keys())

"""
Abstract Agent interface for the AI evaluation harness.

This module defines the abstract base class that all agents must implement
to be evaluated by the harness. Agents wrap LLM APIs and handle tool calling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .types import AgentResponse, Message, ToolDefinition


@dataclass
class AgentConfig:
    """
    Configuration for an agent.

    Attributes:
        model_name: Name of the model to use (e.g., "gpt-4o", "claude-3-5-sonnet")
        provider: Provider name ("openai", "anthropic", etc.)
        api_key: API key for authentication (can be None if using env var)
        base_url: Optional custom API base URL
        parameters: Model parameters (temperature, max_tokens, etc.)
        timeout_seconds: Request timeout
        max_retries: Maximum number of retries for failed requests
    """

    model_name: str
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 120.0
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding sensitive data)."""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "base_url": self.base_url,
            "parameters": self.parameters,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


class Agent(ABC):
    """
    Abstract base class for AI agents.

    Users implement this interface to create custom agents that can be
    evaluated across all supported benchmarks. The agent is responsible
    for:
    - Converting messages to the provider's format
    - Making API calls to generate responses
    - Parsing tool calls from responses
    - Handling errors and retries

    Example:
        ```python
        class MyAgent(Agent):
            async def generate(self, messages, tools=None, **kwargs):
                # Call your LLM API
                response = await self.client.chat.completions.create(...)
                return AgentResponse(content=response.choices[0].message.content)
        ```
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the agent with configuration.

        Args:
            config: Agent configuration including model name and API settings
        """
        self.config = config
        self._conversation_history: list[Message] = []

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Generate a response given conversation history and available tools.

        This is the main method that subclasses must implement. It should:
        1. Convert messages to the provider's format
        2. Make the API call
        3. Parse the response into an AgentResponse

        Args:
            messages: Conversation history as a list of Message objects
            tools: Optional list of tools available for this turn
            **kwargs: Additional model-specific parameters

        Returns:
            AgentResponse containing content and/or tool calls

        Raises:
            Exception: If the API call fails after all retries
        """
        pass

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentResponse]:
        """
        Streaming variant of generate.

        Default implementation calls generate() and yields the full response.
        Subclasses can override this to provide true streaming.

        Args:
            messages: Conversation history
            tools: Optional list of tools
            **kwargs: Additional parameters

        Yields:
            AgentResponse chunks (or full response if not streaming)
        """
        response = await self.generate(messages, tools, **kwargs)
        yield response

    def reset(self) -> None:
        """
        Reset agent state between tasks.

        Called at the start of each new task to clear any accumulated state.
        Subclasses can override to add custom reset logic.
        """
        self._conversation_history = []

    def get_model_info(self) -> dict[str, Any]:
        """
        Return model metadata for logging.

        Returns:
            Dictionary containing model name, provider, and parameters
        """
        return {
            "model_name": self.config.model_name,
            "provider": self.config.provider,
            "parameters": self.config.parameters,
        }

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model_name

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self.config.provider


class AgentError(Exception):
    """Base exception for agent errors."""

    pass


class AgentAPIError(AgentError):
    """Raised when an API call fails."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AgentRateLimitError(AgentAPIError):
    """Raised when rate limited by the API."""

    def __init__(
        self,
        message: str = "Rate limited",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class AgentTimeoutError(AgentError):
    """Raised when a request times out."""

    pass


class AgentToolError(AgentError):
    """Raised when there's an error with tool execution."""

    def __init__(self, message: str, tool_name: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name

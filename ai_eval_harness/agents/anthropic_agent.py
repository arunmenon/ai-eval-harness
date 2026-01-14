"""
Anthropic agent implementation.

This module provides an agent that uses the Anthropic API for generation,
supporting tool use.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

from ..core.agent import AgentAPIError, AgentConfig, AgentRateLimitError, AgentTimeoutError
from ..core.types import AgentResponse, Message, MessageRole, ToolCall, ToolDefinition
from .base import BaseAgent, register_agent

logger = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic, APIError, RateLimitError, APITimeoutError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@register_agent("anthropic")
class AnthropicAgent(BaseAgent):
    """
    Agent using Anthropic's API with tool use support.

    Supports:
    - Claude 3.5 Sonnet, Claude 3 Opus, and other Anthropic models
    - Tool use (function calling)
    - Streaming responses
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the Anthropic agent.

        Args:
            config: Agent configuration

        Raises:
            ImportError: If anthropic package is not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

        super().__init__(config)

        # Get API key from config or environment
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key in config."
            )

        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def _generate_impl(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Generate a response using Anthropic API.

        Args:
            messages: Conversation history
            tools: Optional list of tools
            **kwargs: Additional parameters

        Returns:
            AgentResponse
        """
        # Extract system message and convert others to Anthropic format
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": anthropic_messages,
            "max_tokens": self.config.parameters.get("max_tokens", 4096),
        }

        if system_prompt:
            params["system"] = system_prompt

        # Add other parameters
        for key in ["temperature", "top_p", "top_k"]:
            if key in self.config.parameters:
                params[key] = self.config.parameters[key]

        # Add additional kwargs
        params.update(kwargs)

        # Add tools if provided
        if tools:
            params["tools"] = [t.to_anthropic_format() for t in tools]

        try:
            response = await self.client.messages.create(**params)
        except RateLimitError as e:
            raise AgentRateLimitError(str(e)) from e
        except APITimeoutError as e:
            raise AgentTimeoutError(str(e)) from e
        except APIError as e:
            raise AgentAPIError(str(e), status_code=getattr(e, "status_code", None)) from e

        # Parse response
        content_text: str | None = None
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return AgentResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            raw_response=response,
        )

    async def _generate_stream_impl(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentResponse]:
        """
        Generate a streaming response using Anthropic API.

        Args:
            messages: Conversation history
            tools: Optional list of tools
            **kwargs: Additional parameters

        Yields:
            AgentResponse chunks
        """
        system_prompt, anthropic_messages = self._convert_messages(messages)

        params: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": anthropic_messages,
            "max_tokens": self.config.parameters.get("max_tokens", 4096),
            "stream": True,
        }

        if system_prompt:
            params["system"] = system_prompt

        for key in ["temperature", "top_p", "top_k"]:
            if key in self.config.parameters:
                params[key] = self.config.parameters[key]

        params.update(kwargs)

        if tools:
            params["tools"] = [t.to_anthropic_format() for t in tools]

        try:
            async with self.client.messages.stream(**params) as stream:
                accumulated_text = ""

                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                accumulated_text += event.delta.text
                                yield AgentResponse(
                                    content=accumulated_text,
                                    finish_reason="in_progress",
                                )

                # Get final message
                final_message = await stream.get_final_message()

                content_text: str | None = None
                tool_calls: list[ToolCall] = []

                for block in final_message.content:
                    if block.type == "text":
                        content_text = block.text
                    elif block.type == "tool_use":
                        tool_calls.append(
                            ToolCall(
                                id=block.id,
                                name=block.name,
                                arguments=block.input if isinstance(block.input, dict) else {},
                            )
                        )

                yield AgentResponse(
                    content=content_text,
                    tool_calls=tool_calls,
                    finish_reason=final_message.stop_reason or "stop",
                    usage={
                        "prompt_tokens": final_message.usage.input_tokens,
                        "completion_tokens": final_message.usage.output_tokens,
                        "total_tokens": (
                            final_message.usage.input_tokens + final_message.usage.output_tokens
                        ),
                    },
                )

        except RateLimitError as e:
            raise AgentRateLimitError(str(e)) from e
        except APITimeoutError as e:
            raise AgentTimeoutError(str(e)) from e
        except APIError as e:
            raise AgentAPIError(str(e), status_code=getattr(e, "status_code", None)) from e

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert internal Message format to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
                continue

            if msg.role == MessageRole.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.content or "",
                })

            elif msg.role == MessageRole.ASSISTANT:
                content: list[dict[str, Any]] = []

                if msg.content:
                    content.append({"type": "text", "text": msg.content})

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })

                anthropic_messages.append({
                    "role": "assistant",
                    "content": content if content else [{"type": "text", "text": ""}],
                })

            elif msg.role == MessageRole.TOOL:
                # Tool results go in user messages for Anthropic
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content or "",
                        }
                    ],
                })

        return system_prompt, anthropic_messages

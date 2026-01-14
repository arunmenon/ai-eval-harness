"""
OpenAI agent implementation.

This module provides an agent that uses the OpenAI API for generation,
supporting function calling (tools).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

from ..core.agent import AgentAPIError, AgentConfig, AgentRateLimitError, AgentTimeoutError
from ..core.types import AgentResponse, Message, MessageRole, ToolCall, ToolDefinition
from .base import BaseAgent, register_agent

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@register_agent("openai")
class OpenAIAgent(BaseAgent):
    """
    Agent using OpenAI's API with function calling support.

    Supports:
    - GPT-4, GPT-4o, GPT-3.5-turbo and other OpenAI models
    - Function calling (tools)
    - Streaming responses
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the OpenAI agent.

        Args:
            config: Agent configuration

        Raises:
            ImportError: If openai package is not installed
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        super().__init__(config)

        # Get API key from config or environment
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )

        self.client = AsyncOpenAI(
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
        Generate a response using OpenAI API.

        Args:
            messages: Conversation history
            tools: Optional list of tools
            **kwargs: Additional parameters

        Returns:
            AgentResponse
        """
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": openai_messages,
            **self.config.parameters,
            **kwargs,
        }

        # Add tools if provided
        if tools:
            params["tools"] = [t.to_openai_format() for t in tools]

        try:
            response = await self.client.chat.completions.create(**params)
        except RateLimitError as e:
            raise AgentRateLimitError(str(e)) from e
        except APITimeoutError as e:
            raise AgentTimeoutError(str(e)) from e
        except APIError as e:
            raise AgentAPIError(str(e), status_code=e.status_code) from e

        # Parse response
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": tc.function.arguments}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        return AgentResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
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
        Generate a streaming response using OpenAI API.

        Args:
            messages: Conversation history
            tools: Optional list of tools
            **kwargs: Additional parameters

        Yields:
            AgentResponse chunks
        """
        openai_messages = self._convert_messages(messages)

        params: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": openai_messages,
            "stream": True,
            **self.config.parameters,
            **kwargs,
        }

        if tools:
            params["tools"] = [t.to_openai_format() for t in tools]

        try:
            stream = await self.client.chat.completions.create(**params)

            accumulated_content = ""
            accumulated_tool_calls: dict[int, dict[str, Any]] = {}

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Accumulate content
                if delta.content:
                    accumulated_content += delta.content

                # Accumulate tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function else "",
                                "arguments": "",
                            }
                        if tc.id:
                            accumulated_tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                accumulated_tool_calls[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                accumulated_tool_calls[idx]["arguments"] += tc.function.arguments

                # Yield intermediate response
                yield AgentResponse(
                    content=accumulated_content if accumulated_content else None,
                    finish_reason=chunk.choices[0].finish_reason or "in_progress",
                )

            # Final response with complete tool calls
            tool_calls: list[ToolCall] = []
            for tc_data in accumulated_tool_calls.values():
                try:
                    arguments = json.loads(tc_data["arguments"])
                except json.JSONDecodeError:
                    arguments = {"raw": tc_data["arguments"]}
                tool_calls.append(
                    ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=arguments,
                    )
                )

            yield AgentResponse(
                content=accumulated_content if accumulated_content else None,
                tool_calls=tool_calls,
                finish_reason="stop",
            )

        except RateLimitError as e:
            raise AgentRateLimitError(str(e)) from e
        except APITimeoutError as e:
            raise AgentTimeoutError(str(e)) from e
        except APIError as e:
            raise AgentAPIError(str(e), status_code=e.status_code) from e

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert internal Message format to OpenAI format."""
        openai_messages: list[dict[str, Any]] = []

        for msg in messages:
            openai_msg: dict[str, Any] = {"role": msg.role.value}

            if msg.content is not None:
                openai_msg["content"] = msg.content

            if msg.role == MessageRole.TOOL:
                openai_msg["tool_call_id"] = msg.tool_call_id

            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            openai_messages.append(openai_msg)

        return openai_messages

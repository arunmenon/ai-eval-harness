"""
Example: Creating a Custom Agent

This example shows how to create a custom agent implementation
that wraps any LLM provider and integrates with the evaluation harness.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator

from ai_eval_harness.core.agent import Agent, AgentConfig
from ai_eval_harness.core.types import (
    AgentResponse,
    Message,
    ToolCall,
    ToolDefinition,
)
from ai_eval_harness.agents.base import register_agent


@register_agent("custom")
class CustomAgent(Agent):
    """
    Example custom agent implementation.

    This shows how to wrap any LLM provider and implement
    the required interface for the evaluation harness.
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize the custom agent.

        Args:
            config: Agent configuration with model settings
        """
        super().__init__(config)

        # Initialize your LLM client here
        # For example, using a local model, custom API, or any provider
        self._client = self._initialize_client()

        # Track conversation state
        self._conversation_history: list[dict[str, Any]] = []

    def _initialize_client(self) -> Any:
        """Initialize your LLM client."""
        # Example: Initialize a custom client
        # In a real implementation, this would connect to your LLM
        return None

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """
        Generate a response from the agent.

        This is the main method that the evaluation harness calls.
        Implement your LLM interaction logic here.

        Args:
            messages: Conversation history
            tools: Available tools the agent can use
            **kwargs: Additional generation parameters

        Returns:
            AgentResponse with content and optional tool calls
        """
        # Convert messages to your LLM's format
        formatted_messages = self._format_messages(messages)

        # Convert tools to your LLM's format (if supported)
        formatted_tools = self._format_tools(tools) if tools else None

        # Call your LLM
        response = await self._call_llm(formatted_messages, formatted_tools)

        # Parse the response
        content = response.get("content", "")
        tool_calls = self._parse_tool_calls(response)

        return AgentResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage={
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
            },
        )

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response.

        Implement this if your LLM supports streaming.
        """
        # For non-streaming LLMs, you can yield the full response
        response = await self.generate(messages, tools, **kwargs)
        if response.content:
            yield response.content

    def reset(self) -> None:
        """Reset agent state between tasks."""
        self._conversation_history = []

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to your LLM's format."""
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        return formatted

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tools to your LLM's format."""
        formatted = []
        for tool in tools:
            formatted.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        p.name: {
                            "type": p.type,
                            "description": p.description,
                        }
                        for p in tool.parameters
                    },
                    "required": [p.name for p in tool.parameters if p.required],
                },
            })
        return formatted

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Call your LLM and return the response.

        This is where you implement the actual LLM API call.
        """
        # Example: Simple mock response for demonstration
        # In a real implementation, call your LLM API here

        # Simulate LLM response
        await asyncio.sleep(0.1)

        return {
            "content": "This is a sample response from the custom agent.",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "tool_calls": [],  # No tool calls in this example
        }

    def _parse_tool_calls(self, response: dict[str, Any]) -> list[ToolCall]:
        """Parse tool calls from the LLM response."""
        tool_calls = []
        for tc in response.get("tool_calls", []):
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                arguments=tc.get("arguments", {}),
            ))
        return tool_calls


# Example: Chain-of-Thought Agent
@register_agent("cot")
class ChainOfThoughtAgent(Agent):
    """
    Example agent that uses chain-of-thought prompting.

    This wraps another agent and adds CoT reasoning.
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)

        # Use OpenAI as the base model
        from ai_eval_harness.agents.openai_agent import OpenAIAgent
        self._base_agent = OpenAIAgent(config)

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """Generate with chain-of-thought prompting."""
        # Add CoT instruction to the last user message
        cot_messages = list(messages)

        # Find and modify the last user message
        for i in range(len(cot_messages) - 1, -1, -1):
            if cot_messages[i].role.value == "user":
                original = cot_messages[i].content
                cot_messages[i] = Message.user(
                    f"{original}\n\nLet's think through this step by step:"
                )
                break

        # Call the base agent
        return await self._base_agent.generate(cot_messages, tools, **kwargs)

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async for chunk in self._base_agent.generate_stream(messages, tools, **kwargs):
            yield chunk

    def reset(self) -> None:
        self._base_agent.reset()


# Example: ReAct Agent
@register_agent("react")
class ReActAgent(Agent):
    """
    Example agent implementing ReAct (Reasoning + Acting).

    Uses explicit thought/action/observation format.
    """

    REACT_PROMPT = """You are a helpful assistant that thinks step by step.

For each step, you should:
1. **Thought**: Reason about what to do next
2. **Action**: Choose an action (use a tool or respond)
3. **Observation**: Observe the result

When you have enough information, provide your final answer.

Format your responses as:
Thought: [your reasoning]
Action: [tool_name] OR Answer: [final response]
"""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)

        from ai_eval_harness.agents.openai_agent import OpenAIAgent
        self._base_agent = OpenAIAgent(config)

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """Generate with ReAct prompting."""
        # Prepend ReAct system prompt
        react_messages = [Message.system(self.REACT_PROMPT)] + list(messages)

        return await self._base_agent.generate(react_messages, tools, **kwargs)

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        react_messages = [Message.system(self.REACT_PROMPT)] + list(messages)
        async for chunk in self._base_agent.generate_stream(react_messages, tools, **kwargs):
            yield chunk

    def reset(self) -> None:
        self._base_agent.reset()


async def main() -> None:
    """Example usage of custom agents."""
    from ai_eval_harness.agents.base import get_agent

    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this example")
        return

    # Test the custom agent
    print("Testing CustomAgent...")
    agent = get_agent(
        provider="custom",
        model_name="custom-model",
        api_key=api_key,
    )

    messages = [Message.user("What is 2 + 2?")]
    response = await agent.generate(messages)
    print(f"Custom Agent: {response.content}\n")

    # Test Chain-of-Thought agent
    print("Testing ChainOfThoughtAgent...")
    cot_agent = get_agent(
        provider="cot",
        model_name="gpt-4o",
        api_key=api_key,
    )

    response = await cot_agent.generate(messages)
    print(f"CoT Agent: {response.content}\n")

    # Test ReAct agent
    print("Testing ReActAgent...")
    react_agent = get_agent(
        provider="react",
        model_name="gpt-4o",
        api_key=api_key,
    )

    response = await react_agent.generate(messages)
    print(f"ReAct Agent: {response.content}\n")


if __name__ == "__main__":
    asyncio.run(main())

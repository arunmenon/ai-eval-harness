"""
User simulator for τ-bench.

This module provides LLM-based user simulation for multi-turn
conversations in τ-bench tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...core.types import Message, MessageRole

if TYPE_CHECKING:
    from ...core.agent import Agent

logger = logging.getLogger(__name__)


USER_SIMULATOR_PROMPT = """You are simulating a customer interacting with a customer service agent.

Your goal: {user_goal}

Your persona:
- You are {persona_description}
- You have the following information: {user_info}

Guidelines:
1. Stay in character and pursue your goal naturally
2. Respond naturally to the agent's questions and actions
3. Confirm actions when asked (say "yes" to proceed, "no" to decline)
4. Express satisfaction or dissatisfaction based on how well the agent helps you
5. End the conversation when your goal is achieved or you give up

When your goal is achieved, include "[GOAL_ACHIEVED]" in your response.
When you want to end without achieving the goal, include "[CONVERSATION_END]" in your response.

Respond as the customer would, keeping messages concise and natural."""


class UserSimulator:
    """
    LLM-based user simulator for τ-bench.

    Simulates a user interacting with the agent to complete a task,
    responding naturally to agent messages and confirming actions.
    """

    def __init__(
        self,
        agent: "Agent",
        user_goal: str,
        user_info: dict[str, Any] | None = None,
        persona: str = "a typical customer",
    ) -> None:
        """
        Initialize the user simulator.

        Args:
            agent: The LLM agent to use for simulation
            user_goal: What the user is trying to achieve
            user_info: Information the user has (order ID, account details, etc.)
            persona: Description of the user's persona
        """
        self.agent = agent
        self.user_goal = user_goal
        self.user_info = user_info or {}
        self.persona = persona
        self._conversation_history: list[Message] = []
        self._is_complete = False

    async def initialize(self) -> None:
        """Initialize the simulator state."""
        self._conversation_history = []
        self._is_complete = False

        # Build system prompt
        system_prompt = USER_SIMULATOR_PROMPT.format(
            user_goal=self.user_goal,
            persona_description=self.persona,
            user_info=str(self.user_info),
        )
        self._conversation_history.append(Message.system(system_prompt))

    async def generate_initial_message(self) -> str:
        """
        Generate the initial user message to start the conversation.

        Returns:
            The initial user message
        """
        # Prompt for initial message
        self._conversation_history.append(
            Message.user("Generate your opening message to the customer service agent.")
        )

        response = await self.agent.generate(self._conversation_history)

        # Store as assistant in our history (we're the user in the main conversation)
        self._conversation_history.append(Message.assistant(response.content))

        return response.content or "Hello, I need help with something."

    async def generate_response(
        self,
        agent_message: str | None,
        agent_tool_calls: list[Any] | None = None,
    ) -> tuple[str, bool]:
        """
        Generate a user response to the agent's message.

        Args:
            agent_message: The agent's message to respond to
            agent_tool_calls: Any tool calls the agent made

        Returns:
            Tuple of (user_response, is_conversation_complete)
        """
        if self._is_complete:
            return "[CONVERSATION_END]", True

        # Build context about what the agent did
        context_parts = []
        if agent_message:
            context_parts.append(f"Agent said: {agent_message}")
        if agent_tool_calls:
            for tc in agent_tool_calls:
                context_parts.append(f"Agent used tool: {tc.name} with args: {tc.arguments}")

        context = "\n".join(context_parts) if context_parts else "Agent is waiting for your response."

        # Ask the simulator to respond
        self._conversation_history.append(
            Message.user(f"Respond to this:\n{context}")
        )

        response = await self.agent.generate(self._conversation_history)
        content = response.content or ""

        self._conversation_history.append(Message.assistant(content))

        # Check for completion markers
        if "[GOAL_ACHIEVED]" in content or "[CONVERSATION_END]" in content:
            self._is_complete = True
            # Clean up the content
            content = content.replace("[GOAL_ACHIEVED]", "").replace("[CONVERSATION_END]", "").strip()
            return content, True

        return content, False

    @property
    def is_complete(self) -> bool:
        """Whether the conversation is complete."""
        return self._is_complete


class SimpleUserSimulator:
    """
    Simple rule-based user simulator for testing.

    Uses predefined responses based on patterns in the agent's messages.
    Useful for testing without requiring an LLM.
    """

    def __init__(
        self,
        user_goal: str,
        scripted_responses: list[str] | None = None,
    ) -> None:
        """
        Initialize the simple simulator.

        Args:
            user_goal: The user's goal
            scripted_responses: Optional list of scripted responses
        """
        self.user_goal = user_goal
        self.scripted_responses = scripted_responses or []
        self._response_index = 0
        self._is_complete = False

    async def initialize(self) -> None:
        """Initialize the simulator."""
        self._response_index = 0
        self._is_complete = False

    async def generate_initial_message(self) -> str:
        """Generate initial message."""
        return f"Hi, I need help with: {self.user_goal}"

    async def generate_response(
        self,
        agent_message: str | None,
        agent_tool_calls: list[Any] | None = None,
    ) -> tuple[str, bool]:
        """Generate response based on scripted responses or simple rules."""
        if self._is_complete:
            return "Thank you, goodbye!", True

        # Use scripted response if available
        if self._response_index < len(self.scripted_responses):
            response = self.scripted_responses[self._response_index]
            self._response_index += 1
            return response, False

        # Simple rule-based responses
        if agent_message:
            msg_lower = agent_message.lower()

            # Confirmation requests
            if "confirm" in msg_lower or "proceed" in msg_lower or "would you like" in msg_lower:
                return "Yes, please proceed.", False

            # Success indicators
            if "successfully" in msg_lower or "completed" in msg_lower or "done" in msg_lower:
                self._is_complete = True
                return "Thank you, that's exactly what I needed!", True

            # Questions
            if "?" in agent_message:
                return "Yes.", False

        # Default response
        return "Okay, please continue.", False

    @property
    def is_complete(self) -> bool:
        """Whether the conversation is complete."""
        return self._is_complete

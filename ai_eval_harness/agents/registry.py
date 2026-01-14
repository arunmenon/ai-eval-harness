"""
Agent registry module.

This module provides a unified interface for registering and retrieving
agent implementations. Import this module to ensure all built-in agents
are registered.
"""

from ..core.agent import Agent, AgentConfig
from .base import get_agent, list_agents, register_agent

# Import agent implementations to trigger registration
from .openai_agent import OpenAIAgent
from .anthropic_agent import AnthropicAgent

__all__ = [
    "Agent",
    "AgentConfig",
    "OpenAIAgent",
    "AnthropicAgent",
    "get_agent",
    "list_agents",
    "register_agent",
]

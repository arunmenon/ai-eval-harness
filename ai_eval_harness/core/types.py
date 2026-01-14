"""
Core type definitions for the AI evaluation harness.

This module defines the fundamental data structures used throughout the harness
for representing messages, tool calls, agent responses, tasks, and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class MessageRole(Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    Represents a message in the conversation.

    Attributes:
        role: The role of the message sender
        content: The text content of the message
        name: Optional name for tool messages (the tool name)
        tool_call_id: Reference to the tool call this message responds to
        tool_calls: List of tool calls made in this message (for assistant messages)
        metadata: Additional metadata about the message
    """

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            result["content"] = self.content
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list["ToolCall"] | None = None
    ) -> "Message":
        """Create an assistant message."""
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
        )

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> "Message":
        """Create a tool result message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.

    Attributes:
        name: Parameter name
        type: JSON Schema type (string, number, boolean, object, array)
        description: Description of the parameter
        required: Whether the parameter is required
        enum: Optional list of allowed values
        default: Optional default value
    """

    name: str
    type: str
    description: str
    required: bool = True
    enum: list[Any] | None = None
    default: Any | None = None


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool for LLM function calling.

    Attributes:
        name: Tool name (function name)
        description: Description of what the tool does
        parameters: List of parameter definitions
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolCall:
    """
    Represents a tool invocation request from the agent.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to invoke
        arguments: Arguments to pass to the tool
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ToolResult:
    """
    Represents the result of a tool execution.

    Attributes:
        tool_call_id: ID of the tool call this result corresponds to
        output: The output from the tool execution
        error: Error message if the tool failed
        execution_time_ms: Time taken to execute the tool in milliseconds
    """

    tool_call_id: str
    output: Any
    error: str | None = None
    execution_time_ms: float = 0.0

    @property
    def success(self) -> bool:
        """Whether the tool execution was successful."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_call_id": self.tool_call_id,
            "output": str(self.output) if self.output is not None else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
        }


@dataclass
class AgentResponse:
    """
    Agent's response including content and tool calls.

    Attributes:
        content: Text content of the response
        tool_calls: List of tool calls requested by the agent
        finish_reason: Reason the generation finished (stop, tool_calls, length, etc.)
        usage: Token usage statistics
        raw_response: Original API response for debugging
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the response contains tool calls."""
        return len(self.tool_calls) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": self.usage,
        }


@dataclass
class Task:
    """
    Represents a single benchmark task.

    Attributes:
        task_id: Unique identifier for the task
        instruction: The task instruction/question
        metadata: Additional task metadata
        attached_files: Paths to any attached files
        ground_truth: Expected answer (hidden for test splits)
        difficulty: Difficulty level (1-3 for GAIA)
        domain: Domain name (for τ-bench)
    """

    task_id: str
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)
    attached_files: list[Path] = field(default_factory=list)
    ground_truth: Any = None
    difficulty: int | None = None
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "metadata": self.metadata,
            "attached_files": [str(f) for f in self.attached_files],
            "difficulty": self.difficulty,
            "domain": self.domain,
        }


@dataclass
class TaskResult:
    """
    Result of running a single task.

    Attributes:
        task_id: ID of the task
        agent_answer: The agent's final answer
        is_correct: Whether the answer was correct (None if not scored)
        score: Numeric score (0.0-1.0)
        trace_id: ID of the trace for this task
        metrics: Task-specific metrics
        error: Error message if the task failed
    """

    task_id: str
    agent_answer: Any
    is_correct: bool | None = None
    score: float = 0.0
    trace_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "agent_answer": str(self.agent_answer) if self.agent_answer is not None else None,
            "is_correct": self.is_correct,
            "score": self.score,
            "trace_id": self.trace_id,
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass
class BenchmarkResult:
    """
    Aggregated results for a benchmark run.

    Attributes:
        benchmark_name: Name of the benchmark
        run_id: Unique identifier for this run
        task_results: Results for each task
        aggregate_metrics: Aggregated metrics across all tasks
        config: Configuration used for the run
        timestamp: When the run was executed
    """

    benchmark_name: str
    run_id: str
    task_results: list[TaskResult]
    aggregate_metrics: dict[str, float]
    config: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        scored = [r for r in self.task_results if r.is_correct is not None]
        if not scored:
            return 0.0
        return sum(r.is_correct for r in scored) / len(scored)

    @property
    def total_tasks(self) -> int:
        """Total number of tasks."""
        return len(self.task_results)

    @property
    def successful_tasks(self) -> int:
        """Number of successful tasks."""
        return sum(1 for r in self.task_results if r.is_correct)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "run_id": self.run_id,
            "task_results": [r.to_dict() for r in self.task_results],
            "aggregate_metrics": self.aggregate_metrics,
            "config": self.config,
            "timestamp": self.timestamp,
            "success_rate": self.success_rate,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
        }


class PolicyViolationType(Enum):
    """Types of policy violations in τ-bench/τ²-bench."""

    MISSING_CONFIRMATION = "missing_confirmation"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    INVALID_ARGUMENT = "invalid_argument"
    EXCEEDED_LIMIT = "exceeded_limit"
    WRONG_ORDER = "wrong_order"
    MISSING_VERIFICATION = "missing_verification"


@dataclass
class PolicyViolation:
    """
    Represents a policy violation in τ-bench/τ²-bench.

    Attributes:
        violation_type: Type of the violation
        description: Human-readable description
        tool_call: The tool call that caused the violation
        turn: Turn number when the violation occurred
    """

    violation_type: PolicyViolationType
    description: str
    tool_call: ToolCall | None = None
    turn: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violation_type": self.violation_type.value,
            "description": self.description,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "turn": self.turn,
        }

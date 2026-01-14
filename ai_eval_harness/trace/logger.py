"""
Trace logging system for the AI evaluation harness.

This module provides comprehensive trace logging for benchmark runs,
capturing all interactions for debugging, analysis, and replay.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.types import (
        AgentResponse,
        BenchmarkResult,
        PolicyViolation,
        ToolCall,
        ToolResult,
    )


@dataclass
class TraceEvent:
    """
    Single event in a trace.

    Attributes:
        timestamp: ISO format timestamp
        event_type: Type of event (task_start, agent_response, tool_call, etc.)
        data: Event-specific data
    """

    timestamp: str
    event_type: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "data": self.data,
        }


@dataclass
class TaskTrace:
    """
    Complete trace for a single task.

    Attributes:
        task_id: ID of the task
        trace_id: Unique trace identifier
        events: List of trace events
        metadata: Additional metadata
        outcome: Whether the task was successful
    """

    task_id: str
    trace_id: str
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    outcome: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
            "outcome": self.outcome,
        }


class TraceLogger:
    """
    Comprehensive trace logging system for benchmark runs.

    Captures all interactions for debugging, analysis, and replay.
    Events are logged with timestamps and can be exported to JSON.

    Example:
        ```python
        logger = TraceLogger(output_dir=Path("./traces"))
        logger.start_task("task_123")
        logger.log_agent_response(response)
        logger.log_tool_call(tool_call, result)
        logger.end_task("task_123", success=True)
        ```
    """

    def __init__(
        self,
        output_dir: Path,
        run_id: str | None = None,
        include_raw_responses: bool = True,
        max_output_length: int = 10000,
    ) -> None:
        """
        Initialize the trace logger.

        Args:
            output_dir: Directory to save trace files
            run_id: Unique run identifier (auto-generated if not provided)
            include_raw_responses: Whether to include raw API responses
            max_output_length: Maximum length for tool outputs (truncated if longer)
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.include_raw_responses = include_raw_responses
        self.max_output_length = max_output_length

        self._current_task_trace: TaskTrace | None = None
        self._all_traces: list[TaskTrace] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_trace_id(self) -> str:
        """Get the current trace ID."""
        if self._current_task_trace:
            return self._current_task_trace.trace_id
        return ""

    def start_task(self, task_id: str, metadata: dict[str, Any] | None = None) -> str:
        """
        Start tracing a new task.

        Args:
            task_id: ID of the task
            metadata: Optional task metadata

        Returns:
            Unique trace ID for this task
        """
        trace_id = f"{self.run_id}_{task_id}_{uuid.uuid4().hex[:8]}"

        self._current_task_trace = TaskTrace(
            task_id=task_id,
            trace_id=trace_id,
            metadata=metadata or {},
        )

        self._add_event(
            "task_start",
            {
                "task_id": task_id,
                "trace_id": trace_id,
                "metadata": metadata or {},
            },
        )

        return trace_id

    def end_task(self, task_id: str, outcome: bool) -> None:
        """
        End current task trace.

        Args:
            task_id: ID of the task
            outcome: Whether the task was successful
        """
        if self._current_task_trace:
            self._current_task_trace.outcome = outcome
            self._add_event(
                "task_end",
                {
                    "task_id": task_id,
                    "outcome": outcome,
                },
            )

            # Save individual trace
            self._save_trace(self._current_task_trace)
            self._all_traces.append(self._current_task_trace)
            self._current_task_trace = None

    def log_agent_response(self, response: "AgentResponse") -> None:
        """
        Log an agent response.

        Args:
            response: The agent's response
        """
        data: dict[str, Any] = {
            "content": response.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ],
            "finish_reason": response.finish_reason,
            "usage": response.usage,
        }
        if self.include_raw_responses and response.raw_response:
            data["raw_response"] = self._truncate(str(response.raw_response))

        self._add_event("agent_response", data)

    def log_tool_call(self, tool_call: "ToolCall", result: "ToolResult") -> None:
        """
        Log a tool call and its result.

        Args:
            tool_call: The tool call
            result: The result of executing the tool
        """
        self._add_event(
            "tool_call",
            {
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "arguments": tool_call.arguments,
                "result": {
                    "output": self._truncate(str(result.output)),
                    "error": result.error,
                    "execution_time_ms": result.execution_time_ms,
                    "success": result.success,
                },
            },
        )

    def log_user_message(
        self,
        message: str,
        tool_calls: list["ToolCall"] | None = None,
    ) -> None:
        """
        Log a user simulator message.

        Args:
            message: The user's message content
            tool_calls: Optional list of tool calls made by the user (τ²-bench)
        """
        self._add_event(
            "user_message",
            {
                "content": message,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in (tool_calls or [])
                ],
            },
        )

    def log_user_tool_call(self, tool_call: "ToolCall", result: "ToolResult" | None = None) -> None:
        """
        Log a user tool call (τ²-bench specific).

        Args:
            tool_call: The tool call made by the user
            result: Optional result of the tool call
        """
        data: dict[str, Any] = {
            "tool_name": tool_call.name,
            "arguments": tool_call.arguments,
        }
        if result:
            data["result"] = {
                "output": self._truncate(str(result.output)),
                "error": result.error,
            }
        self._add_event("user_tool_call", data)

    def log_policy_violation(self, violation: "PolicyViolation") -> None:
        """
        Log a policy violation.

        Args:
            violation: The policy violation
        """
        self._add_event("policy_violation", violation.to_dict())

    def log_state_change(self, old_state: dict[str, Any], new_state: dict[str, Any]) -> None:
        """
        Log environment state change.

        Args:
            old_state: State before the change
            new_state: State after the change
        """
        self._add_event(
            "state_change",
            {
                "old_state": old_state,
                "new_state": new_state,
            },
        )

    def log_custom(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Log a custom event.

        Args:
            event_type: Type of the event
            data: Event data
        """
        self._add_event(event_type, data)

    def _add_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Add an event to the current trace."""
        if self._current_task_trace:
            event = TraceEvent(
                timestamp=datetime.utcnow().isoformat(),
                event_type=event_type,
                data=data,
            )
            self._current_task_trace.events.append(event)

    def _truncate(self, text: str) -> str:
        """Truncate text to max length."""
        if len(text) > self.max_output_length:
            return text[: self.max_output_length] + f"... [truncated, total {len(text)} chars]"
        return text

    def _save_trace(self, trace: TaskTrace) -> None:
        """Save a trace to disk."""
        trace_file = self.output_dir / f"{trace.trace_id}.json"

        with open(trace_file, "w") as f:
            json.dump(trace.to_dict(), f, indent=2, default=str)

    def save_run_summary(self, benchmark_result: "BenchmarkResult") -> Path:
        """
        Save complete run summary.

        Args:
            benchmark_result: The benchmark result to save

        Returns:
            Path to the saved summary file
        """
        summary_file = self.output_dir / f"run_summary_{self.run_id}.json"

        summary = {
            "run_id": self.run_id,
            "benchmark": benchmark_result.benchmark_name,
            "timestamp": benchmark_result.timestamp,
            "config": benchmark_result.config,
            "aggregate_metrics": benchmark_result.aggregate_metrics,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "is_correct": r.is_correct,
                    "score": r.score,
                    "trace_id": r.trace_id,
                    "metrics": r.metrics,
                    "error": r.error,
                }
                for r in benchmark_result.task_results
            ],
            "total_tasks": benchmark_result.total_tasks,
            "success_rate": benchmark_result.success_rate,
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary_file

    def get_all_traces(self) -> list[TaskTrace]:
        """Get all completed traces."""
        return self._all_traces.copy()

"""
GAIA benchmark implementation.

This module provides the main GAIA benchmark class that orchestrates
task execution, tool use, and scoring.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...core.benchmark import Benchmark, BenchmarkConfig, register_benchmark
from ...core.types import Message, Task, TaskResult
from .dataloader import GAIADataLoader
from .scorer import GAIAScorer, extract_final_answer
from .tools import Tool, get_gaia_tools

if TYPE_CHECKING:
    from ...core.agent import Agent
    from ...trace.logger import TraceLogger

logger = logging.getLogger(__name__)


GAIA_SYSTEM_PROMPT = """You are a helpful AI assistant capable of solving complex tasks.

You have access to the following tools:
- web_browser: Fetch content from URLs
- python: Execute Python code
- bash: Execute shell commands
- read_file: Read file contents

When you have found the answer, respond with:
FINAL ANSWER: <your answer>

The answer should be concise and directly answer the question.
For numeric answers, provide just the number.
For lists, use comma separation.
Do not include explanations in your final answer.

Think step by step and use tools as needed to find the answer."""


@register_benchmark("gaia")
class GAIABenchmark(Benchmark):
    """
    GAIA Benchmark implementation.

    GAIA tests general AI assistant capabilities on real-world tasks
    requiring reasoning, web browsing, and tool use. Tasks have three
    difficulty levels with varying complexity.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """
        Initialize the GAIA benchmark.

        Args:
            config: Benchmark configuration
        """
        super().__init__(config)

        self.dataloader = GAIADataLoader(
            hf_token=os.environ.get("HF_TOKEN"),
            data_dir=config.data_dir,
        )
        self.scorer = GAIAScorer()
        self._tools: list[Tool] = []
        self._file_base_dir: Path | None = None

    @property
    def name(self) -> str:
        return "gaia"

    def _initialize_tools(self) -> None:
        """Initialize tools for GAIA tasks."""
        self._tools = get_gaia_tools(
            file_base_dir=self._file_base_dir,
            sandbox_mode=False,  # TODO: Add config option
        )

    async def load_tasks(self) -> list[Task]:
        """Load GAIA tasks from HuggingFace."""
        raw_tasks = await self.dataloader.load(
            split=self.config.split,
            subset=self.config.subset,
        )

        # Store file base dir for tools
        if self.dataloader._dataset_path:
            self._file_base_dir = self.dataloader._dataset_path

        self._initialize_tools()

        tasks = []
        for item in raw_tasks:
            attached_files: list[Path] = []
            if item.get("file_name"):
                file_path = self.dataloader.get_file_path(item["file_name"])
                if file_path:
                    attached_files.append(file_path)

            task = Task(
                task_id=item.get("task_id", ""),
                instruction=item.get("Question", ""),
                metadata={
                    "annotator_metadata": item.get("Annotator Metadata", {}),
                    "file_name": item.get("file_name"),
                },
                attached_files=attached_files,
                ground_truth=item.get("Final answer"),
                difficulty=int(item.get("Level", 1)),
            )
            tasks.append(task)

        return tasks

    async def run_task(
        self,
        task: Task,
        agent: "Agent",
        trace_logger: "TraceLogger",
    ) -> TaskResult:
        """
        Execute a GAIA task with agentic loop.

        Args:
            task: The task to execute
            agent: The agent to evaluate
            trace_logger: Logger for recording execution trace

        Returns:
            TaskResult with answer and metrics
        """
        trace_logger.start_task(task.task_id, {"difficulty": task.difficulty})

        # Build system prompt
        system_prompt = self._build_system_prompt(task)

        messages: list[Message] = [
            Message.system(system_prompt),
            Message.user(task.instruction),
        ]

        # Add file context if present
        if task.attached_files:
            file_context = await self._load_file_context(task.attached_files)
            if file_context:
                messages.append(Message.user(f"Attached file content:\n{file_context}"))

        tool_definitions = [t.definition for t in self._tools]
        final_answer: str | None = None
        turn_count = 0
        tool_call_count = 0
        tool_names_used: list[str] = []

        while turn_count < self.config.max_turns:
            turn_count += 1

            # Get agent response
            response = await agent.generate(
                messages=messages,
                tools=tool_definitions,
            )

            trace_logger.log_agent_response(response)

            # Check for final answer in content
            if response.content:
                extracted = extract_final_answer(response.content)
                if extracted:
                    final_answer = extracted
                    break

            # Process tool calls
            if response.tool_calls:
                # Add assistant message with tool calls
                messages.append(Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                for tool_call in response.tool_calls:
                    if tool_call_count >= self.config.max_tool_calls_per_turn * self.config.max_turns:
                        break

                    tool_call_count += 1
                    tool_names_used.append(tool_call.name)

                    # Execute tool
                    tool = self._get_tool(tool_call.name)
                    if tool:
                        result = await tool.execute(
                            tool_call_id=tool_call.id,
                            **tool_call.arguments,
                        )
                    else:
                        from ...core.types import ToolResult
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            output=None,
                            error=f"Unknown tool: {tool_call.name}",
                        )

                    trace_logger.log_tool_call(tool_call, result)

                    # Add tool result to messages
                    messages.append(Message.tool(
                        content=str(result.output) if result.output else str(result.error),
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    ))

            # No tool calls and no final answer - prompt for answer
            elif not response.tool_calls and not final_answer:
                if turn_count >= 3:
                    # After a few turns, ask for the answer
                    messages.append(Message.user(
                        "Please provide your FINAL ANSWER: based on what you've found."
                    ))
                else:
                    # Add assistant response
                    if response.content:
                        messages.append(Message.assistant(content=response.content))

        # If no explicit answer, try to extract from last response
        if not final_answer and messages:
            for msg in reversed(messages):
                if msg.role.value == "assistant" and msg.content:
                    final_answer = extract_final_answer(msg.content)
                    if final_answer:
                        break

        # Score the answer
        is_correct, score = self.score_answer(task, final_answer)

        trace_logger.end_task(task.task_id, is_correct or False)

        return TaskResult(
            task_id=task.task_id,
            agent_answer=final_answer,
            is_correct=is_correct if task.ground_truth else None,
            score=score,
            trace_id=trace_logger.current_trace_id,
            metrics={
                "turns": turn_count,
                "tool_calls": tool_call_count,
                "tools_used": list(set(tool_names_used)),
                "difficulty": task.difficulty,
            },
        )

    def score_answer(self, task: Task, answer: Any) -> tuple[bool, float]:
        """Score answer using GAIA's flexible matching."""
        if task.ground_truth is None:
            return False, 0.0
        return self.scorer.score(answer, task.ground_truth)

    def _compute_aggregate_metrics(
        self,
        results: list[TaskResult],
    ) -> dict[str, float]:
        """Compute GAIA-specific metrics."""
        valid_results = [r for r in results if r.is_correct is not None]

        if not valid_results:
            return {"total_tasks": len(results)}

        # Overall accuracy
        accuracy = sum(1 for r in valid_results if r.is_correct) / len(valid_results)

        # Per-level accuracy
        level_metrics: dict[str, float] = {}
        for level in [1, 2, 3]:
            level_results = [
                r for r in valid_results if r.metrics.get("difficulty") == level
            ]
            if level_results:
                level_acc = sum(1 for r in level_results if r.is_correct) / len(level_results)
                level_metrics[f"accuracy_level_{level}"] = level_acc
                level_metrics[f"count_level_{level}"] = float(len(level_results))

        # Efficiency metrics
        avg_turns = sum(r.metrics.get("turns", 0) for r in valid_results) / len(valid_results)
        avg_tool_calls = sum(
            r.metrics.get("tool_calls", 0) for r in valid_results
        ) / len(valid_results)

        return {
            "accuracy": accuracy,
            **level_metrics,
            "avg_turns": avg_turns,
            "avg_tool_calls": avg_tool_calls,
            "total_tasks": float(len(valid_results)),
        }

    def _build_system_prompt(self, task: Task) -> str:
        """Build system prompt for a task."""
        prompt = GAIA_SYSTEM_PROMPT

        # Add file info if present
        if task.attached_files:
            file_names = [f.name for f in task.attached_files]
            prompt += f"\n\nYou have access to the following files: {', '.join(file_names)}"
            prompt += "\nUse the read_file tool to access their contents."

        return prompt

    async def _load_file_context(self, files: list[Path]) -> str:
        """Load content from attached files for initial context."""
        parts = []
        for file_path in files:
            try:
                # For small text files, include directly
                if file_path.suffix.lower() in [".txt", ".md", ".csv", ".json"]:
                    if file_path.stat().st_size < 10000:
                        content = file_path.read_text()
                        parts.append(f"=== {file_path.name} ===\n{content}")
                else:
                    parts.append(f"[File: {file_path.name} - use read_file tool to access]")
            except Exception as e:
                logger.warning(f"Could not load file {file_path}: {e}")
                parts.append(f"[File: {file_path.name} - error loading: {e}]")

        return "\n\n".join(parts)

    def _get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        for tool in self._tools:
            if tool.definition.name == name:
                return tool
        return None

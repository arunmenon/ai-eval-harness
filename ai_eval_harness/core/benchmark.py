"""
Abstract Benchmark interface for the AI evaluation harness.

This module defines the abstract base class for benchmarks. Each benchmark
(GAIA, τ-bench, τ²-bench) implements this interface with its specific
data loading, task execution, and scoring logic.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .types import BenchmarkResult, Task, TaskResult

if TYPE_CHECKING:
    from .agent import Agent
    from ..trace.logger import TraceLogger


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark execution.

    Attributes:
        data_dir: Directory containing benchmark data
        split: Data split to use (validation, test)
        subset: Optional subset (e.g., level1, retail)
        max_tasks: Maximum number of tasks to run
        shuffle: Whether to shuffle tasks
        seed: Random seed for reproducibility
        max_turns: Maximum conversation turns per task
        max_tool_calls_per_turn: Maximum tool calls per turn
        timeout_per_task_seconds: Timeout for each task
    """

    data_dir: Path | None = None
    split: str = "validation"
    subset: str | None = None
    max_tasks: int | None = None
    shuffle: bool = False
    seed: int = 42
    max_turns: int = 50
    max_tool_calls_per_turn: int = 10
    timeout_per_task_seconds: float = 300.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_dir": str(self.data_dir) if self.data_dir else None,
            "split": self.split,
            "subset": self.subset,
            "max_tasks": self.max_tasks,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "max_turns": self.max_turns,
            "max_tool_calls_per_turn": self.max_tool_calls_per_turn,
            "timeout_per_task_seconds": self.timeout_per_task_seconds,
        }


class Benchmark(ABC):
    """
    Abstract base class for benchmarks.

    Each benchmark implementation handles its own:
    - Data loading (from HuggingFace, GitHub, local files)
    - Task execution (agentic loop, conversation orchestration)
    - Scoring logic (answer matching, state comparison)

    Example:
        ```python
        class MyBenchmark(Benchmark):
            @property
            def name(self) -> str:
                return "my_benchmark"

            async def load_tasks(self) -> list[Task]:
                # Load tasks from your data source
                return tasks

            async def run_task(self, task, agent, trace_logger) -> TaskResult:
                # Execute the task with the agent
                return result

            def score_answer(self, task, answer) -> tuple[bool, float]:
                # Compare answer to ground truth
                return is_correct, score
        ```
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """
        Initialize the benchmark with configuration.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._tasks: list[Task] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Benchmark identifier.

        Returns:
            Unique string identifier for this benchmark
        """
        pass

    @abstractmethod
    async def load_tasks(self) -> list[Task]:
        """
        Load tasks from the benchmark dataset.

        This method should:
        1. Load data from the appropriate source
        2. Parse into Task objects
        3. Apply any filtering based on config

        Returns:
            List of Task objects to evaluate

        Raises:
            BenchmarkDataError: If data loading fails
        """
        pass

    @abstractmethod
    async def run_task(
        self,
        task: Task,
        agent: "Agent",
        trace_logger: "TraceLogger",
    ) -> TaskResult:
        """
        Execute a single task with the given agent.

        This is the core execution method that:
        1. Initializes the task context
        2. Runs the agentic loop (generate → tool call → response)
        3. Extracts the final answer
        4. Computes task-specific metrics

        Args:
            task: The task to execute
            agent: The agent to evaluate
            trace_logger: Logger for recording execution trace

        Returns:
            TaskResult with answer and metrics

        Raises:
            BenchmarkExecutionError: If task execution fails
        """
        pass

    @abstractmethod
    def score_answer(self, task: Task, answer: Any) -> tuple[bool, float]:
        """
        Score an answer against ground truth.

        Args:
            task: The task being scored
            answer: The agent's answer

        Returns:
            Tuple of (is_correct, score) where:
            - is_correct: Boolean indicating exact correctness
            - score: Float score between 0 and 1
        """
        pass

    async def run(
        self,
        agent: "Agent",
        trace_logger: "TraceLogger",
        max_concurrency: int = 1,
        task_filter: Callable[[Task], bool] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        """
        Run the full benchmark.

        Args:
            agent: Agent to evaluate
            trace_logger: Trace logging system
            max_concurrency: Number of concurrent tasks
            task_filter: Optional filter for selecting tasks
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            BenchmarkResult with all task results and metrics
        """
        # Load tasks
        tasks = await self.load_tasks()

        # Apply filter
        if task_filter:
            tasks = [t for t in tasks if task_filter(t)]

        # Apply max_tasks limit
        if self.config.max_tasks:
            tasks = tasks[: self.config.max_tasks]

        # Shuffle if requested
        if self.config.shuffle:
            import random

            rng = random.Random(self.config.seed)
            rng.shuffle(tasks)

        # Run tasks with concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)
        completed = 0

        async def run_with_semaphore(task: Task) -> TaskResult:
            nonlocal completed
            async with semaphore:
                try:
                    agent.reset()
                    result = await asyncio.wait_for(
                        self.run_task(task, agent, trace_logger),
                        timeout=self.config.timeout_per_task_seconds,
                    )
                except asyncio.TimeoutError:
                    result = TaskResult(
                        task_id=task.task_id,
                        agent_answer=None,
                        error=f"Task timed out after {self.config.timeout_per_task_seconds}s",
                    )
                except Exception as e:
                    result = TaskResult(
                        task_id=task.task_id,
                        agent_answer=None,
                        error=str(e),
                    )

                completed += 1
                if progress_callback:
                    progress_callback(completed, len(tasks))

                return result

        # Execute all tasks
        results = await asyncio.gather(
            *[run_with_semaphore(t) for t in tasks],
            return_exceptions=False,
        )

        # Compute aggregate metrics
        task_results = list(results)
        aggregate_metrics = self._compute_aggregate_metrics(task_results)

        return BenchmarkResult(
            benchmark_name=self.name,
            run_id=trace_logger.run_id,
            task_results=task_results,
            aggregate_metrics=aggregate_metrics,
            config=self.config.to_dict(),
            timestamp=datetime.utcnow().isoformat(),
        )

    @abstractmethod
    def _compute_aggregate_metrics(
        self,
        results: list[TaskResult],
    ) -> dict[str, float]:
        """
        Compute benchmark-specific aggregate metrics.

        Args:
            results: List of task results

        Returns:
            Dictionary of metric names to values
        """
        pass


class BenchmarkError(Exception):
    """Base exception for benchmark errors."""

    pass


class BenchmarkDataError(BenchmarkError):
    """Raised when there's an error loading benchmark data."""

    pass


class BenchmarkExecutionError(BenchmarkError):
    """Raised when there's an error executing a task."""

    def __init__(self, message: str, task_id: str) -> None:
        super().__init__(f"Task {task_id}: {message}")
        self.task_id = task_id


# Registry for benchmark implementations
_BENCHMARK_REGISTRY: dict[str, type[Benchmark]] = {}


def register_benchmark(name: str) -> Callable[[type[Benchmark]], type[Benchmark]]:
    """
    Decorator to register a benchmark implementation.

    Args:
        name: Name to register the benchmark under

    Returns:
        Decorator function
    """

    def decorator(cls: type[Benchmark]) -> type[Benchmark]:
        _BENCHMARK_REGISTRY[name] = cls
        return cls

    return decorator


def get_benchmark(name: str, config: BenchmarkConfig | dict[str, Any]) -> Benchmark:
    """
    Get a benchmark instance by name.

    Args:
        name: Registered benchmark name
        config: BenchmarkConfig or dict of config values

    Returns:
        Benchmark instance

    Raises:
        ValueError: If benchmark name is not registered
    """
    if name not in _BENCHMARK_REGISTRY:
        available = ", ".join(_BENCHMARK_REGISTRY.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")

    benchmark_cls = _BENCHMARK_REGISTRY[name]

    if isinstance(config, dict):
        config = BenchmarkConfig(**config)

    return benchmark_cls(config)


def list_benchmarks() -> list[str]:
    """
    List all registered benchmark names.

    Returns:
        List of benchmark names
    """
    return list(_BENCHMARK_REGISTRY.keys())

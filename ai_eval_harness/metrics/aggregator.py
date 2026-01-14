"""
Metrics aggregation for evaluation harness.

This module provides comprehensive metrics computation including
Pass@k, policy violation rates, efficiency metrics, and more.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.types import BenchmarkResult, TaskResult

logger = logging.getLogger(__name__)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute Pass@k metric using the unbiased estimator.

    This computes the probability that at least one of k random samples
    from n total samples (with c correct) is correct.

    Formula: 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to draw

    Returns:
        Pass@k probability (0.0 to 1.0)
    """
    if n - c < k:
        return 1.0

    # Use log to avoid overflow with large numbers
    # C(n-c, k) / C(n, k) = prod_{i=0}^{k-1} (n-c-i) / (n-i)
    log_prob = 0.0
    for i in range(k):
        log_prob += math.log(n - c - i) - math.log(n - i)

    return 1.0 - math.exp(log_prob)


def compute_pass_power_k(n: int, c: int, k: int) -> float:
    """
    Compute Pass^k metric (all k trials succeed).

    This measures reliability - the probability that ALL of k random
    samples from n total samples (with c correct) are correct.

    Formula: C(c, k) / C(n, k)

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to draw

    Returns:
        Pass^k probability (0.0 to 1.0)
    """
    if c < k:
        return 0.0
    if n == 0:
        return 0.0

    # C(c, k) / C(n, k) = prod_{i=0}^{k-1} (c-i) / (n-i)
    log_prob = 0.0
    for i in range(k):
        log_prob += math.log(c - i) - math.log(n - i)

    return math.exp(log_prob)


@dataclass
class AggregatedMetrics:
    """Container for aggregated metrics across a benchmark run."""

    # Core success metrics
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    success_rate: float = 0.0

    # Pass@k metrics (for multiple trials)
    pass_at_1: float = 0.0
    pass_at_2: float | None = None
    pass_at_4: float | None = None
    pass_power_1: float = 0.0
    pass_power_2: float | None = None
    pass_power_4: float | None = None

    # Efficiency metrics
    avg_turns: float = 0.0
    avg_tool_calls: float = 0.0
    total_tokens: int = 0
    avg_tokens_per_task: float = 0.0

    # Policy compliance (τ-bench/τ²-bench)
    total_policy_violations: int = 0
    policy_violation_rate: float = 0.0
    recovery_rate: float = 0.0

    # τ²-bench specific
    avg_user_tool_calls: float = 0.0
    dual_control_tasks: int = 0

    # Per-category breakdowns
    metrics_by_level: dict[str, dict[str, float]] = field(default_factory=dict)
    metrics_by_domain: dict[str, dict[str, float]] = field(default_factory=dict)

    # Timing
    total_duration_seconds: float = 0.0
    avg_task_duration_seconds: float = 0.0

    # Raw data for further analysis
    task_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.success_rate,
            "pass_at_1": self.pass_at_1,
            "pass_power_1": self.pass_power_1,
            "avg_turns": self.avg_turns,
            "avg_tool_calls": self.avg_tool_calls,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_task": self.avg_tokens_per_task,
            "total_policy_violations": self.total_policy_violations,
            "policy_violation_rate": self.policy_violation_rate,
            "recovery_rate": self.recovery_rate,
            "total_duration_seconds": self.total_duration_seconds,
            "avg_task_duration_seconds": self.avg_task_duration_seconds,
        }

        # Add optional pass@k metrics
        if self.pass_at_2 is not None:
            result["pass_at_2"] = self.pass_at_2
        if self.pass_at_4 is not None:
            result["pass_at_4"] = self.pass_at_4
        if self.pass_power_2 is not None:
            result["pass_power_2"] = self.pass_power_2
        if self.pass_power_4 is not None:
            result["pass_power_4"] = self.pass_power_4

        # Add τ²-bench metrics if present
        if self.avg_user_tool_calls > 0 or self.dual_control_tasks > 0:
            result["avg_user_tool_calls"] = self.avg_user_tool_calls
            result["dual_control_tasks"] = self.dual_control_tasks

        # Add breakdowns
        if self.metrics_by_level:
            result["metrics_by_level"] = self.metrics_by_level
        if self.metrics_by_domain:
            result["metrics_by_domain"] = self.metrics_by_domain

        return result


class MetricsAggregator:
    """
    Aggregates and computes metrics from benchmark results.

    Handles metrics computation for GAIA, τ-bench, and τ²-bench benchmarks
    with support for Pass@k computation over multiple trials.
    """

    def __init__(self) -> None:
        """Initialize the metrics aggregator."""
        self._task_results: list[TaskResult] = []
        self._trials_by_task: dict[str, list[TaskResult]] = defaultdict(list)
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

    def add_result(self, result: TaskResult) -> None:
        """Add a task result for aggregation."""
        self._task_results.append(result)
        self._trials_by_task[result.task_id].append(result)

    def add_results(self, results: list[TaskResult]) -> None:
        """Add multiple task results."""
        for result in results:
            self.add_result(result)

    def set_timing(self, start_time: datetime, end_time: datetime) -> None:
        """Set the timing information for the benchmark run."""
        self._start_time = start_time
        self._end_time = end_time

    def compute_metrics(self, benchmark_type: str = "gaia") -> AggregatedMetrics:
        """
        Compute aggregated metrics from all results.

        Args:
            benchmark_type: Type of benchmark ("gaia", "tau_bench", "tau2_bench")

        Returns:
            AggregatedMetrics with computed values
        """
        metrics = AggregatedMetrics()

        if not self._task_results:
            return metrics

        # Basic counts
        metrics.total_tasks = len(set(r.task_id for r in self._task_results))
        metrics.successful_tasks = sum(
            1 for task_id in self._trials_by_task
            if any(r.is_correct for r in self._trials_by_task[task_id])
        )
        metrics.failed_tasks = metrics.total_tasks - metrics.successful_tasks
        metrics.success_rate = (
            metrics.successful_tasks / metrics.total_tasks
            if metrics.total_tasks > 0 else 0.0
        )

        # Pass@k computation (when multiple trials per task)
        metrics.pass_at_1 = self._compute_pass_at_k(k=1)

        # Check if we have multiple trials
        trials_per_task = [len(trials) for trials in self._trials_by_task.values()]
        max_trials = max(trials_per_task) if trials_per_task else 1

        if max_trials >= 2:
            metrics.pass_at_2 = self._compute_pass_at_k(k=2)
            metrics.pass_power_2 = self._compute_pass_power_k(k=2)
        if max_trials >= 4:
            metrics.pass_at_4 = self._compute_pass_at_k(k=4)
            metrics.pass_power_4 = self._compute_pass_power_k(k=4)

        metrics.pass_power_1 = self._compute_pass_power_k(k=1)

        # Efficiency metrics
        turns = [r.metrics.get("turns", 0) for r in self._task_results]
        tool_calls = [r.metrics.get("tool_calls", 0) for r in self._task_results]
        tokens = [r.metrics.get("total_tokens", 0) for r in self._task_results]

        metrics.avg_turns = sum(turns) / len(turns) if turns else 0.0
        metrics.avg_tool_calls = sum(tool_calls) / len(tool_calls) if tool_calls else 0.0
        metrics.total_tokens = sum(tokens)
        metrics.avg_tokens_per_task = (
            metrics.total_tokens / metrics.total_tasks
            if metrics.total_tasks > 0 else 0.0
        )

        # Policy metrics (τ-bench/τ²-bench)
        if benchmark_type in ("tau_bench", "tau2_bench"):
            metrics = self._compute_policy_metrics(metrics)

        # τ²-bench specific metrics
        if benchmark_type == "tau2_bench":
            metrics = self._compute_tau2_metrics(metrics)

        # Per-level metrics (GAIA)
        if benchmark_type == "gaia":
            metrics.metrics_by_level = self._compute_level_metrics()

        # Per-domain metrics (τ-bench/τ²-bench)
        if benchmark_type in ("tau_bench", "tau2_bench"):
            metrics.metrics_by_domain = self._compute_domain_metrics()

        # Timing
        if self._start_time and self._end_time:
            metrics.total_duration_seconds = (
                self._end_time - self._start_time
            ).total_seconds()
            metrics.avg_task_duration_seconds = (
                metrics.total_duration_seconds / metrics.total_tasks
                if metrics.total_tasks > 0 else 0.0
            )

        # Store raw results for further analysis
        metrics.task_results = [
            {
                "task_id": r.task_id,
                "is_correct": r.is_correct,
                "score": r.score,
                "metrics": r.metrics,
            }
            for r in self._task_results
        ]

        return metrics

    def _compute_pass_at_k(self, k: int) -> float:
        """Compute Pass@k across all tasks."""
        if not self._trials_by_task:
            return 0.0

        pass_at_k_values = []
        for task_id, trials in self._trials_by_task.items():
            n = len(trials)
            c = sum(1 for t in trials if t.is_correct)
            if n >= k:
                pass_at_k_values.append(compute_pass_at_k(n, c, k))

        return sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0.0

    def _compute_pass_power_k(self, k: int) -> float:
        """Compute Pass^k (all k succeed) across all tasks."""
        if not self._trials_by_task:
            return 0.0

        pass_power_k_values = []
        for task_id, trials in self._trials_by_task.items():
            n = len(trials)
            c = sum(1 for t in trials if t.is_correct)
            if n >= k:
                pass_power_k_values.append(compute_pass_power_k(n, c, k))

        return (
            sum(pass_power_k_values) / len(pass_power_k_values)
            if pass_power_k_values else 0.0
        )

    def _compute_policy_metrics(self, metrics: AggregatedMetrics) -> AggregatedMetrics:
        """Compute policy violation metrics for τ-bench/τ²-bench."""
        violations = [
            r.metrics.get("policy_violations", 0) for r in self._task_results
        ]
        recovery_attempts = [
            r.metrics.get("recovery_attempts", 0) for r in self._task_results
        ]

        metrics.total_policy_violations = sum(violations)
        metrics.policy_violation_rate = (
            metrics.total_policy_violations / metrics.total_tasks
            if metrics.total_tasks > 0 else 0.0
        )

        # Recovery rate: among tasks with violations, how many recovered
        tasks_with_violations = [
            (v, r) for v, r in zip(violations, recovery_attempts) if v > 0
        ]
        if tasks_with_violations:
            recovered = sum(1 for v, r in tasks_with_violations if r > 0)
            metrics.recovery_rate = recovered / len(tasks_with_violations)

        return metrics

    def _compute_tau2_metrics(self, metrics: AggregatedMetrics) -> AggregatedMetrics:
        """Compute τ²-bench specific metrics."""
        user_tool_calls = [
            r.metrics.get("user_tool_calls", 0) for r in self._task_results
        ]

        metrics.avg_user_tool_calls = (
            sum(user_tool_calls) / len(user_tool_calls)
            if user_tool_calls else 0.0
        )

        # Count tasks with user tool calls (dual-control)
        metrics.dual_control_tasks = sum(1 for u in user_tool_calls if u > 0)

        return metrics

    def _compute_level_metrics(self) -> dict[str, dict[str, float]]:
        """Compute per-level metrics for GAIA."""
        level_results: dict[int, list[TaskResult]] = defaultdict(list)

        for result in self._task_results:
            level = result.metrics.get("level", 0)
            level_results[level].append(result)

        metrics_by_level = {}
        for level, results in sorted(level_results.items()):
            if not results:
                continue

            correct = sum(1 for r in results if r.is_correct)
            total = len(results)

            metrics_by_level[f"level_{level}"] = {
                "total": float(total),
                "correct": float(correct),
                "accuracy": correct / total if total > 0 else 0.0,
            }

        return metrics_by_level

    def _compute_domain_metrics(self) -> dict[str, dict[str, float]]:
        """Compute per-domain metrics for τ-bench/τ²-bench."""
        domain_results: dict[str, list[TaskResult]] = defaultdict(list)

        for result in self._task_results:
            domain = result.metrics.get("domain", "unknown")
            domain_results[domain].append(result)

        metrics_by_domain = {}
        for domain, results in sorted(domain_results.items()):
            if not results:
                continue

            correct = sum(1 for r in results if r.is_correct)
            total = len(results)
            violations = sum(r.metrics.get("policy_violations", 0) for r in results)

            metrics_by_domain[domain] = {
                "total": float(total),
                "correct": float(correct),
                "accuracy": correct / total if total > 0 else 0.0,
                "policy_violations": float(violations),
                "violation_rate": violations / total if total > 0 else 0.0,
            }

        return metrics_by_domain

    def export_report(
        self,
        output_path: Path | str,
        benchmark_type: str = "gaia",
        format: str = "json",
    ) -> None:
        """
        Export metrics report to file.

        Args:
            output_path: Path to write the report
            benchmark_type: Type of benchmark
            format: Output format ("json" or "markdown")
        """
        output_path = Path(output_path)
        metrics = self.compute_metrics(benchmark_type)

        if format == "json":
            output_path.write_text(
                json.dumps(metrics.to_dict(), indent=2, default=str)
            )
        elif format == "markdown":
            report = self._generate_markdown_report(metrics, benchmark_type)
            output_path.write_text(report)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Exported metrics report to {output_path}")

    def _generate_markdown_report(
        self,
        metrics: AggregatedMetrics,
        benchmark_type: str,
    ) -> str:
        """Generate a markdown report from metrics."""
        lines = [
            f"# Evaluation Report - {benchmark_type.upper()}",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"- **Total Tasks**: {metrics.total_tasks}",
            f"- **Successful**: {metrics.successful_tasks}",
            f"- **Failed**: {metrics.failed_tasks}",
            f"- **Success Rate**: {metrics.success_rate:.2%}",
            "",
        ]

        # Pass@k metrics
        lines.extend([
            "## Pass@k Metrics",
            "",
            f"- **Pass@1**: {metrics.pass_at_1:.2%}",
        ])
        if metrics.pass_at_2 is not None:
            lines.append(f"- **Pass@2**: {metrics.pass_at_2:.2%}")
        if metrics.pass_at_4 is not None:
            lines.append(f"- **Pass@4**: {metrics.pass_at_4:.2%}")

        lines.extend([
            "",
            "## Pass^k (Reliability) Metrics",
            "",
            f"- **Pass^1**: {metrics.pass_power_1:.2%}",
        ])
        if metrics.pass_power_2 is not None:
            lines.append(f"- **Pass^2**: {metrics.pass_power_2:.2%}")
        if metrics.pass_power_4 is not None:
            lines.append(f"- **Pass^4**: {metrics.pass_power_4:.2%}")

        # Efficiency
        lines.extend([
            "",
            "## Efficiency",
            "",
            f"- **Average Turns**: {metrics.avg_turns:.2f}",
            f"- **Average Tool Calls**: {metrics.avg_tool_calls:.2f}",
            f"- **Total Tokens**: {metrics.total_tokens:,}",
            f"- **Avg Tokens/Task**: {metrics.avg_tokens_per_task:.0f}",
            "",
        ])

        # Policy metrics
        if benchmark_type in ("tau_bench", "tau2_bench"):
            lines.extend([
                "## Policy Compliance",
                "",
                f"- **Total Violations**: {metrics.total_policy_violations}",
                f"- **Violation Rate**: {metrics.policy_violation_rate:.2f} per task",
                f"- **Recovery Rate**: {metrics.recovery_rate:.2%}",
                "",
            ])

        # τ²-bench specific
        if benchmark_type == "tau2_bench":
            lines.extend([
                "## Dual-Control Metrics",
                "",
                f"- **Avg User Tool Calls**: {metrics.avg_user_tool_calls:.2f}",
                f"- **Dual-Control Tasks**: {metrics.dual_control_tasks}",
                "",
            ])

        # Timing
        if metrics.total_duration_seconds > 0:
            lines.extend([
                "## Timing",
                "",
                f"- **Total Duration**: {metrics.total_duration_seconds:.1f}s",
                f"- **Avg per Task**: {metrics.avg_task_duration_seconds:.1f}s",
                "",
            ])

        # Per-level breakdown (GAIA)
        if metrics.metrics_by_level:
            lines.extend([
                "## Results by Level",
                "",
                "| Level | Total | Correct | Accuracy |",
                "|-------|-------|---------|----------|",
            ])
            for level, data in sorted(metrics.metrics_by_level.items()):
                lines.append(
                    f"| {level} | {int(data['total'])} | {int(data['correct'])} | "
                    f"{data['accuracy']:.2%} |"
                )
            lines.append("")

        # Per-domain breakdown
        if metrics.metrics_by_domain:
            lines.extend([
                "## Results by Domain",
                "",
                "| Domain | Total | Correct | Accuracy | Violations |",
                "|--------|-------|---------|----------|------------|",
            ])
            for domain, data in sorted(metrics.metrics_by_domain.items()):
                lines.append(
                    f"| {domain} | {int(data['total'])} | {int(data['correct'])} | "
                    f"{data['accuracy']:.2%} | {int(data.get('policy_violations', 0))} |"
                )
            lines.append("")

        return "\n".join(lines)


def create_benchmark_result(
    benchmark_name: str,
    task_results: list[TaskResult],
    config: dict[str, Any],
    start_time: datetime,
    end_time: datetime,
) -> BenchmarkResult:
    """
    Create a BenchmarkResult from task results.

    Args:
        benchmark_name: Name of the benchmark
        task_results: List of TaskResult objects
        config: Configuration used for the run
        start_time: When the benchmark started
        end_time: When the benchmark ended

    Returns:
        BenchmarkResult with computed metrics
    """
    # Determine benchmark type
    if "gaia" in benchmark_name.lower():
        benchmark_type = "gaia"
    elif "tau2" in benchmark_name.lower():
        benchmark_type = "tau2_bench"
    elif "tau" in benchmark_name.lower():
        benchmark_type = "tau_bench"
    else:
        benchmark_type = "gaia"

    # Compute metrics
    aggregator = MetricsAggregator()
    aggregator.add_results(task_results)
    aggregator.set_timing(start_time, end_time)
    metrics = aggregator.compute_metrics(benchmark_type)

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        total_tasks=metrics.total_tasks,
        successful_tasks=metrics.successful_tasks,
        failed_tasks=metrics.failed_tasks,
        task_results=task_results,
        aggregate_metrics=metrics.to_dict(),
        metadata={
            "config": config,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": metrics.total_duration_seconds,
        },
    )

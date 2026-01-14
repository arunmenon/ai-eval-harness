#!/usr/bin/env python3
"""
CLI entry point for the AI evaluation harness.

This module provides a Click-based CLI for running benchmarks
against various AI agents.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_eval_harness.core.config import RunConfig, load_config
from ai_eval_harness.core.benchmark import get_benchmark
from ai_eval_harness.agents.base import get_agent
from ai_eval_harness.trace.logger import TraceLogger
from ai_eval_harness.metrics.aggregator import MetricsAggregator, create_benchmark_result

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="ai-eval")
def cli() -> None:
    """AI Assistant Evaluation Harness - Run benchmarks against AI agents."""
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Override output directory from config",
)
@click.option(
    "--max-tasks",
    "-n",
    type=int,
    default=None,
    help="Maximum number of tasks to run (for testing)",
)
@click.option(
    "--task-ids",
    "-t",
    type=str,
    default=None,
    help="Comma-separated list of specific task IDs to run",
)
@click.option(
    "--trials",
    type=int,
    default=1,
    help="Number of trials per task (for Pass@k computation)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be run without executing",
)
def run(
    config: Path,
    output_dir: Path | None,
    max_tasks: int | None,
    task_ids: str | None,
    trials: int,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Run a benchmark evaluation."""
    setup_logging(verbose)

    # Load configuration
    console.print(f"[bold blue]Loading configuration from {config}[/]")
    run_config = load_config(config)

    # Override output directory if specified
    if output_dir:
        run_config.output_dir = str(output_dir)

    # Parse task IDs if specified
    specific_tasks: list[str] | None = None
    if task_ids:
        specific_tasks = [t.strip() for t in task_ids.split(",")]

    if dry_run:
        _show_dry_run(run_config, max_tasks, specific_tasks, trials)
        return

    # Run the benchmark
    asyncio.run(_run_benchmark(
        run_config,
        max_tasks=max_tasks,
        specific_tasks=specific_tasks,
        trials=trials,
        verbose=verbose,
    ))


async def _run_benchmark(
    config: RunConfig,
    max_tasks: int | None = None,
    specific_tasks: list[str] | None = None,
    trials: int = 1,
    verbose: bool = False,
) -> None:
    """Execute the benchmark run."""
    start_time = datetime.now()

    # Initialize benchmark
    console.print(f"[bold]Initializing benchmark: {config.benchmark}[/]")
    benchmark = get_benchmark(config.benchmark, config.benchmark_config)

    # Initialize agent
    console.print(f"[bold]Initializing agent: {config.agent.provider}/{config.agent.model_name}[/]")
    agent = get_agent(
        provider=config.agent.provider,
        model_name=config.agent.model_name,
        api_key=config.agent.api_key,
        **config.agent.extra_params,
    )

    # Initialize trace logger
    output_path = Path(config.output_dir) if config.output_dir else Path("./results")
    output_path.mkdir(parents=True, exist_ok=True)

    trace_logger = TraceLogger(
        output_dir=output_path / "traces",
        save_traces=config.save_traces,
    )

    # Load tasks
    console.print("[bold]Loading tasks...[/]")
    tasks = await benchmark.load_tasks()
    console.print(f"  Loaded {len(tasks)} tasks")

    # Filter tasks if specified
    if specific_tasks:
        tasks = [t for t in tasks if t.task_id in specific_tasks]
        console.print(f"  Filtered to {len(tasks)} specific tasks")

    # Limit tasks if specified
    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]
        console.print(f"  Limited to {max_tasks} tasks")

    # Run tasks
    all_results = []
    metrics_aggregator = MetricsAggregator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_progress = progress.add_task(
            "[cyan]Running benchmark...",
            total=len(tasks) * trials,
        )

        for trial in range(trials):
            trial_label = f" (trial {trial + 1}/{trials})" if trials > 1 else ""

            for task in tasks:
                progress.update(
                    task_progress,
                    description=f"[cyan]Task {task.task_id}{trial_label}",
                )

                try:
                    result = await benchmark.run_task(task, agent, trace_logger)
                    all_results.append(result)
                    metrics_aggregator.add_result(result)

                    status = "[green]✓[/]" if result.is_correct else "[red]✗[/]"
                    if verbose:
                        console.print(f"  {status} {task.task_id}: score={result.score:.2f}")

                except Exception as e:
                    console.print(f"  [red]Error on {task.task_id}: {e}[/]")
                    if verbose:
                        console.print_exception()

                progress.advance(task_progress)

                # Reset agent state between tasks
                agent.reset()

    end_time = datetime.now()
    metrics_aggregator.set_timing(start_time, end_time)

    # Determine benchmark type for metrics
    benchmark_type = "gaia"
    if "tau2" in config.benchmark.lower():
        benchmark_type = "tau2_bench"
    elif "tau" in config.benchmark.lower():
        benchmark_type = "tau_bench"

    # Compute and display results
    metrics = metrics_aggregator.compute_metrics(benchmark_type)
    _display_results(metrics, benchmark_type)

    # Save results
    result = create_benchmark_result(
        benchmark_name=config.benchmark,
        task_results=all_results,
        config=config.to_dict() if hasattr(config, 'to_dict') else {},
        start_time=start_time,
        end_time=end_time,
    )

    # Export reports
    results_file = output_path / f"results_{config.benchmark}_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "benchmark": result.benchmark_name,
            "total_tasks": result.total_tasks,
            "successful_tasks": result.successful_tasks,
            "failed_tasks": result.failed_tasks,
            "aggregate_metrics": result.aggregate_metrics,
            "metadata": result.metadata,
            "task_results": [
                {
                    "task_id": tr.task_id,
                    "is_correct": tr.is_correct,
                    "score": tr.score,
                    "metrics": tr.metrics,
                }
                for tr in result.task_results
            ],
        }, f, indent=2, default=str)

    console.print(f"\n[bold green]Results saved to {results_file}[/]")

    # Export markdown report
    markdown_file = output_path / f"report_{config.benchmark}_{start_time.strftime('%Y%m%d_%H%M%S')}.md"
    metrics_aggregator.export_report(markdown_file, benchmark_type, format="markdown")
    console.print(f"[bold green]Report saved to {markdown_file}[/]")


def _display_results(metrics: Any, benchmark_type: str) -> None:
    """Display results in a formatted table."""
    console.print("\n[bold]═══ Results ═══[/]\n")

    # Summary table
    summary_table = Table(title="Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Total Tasks", str(metrics.total_tasks))
    summary_table.add_row("Successful", f"[green]{metrics.successful_tasks}[/]")
    summary_table.add_row("Failed", f"[red]{metrics.failed_tasks}[/]")
    summary_table.add_row("Success Rate", f"{metrics.success_rate:.2%}")
    summary_table.add_row("Pass@1", f"{metrics.pass_at_1:.2%}")

    if metrics.pass_at_2 is not None:
        summary_table.add_row("Pass@2", f"{metrics.pass_at_2:.2%}")
    if metrics.pass_at_4 is not None:
        summary_table.add_row("Pass@4", f"{metrics.pass_at_4:.2%}")

    console.print(summary_table)

    # Efficiency table
    efficiency_table = Table(title="Efficiency", show_header=True, header_style="bold magenta")
    efficiency_table.add_column("Metric", style="cyan")
    efficiency_table.add_column("Value", justify="right")

    efficiency_table.add_row("Avg Turns", f"{metrics.avg_turns:.2f}")
    efficiency_table.add_row("Avg Tool Calls", f"{metrics.avg_tool_calls:.2f}")
    efficiency_table.add_row("Total Tokens", f"{metrics.total_tokens:,}")

    console.print(efficiency_table)

    # Policy metrics for τ-bench
    if benchmark_type in ("tau_bench", "tau2_bench"):
        policy_table = Table(title="Policy Compliance", show_header=True, header_style="bold magenta")
        policy_table.add_column("Metric", style="cyan")
        policy_table.add_column("Value", justify="right")

        policy_table.add_row("Total Violations", str(metrics.total_policy_violations))
        policy_table.add_row("Violation Rate", f"{metrics.policy_violation_rate:.2f}/task")
        policy_table.add_row("Recovery Rate", f"{metrics.recovery_rate:.2%}")

        if benchmark_type == "tau2_bench":
            policy_table.add_row("Avg User Tool Calls", f"{metrics.avg_user_tool_calls:.2f}")

        console.print(policy_table)

    # Per-level breakdown for GAIA
    if metrics.metrics_by_level:
        level_table = Table(title="Results by Level", show_header=True, header_style="bold magenta")
        level_table.add_column("Level", style="cyan")
        level_table.add_column("Total", justify="right")
        level_table.add_column("Correct", justify="right")
        level_table.add_column("Accuracy", justify="right")

        for level, data in sorted(metrics.metrics_by_level.items()):
            level_table.add_row(
                level,
                str(int(data["total"])),
                str(int(data["correct"])),
                f"{data['accuracy']:.2%}",
            )

        console.print(level_table)

    # Per-domain breakdown
    if metrics.metrics_by_domain:
        domain_table = Table(title="Results by Domain", show_header=True, header_style="bold magenta")
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("Total", justify="right")
        domain_table.add_column("Correct", justify="right")
        domain_table.add_column("Accuracy", justify="right")
        domain_table.add_column("Violations", justify="right")

        for domain, data in sorted(metrics.metrics_by_domain.items()):
            domain_table.add_row(
                domain,
                str(int(data["total"])),
                str(int(data["correct"])),
                f"{data['accuracy']:.2%}",
                str(int(data.get("policy_violations", 0))),
            )

        console.print(domain_table)


def _show_dry_run(
    config: RunConfig,
    max_tasks: int | None,
    specific_tasks: list[str] | None,
    trials: int,
) -> None:
    """Show what would be run without executing."""
    console.print("\n[bold yellow]═══ Dry Run ═══[/]\n")

    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Benchmark", config.benchmark)
    table.add_row("Agent Provider", config.agent.provider)
    table.add_row("Agent Model", config.agent.model_name)
    table.add_row("Max Turns", str(config.benchmark_config.max_turns))
    table.add_row("Output Dir", config.output_dir or "./results")
    table.add_row("Save Traces", str(config.save_traces))
    table.add_row("Trials", str(trials))

    if max_tasks:
        table.add_row("Max Tasks", str(max_tasks))
    if specific_tasks:
        table.add_row("Specific Tasks", ", ".join(specific_tasks))

    console.print(table)
    console.print("\n[yellow]Run without --dry-run to execute.[/]")


@cli.command()
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic"]),
    required=True,
    help="Agent provider",
)
@click.option(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Model name",
)
@click.option(
    "--prompt",
    type=str,
    default="Hello! What is 2 + 2?",
    help="Test prompt",
)
def test_agent(provider: str, model: str, prompt: str) -> None:
    """Test agent configuration with a simple prompt."""
    import os

    setup_logging(verbose=True)

    # Get API key from environment
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY not set[/]")
            return
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            console.print("[red]Error: ANTHROPIC_API_KEY not set[/]")
            return

    console.print(f"[bold]Testing {provider}/{model}...[/]\n")

    agent = get_agent(provider=provider, model_name=model, api_key=api_key)

    from ai_eval_harness.core.types import Message

    messages = [Message.user(prompt)]

    async def _test() -> None:
        response = await agent.generate(messages)
        console.print(f"[green]Response:[/] {response.content}")
        if response.usage:
            console.print(f"[dim]Tokens: {response.usage}[/]")

    asyncio.run(_test())


@cli.command()
def list_benchmarks() -> None:
    """List available benchmarks."""
    from ai_eval_harness.core.benchmark import BENCHMARK_REGISTRY

    console.print("\n[bold]Available Benchmarks:[/]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    benchmark_info = {
        "gaia": "General AI Assistants - Real-world tasks with tools",
        "tau_bench": "Tool-Agent-User dialogues with policy compliance",
        "tau2_bench": "Dual-control environments with user tools",
    }

    for name in sorted(BENCHMARK_REGISTRY.keys()):
        desc = benchmark_info.get(name, "No description available")
        table.add_row(name, desc)

    console.print(table)


@cli.command()
def list_agents() -> None:
    """List available agent providers."""
    from ai_eval_harness.agents.base import AGENT_REGISTRY

    console.print("\n[bold]Available Agent Providers:[/]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Models")

    agent_models = {
        "openai": "gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.",
        "anthropic": "claude-3-opus, claude-3-sonnet, claude-3-haiku, etc.",
    }

    for name in sorted(AGENT_REGISTRY.keys()):
        models = agent_models.get(name, "Various")
        table.add_row(name, models)

    console.print(table)


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
def replay_trace(trace_file: Path) -> None:
    """Replay and display a trace file."""
    console.print(f"\n[bold]Replaying trace: {trace_file}[/]\n")

    with open(trace_file) as f:
        trace_data = json.load(f)

    # Display trace info
    console.print(f"[bold]Task ID:[/] {trace_data.get('task_id', 'unknown')}")
    console.print(f"[bold]Status:[/] {'[green]Success[/]' if trace_data.get('success') else '[red]Failed[/]'}")
    console.print(f"[bold]Events:[/] {len(trace_data.get('events', []))}")
    console.print()

    # Display events
    for i, event in enumerate(trace_data.get("events", [])):
        event_type = event.get("type", "unknown")
        timestamp = event.get("timestamp", "")

        if event_type == "agent_response":
            console.print(f"[{i}] [bold blue]Agent:[/] {event.get('content', '')[:200]}...")
        elif event_type == "tool_call":
            console.print(f"[{i}] [bold yellow]Tool:[/] {event.get('tool_name', '')}({event.get('arguments', {})})")
        elif event_type == "tool_result":
            result = str(event.get("output", ""))[:100]
            console.print(f"[{i}] [bold green]Result:[/] {result}...")
        elif event_type == "user_message":
            console.print(f"[{i}] [bold magenta]User:[/] {event.get('content', '')[:200]}...")
        elif event_type == "policy_violation":
            console.print(f"[{i}] [bold red]Violation:[/] {event.get('description', '')}")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()

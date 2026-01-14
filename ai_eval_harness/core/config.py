"""
Configuration management for the AI evaluation harness.

This module provides YAML-based configuration with environment variable
substitution and validation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .agent import AgentConfig
from .benchmark import BenchmarkConfig


@dataclass
class RunConfig:
    """
    Complete configuration for a benchmark run.

    Attributes:
        run_id: Unique identifier for this run (auto-generated if not provided)
        experiment_name: Human-readable name for the experiment
        agent: Agent configuration
        benchmark: Benchmark name to run
        benchmark_config: Benchmark-specific configuration
        max_concurrency: Number of concurrent tasks
        timeout_per_task: Timeout for each task in seconds
        output_dir: Directory for output files
        trace_dir: Directory for trace files (defaults to output_dir/traces)
        save_traces: Whether to save execution traces
        save_raw_responses: Whether to include raw API responses in traces
        seed: Random seed for reproducibility
    """

    run_id: str | None = None
    experiment_name: str = "default"

    # Agent configuration
    agent: AgentConfig = field(default_factory=lambda: AgentConfig(
        model_name="gpt-4o",
        provider="openai",
    ))

    # Benchmark configuration
    benchmark: str = "gaia"
    benchmark_config: dict[str, Any] = field(default_factory=dict)

    # Execution settings
    max_concurrency: int = 1
    timeout_per_task: float = 300.0

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./results"))
    trace_dir: Path | None = None
    save_traces: bool = True
    save_raw_responses: bool = False

    # Reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        """Generate run_id if not provided."""
        if self.run_id is None:
            self.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Convert paths
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.trace_dir, str):
            self.trace_dir = Path(self.trace_dir)

        # Default trace_dir
        if self.trace_dir is None:
            self.trace_dir = self.output_dir / "traces"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RunConfig":
        """
        Load configuration from a YAML file.

        Supports environment variable substitution using ${VAR} syntax.

        Args:
            path: Path to the YAML configuration file

        Returns:
            RunConfig instance
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        # Substitute environment variables
        data = cls._substitute_env_vars(data)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        """
        Create configuration from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            RunConfig instance
        """
        # Extract agent config
        agent_data = data.pop("agent", {})
        agent_config = AgentConfig(
            model_name=agent_data.get("model_name", "gpt-4o"),
            provider=agent_data.get("provider", "openai"),
            api_key=agent_data.get("api_key"),
            base_url=agent_data.get("base_url"),
            parameters=agent_data.get("parameters", {}),
            timeout_seconds=agent_data.get("timeout_seconds", 120.0),
            max_retries=agent_data.get("max_retries", 3),
        )

        # Convert paths
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        if "trace_dir" in data and data["trace_dir"]:
            data["trace_dir"] = Path(data["trace_dir"])

        return cls(agent=agent_config, **data)

    @staticmethod
    def _substitute_env_vars(data: Any) -> Any:
        """
        Recursively substitute ${VAR} patterns with environment variables.

        Args:
            data: Data to process (dict, list, or scalar)

        Returns:
            Data with environment variables substituted
        """
        if isinstance(data, str):
            # Match ${VAR} or ${VAR:default}
            pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

            def replace(match: re.Match[str]) -> str:
                var_name = match.group(1)
                default = match.group(2)
                value = os.environ.get(var_name)
                if value is None:
                    if default is not None:
                        return default
                    return match.group(0)  # Keep original if no default
                return value

            return re.sub(pattern, replace, data)

        elif isinstance(data, dict):
            return {k: RunConfig._substitute_env_vars(v) for k, v in data.items()}

        elif isinstance(data, list):
            return [RunConfig._substitute_env_vars(v) for v in data]

        return data

    def to_yaml(self, path: Path | str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path to save the configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Note: Sensitive data (API keys) is excluded.

        Returns:
            Configuration dictionary
        """
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "agent": {
                "model_name": self.agent.model_name,
                "provider": self.agent.provider,
                "base_url": self.agent.base_url,
                "parameters": self.agent.parameters,
                "timeout_seconds": self.agent.timeout_seconds,
                "max_retries": self.agent.max_retries,
            },
            "benchmark": self.benchmark,
            "benchmark_config": self.benchmark_config,
            "max_concurrency": self.max_concurrency,
            "timeout_per_task": self.timeout_per_task,
            "output_dir": str(self.output_dir),
            "trace_dir": str(self.trace_dir) if self.trace_dir else None,
            "save_traces": self.save_traces,
            "save_raw_responses": self.save_raw_responses,
            "seed": self.seed,
        }

    def get_benchmark_config(self) -> BenchmarkConfig:
        """
        Create a BenchmarkConfig from the run configuration.

        Returns:
            BenchmarkConfig instance
        """
        return BenchmarkConfig(
            data_dir=Path(self.benchmark_config.get("data_dir", "."))
            if self.benchmark_config.get("data_dir")
            else None,
            split=self.benchmark_config.get("split", "validation"),
            subset=self.benchmark_config.get("subset"),
            max_tasks=self.benchmark_config.get("max_tasks"),
            shuffle=self.benchmark_config.get("shuffle", False),
            seed=self.seed,
            max_turns=self.benchmark_config.get("max_turns", 50),
            max_tool_calls_per_turn=self.benchmark_config.get("max_tool_calls_per_turn", 10),
            timeout_per_task_seconds=self.timeout_per_task,
        )


def load_config(path: Path | str | None = None, **overrides: Any) -> RunConfig:
    """
    Load configuration from file or create default, with optional overrides.

    Args:
        path: Optional path to YAML configuration file
        **overrides: Values to override in the configuration

    Returns:
        RunConfig instance
    """
    if path:
        config = RunConfig.from_yaml(path)
    else:
        config = RunConfig()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif key in config.benchmark_config or key not in config.to_dict():
            config.benchmark_config[key] = value

    return config

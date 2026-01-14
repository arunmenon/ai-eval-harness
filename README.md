# AI Eval Harness

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

A unified evaluation harness for AI assistant benchmarks, supporting **GAIA**, **τ-bench**, and **τ²-bench** with pluggable agent implementations.

## Overview

AI Eval Harness provides a standardized framework for evaluating AI assistants across multiple benchmarks. It abstracts away the complexity of dataset loading, conversation orchestration, and metrics computation, letting you focus on your agent implementation.

**Key Benefits:**
- Run the same agent across different benchmarks with a single command
- Comprehensive metrics including Pass@k, policy compliance, and efficiency
- Full execution traces for debugging and analysis
- Extensible architecture for custom agents and benchmarks

## Features

- **Three Benchmarks**: GAIA (tool use), τ-bench (policy compliance), τ²-bench (dual-control)
- **Multiple Providers**: Built-in support for OpenAI and Anthropic, plus custom agent interface
- **Rich Metrics**: Pass@k, Pass^k (reliability), policy violations, recovery rate
- **Trace Logging**: Complete execution traces with JSON export and replay
- **CLI Interface**: Rich terminal output with progress bars and formatted tables
- **Flexible Configuration**: YAML configs with environment variable substitution

## Quick Start

### Installation

```bash
git clone https://github.com/arunmenon/ai-eval-harness.git
cd ai-eval-harness
pip install -e .
```

### Environment Setup

```bash
# Set your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# For GAIA benchmark (requires HuggingFace access)
export HF_TOKEN="hf_..."
```

### Run Your First Benchmark

```bash
# Test your agent setup
ai-eval test-agent --provider openai --model gpt-4o

# Run GAIA benchmark
ai-eval run --config configs/gaia.yaml --max-tasks 5

# Run τ-bench with PayPal domain
ai-eval run --config configs/tau_bench.yaml
```

## Supported Benchmarks

### GAIA (General AI Assistants)

Real-world tasks requiring multi-step reasoning with tools like web browsing, code execution, and file handling.

| Property | Details |
|----------|---------|
| Source | [HuggingFace gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) |
| Tasks | 450+ across 3 difficulty levels |
| Tools | Web browser, Python, Bash, file reader |
| Metric | Accuracy per level |

```bash
ai-eval run --config configs/gaia.yaml
```

### τ-bench (Tool-Agent-User)

Multi-turn customer service dialogues testing policy compliance in domain-specific scenarios.

| Property | Details |
|----------|---------|
| Domains | PayPal (disputes, payments), Retail (orders, returns) |
| Format | Interactive conversations with user simulator |
| Metric | Pass@k, policy violation rate, recovery rate |

```bash
ai-eval run --config configs/tau_bench.yaml
```

### τ²-bench (Dual-Control)

Extended τ-bench where both agent AND user have tool capabilities, testing coordination in shared-state environments.

| Property | Details |
|----------|---------|
| Unique Feature | Users can execute tools (accept offers, restart devices) |
| Domains | PayPal with buyer actions, Retail |
| Metric | Pass@k + user tool efficiency |

```bash
ai-eval run --config configs/tau2_bench.yaml
```

## Project Structure

```
ai_eval_harness/
├── core/                    # Core abstractions
│   ├── agent.py            # Agent interface (ABC)
│   ├── benchmark.py        # Benchmark interface (ABC)
│   ├── config.py           # Configuration management
│   └── types.py            # Shared types (Message, ToolCall, etc.)
│
├── agents/                  # Built-in agents
│   ├── base.py             # Base agent with retry logic
│   ├── openai_agent.py     # OpenAI GPT models
│   └── anthropic_agent.py  # Anthropic Claude models
│
├── benchmarks/
│   ├── gaia/               # GAIA benchmark
│   │   ├── dataloader.py   # HuggingFace loader
│   │   ├── scorer.py       # Answer validation
│   │   └── tools.py        # Web, Python, Bash tools
│   │
│   ├── tau_bench/          # τ-bench benchmark
│   │   ├── domains/        # PayPal, Retail domains
│   │   └── user_simulator.py
│   │
│   └── tau2_bench/         # τ²-bench (dual-control)
│
├── metrics/                 # Metrics computation
│   └── aggregator.py       # Pass@k, policy metrics
│
└── trace/                   # Execution tracing
    └── logger.py           # JSON trace export

configs/                     # YAML configurations
scripts/                     # CLI entry point
examples/                    # Custom agent examples
```

## Configuration

Configurations use YAML format with environment variable substitution (`${VAR}` or `${VAR:default}`).

```yaml
# configs/gaia.yaml
benchmark: gaia

agent:
  provider: openai
  model_name: gpt-4o
  api_key: ${OPENAI_API_KEY}
  temperature: 0.0
  max_tokens: 4096

benchmark_config:
  split: validation
  subset: "2023_level1"  # 2023_all, 2023_level1, 2023_level2, 2023_level3
  max_turns: 50

output_dir: ./results/gaia
save_traces: true
```

See [`configs/`](configs/) for complete examples.

## CLI Reference

### Run Benchmark

```bash
ai-eval run --config <config.yaml> [OPTIONS]

Options:
  -c, --config PATH      Configuration file (required)
  -o, --output-dir PATH  Override output directory
  -n, --max-tasks INT    Limit number of tasks
  -t, --task-ids TEXT    Comma-separated task IDs
  --trials INT           Trials per task (for Pass@k)
  -v, --verbose          Verbose output
  --dry-run              Show config without running
```

### Other Commands

```bash
# Test agent connectivity
ai-eval test-agent --provider openai --model gpt-4o --prompt "Hello!"

# List available benchmarks
ai-eval list-benchmarks

# List available agent providers
ai-eval list-agents

# Replay a trace file
ai-eval replay-trace results/traces/task_001.json
```

## Custom Agents

Implement the `Agent` interface to integrate any LLM:

```python
from ai_eval_harness.core.agent import Agent, AgentConfig
from ai_eval_harness.core.types import AgentResponse, Message, ToolDefinition
from ai_eval_harness.agents.base import register_agent

@register_agent("my_agent")
class MyCustomAgent(Agent):
    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        # Initialize your LLM client

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AgentResponse:
        # Call your LLM and return response
        return AgentResponse(
            content="response text",
            tool_calls=[...],  # Optional tool calls
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

    def reset(self) -> None:
        # Clear state between tasks
        pass
```

See [`examples/custom_agent.py`](examples/custom_agent.py) for complete examples including Chain-of-Thought and ReAct patterns.

## Metrics

### By Benchmark Type

| Metric | GAIA | τ-bench | τ²-bench |
|--------|:----:|:-------:|:--------:|
| Accuracy | ✓ | ✓ | ✓ |
| Pass@k | ✓ | ✓ | ✓ |
| Pass^k (reliability) | ✓ | ✓ | ✓ |
| Policy violations | - | ✓ | ✓ |
| Recovery rate | - | ✓ | ✓ |
| User tool calls | - | - | ✓ |
| Per-level breakdown | ✓ | - | - |
| Per-domain breakdown | - | ✓ | ✓ |

### Pass@k Explained

- **Pass@k**: Probability that at least 1 of k trials succeeds
- **Pass^k**: Probability that ALL k trials succeed (measures reliability)

```
Pass@k = 1 - C(n-c, k) / C(n, k)
Pass^k = C(c, k) / C(n, k)

where n = total trials, c = correct trials, k = sample size
```

Why Pass^k matters: An agent with 60% Pass@1 but 25% Pass^4 has reliability issues.

## Domain Tools

### PayPal Domain (τ-bench / τ²-bench)

**Agent Tools:**
| Tool | Description |
|------|-------------|
| `list_disputes` | Query disputes by status |
| `get_dispute_details` | Get full dispute info |
| `provide_evidence` | Submit tracking/refund proof |
| `accept_claim` | Accept liability, auto-refund |
| `make_offer` | Propose settlement |
| `get_transaction` | Retrieve transaction details |
| `refund_payment` | Issue full/partial refund |
| `get_balance` | Check account balance |

**User Tools (τ²-bench only):**
| Tool | Description |
|------|-------------|
| `respond_to_offer` | Accept/reject settlement |
| `provide_buyer_evidence` | Upload photos, receipts |

**Policy Constraints:**
- 20-day dispute response window
- 180-day refund limit
- Confirmation required for all modifying actions
- High-value transactions require verified merchant

### Retail Domain

**Tools:** `search_products`, `get_order`, `create_order`, `cancel_order`, `process_return`, `check_inventory`, `get_customer_info`

**Policy:** Explicit user confirmation required before database modifications.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/arunmenon/ai-eval-harness.git
cd ai-eval-harness
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This harness builds on the following benchmark research:

- **GAIA**: [GAIA: A Benchmark for General AI Assistants](https://huggingface.co/datasets/gaia-benchmark/GAIA)
- **τ-bench**: [τ-bench: A Benchmark for Tool-Agent-User Interaction](https://github.com/sierra-research/tau-bench)
- **τ²-bench**: [τ²-bench: Benchmarking Agents in Dual-Control Environments](https://github.com/sierra-research/tau2-bench)

---

Built with the goal of making AI assistant evaluation accessible and reproducible.

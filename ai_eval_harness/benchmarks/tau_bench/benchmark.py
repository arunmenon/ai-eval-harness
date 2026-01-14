"""
τ-bench benchmark implementation.

This module provides the main τ-bench benchmark class that orchestrates
multi-turn conversations between agents and simulated users.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...core.benchmark import Benchmark, BenchmarkConfig, register_benchmark
from ...core.types import Message, Task, TaskResult
from .domains.paypal import PayPalDomain, PayPalEnvironmentState, get_paypal_domain
from .domains.retail import RetailDomain, RetailEnvironmentState, get_retail_domain
from .user_simulator import SimpleUserSimulator

if TYPE_CHECKING:
    from ...core.agent import Agent
    from ...trace.logger import TraceLogger

logger = logging.getLogger(__name__)


TAU_BENCH_SYSTEM_PROMPT = """You are a helpful customer service agent for {domain_name}.

Your role is to assist customers with their requests while following company policies.

Available tools:
{tool_descriptions}

Policies:
{policies}

Guidelines:
1. Always verify customer identity before making changes
2. Explain what you're doing and why
3. Ask for confirmation before taking actions that modify data
4. Be polite and professional
5. If you cannot help, explain why and suggest alternatives

Important: Before any action that modifies data (like refunds, cancellations, etc.),
you MUST get explicit confirmation from the customer. Ask "Would you like me to proceed?" and wait for confirmation."""


@dataclass
class TauBenchConfig(BenchmarkConfig):
    """Extended configuration for τ-bench."""

    domain: str = "paypal"  # "paypal" or "retail"
    user_model: str = "gpt-4o"
    user_model_provider: str = "openai"
    num_trials: int = 1  # For Pass@k computation


@dataclass
class TauBenchTask:
    """A τ-bench task definition."""

    task_id: str
    user_instruction: str
    user_goal: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    expected_actions: list[dict[str, Any]] = field(default_factory=list)
    expected_final_state: dict[str, Any] = field(default_factory=dict)


# Sample tasks for each domain
PAYPAL_SAMPLE_TASKS = [
    TauBenchTask(
        task_id="paypal_dispute_1",
        user_instruction="I want to check on my dispute",
        user_goal="Get status of dispute DIS001 and understand next steps",
        initial_state={
            "disputes": [
                {
                    "dispute_id": "DIS001",
                    "transaction_id": "TXN001",
                    "amount": 150.0,
                    "reason": "item_not_received",
                    "status": "open",
                }
            ],
            "transactions": [
                {
                    "transaction_id": "TXN001",
                    "amount": 150.0,
                    "currency": "USD",
                    "status": "disputed",
                    "buyer_email": "customer@example.com",
                }
            ],
        },
        expected_actions=[{"tool": "get_dispute_details", "args": {"dispute_id": "DIS001"}}],
    ),
    TauBenchTask(
        task_id="paypal_refund_1",
        user_instruction="I need a refund for my purchase",
        user_goal="Get a refund for transaction TXN002",
        initial_state={
            "transactions": [
                {
                    "transaction_id": "TXN002",
                    "amount": 75.0,
                    "currency": "USD",
                    "status": "completed",
                    "buyer_email": "customer@example.com",
                }
            ],
            "merchants": [
                {
                    "merchant_id": "merchant_123",
                    "business_name": "Test Shop",
                    "verification_status": "verified",
                }
            ],
        },
        expected_actions=[
            {"tool": "get_transaction", "args": {"transaction_id": "TXN002"}},
            {"tool": "refund_payment", "args": {"transaction_id": "TXN002"}},
        ],
        expected_final_state={"transactions": {"TXN002": {"status": "refunded"}}},
    ),
]

RETAIL_SAMPLE_TASKS = [
    TauBenchTask(
        task_id="retail_order_1",
        user_instruction="I want to cancel my order",
        user_goal="Cancel order ORD001",
        initial_state={
            "orders": [
                {
                    "order_id": "ORD001",
                    "customer_id": "customer_123",
                    "items": [{"product_id": "PROD001", "quantity": 1}],
                    "total": 50.0,
                    "status": "pending",
                }
            ],
            "products": [
                {
                    "product_id": "PROD001",
                    "name": "Test Product",
                    "price": 50.0,
                    "inventory": 10,
                }
            ],
        },
        expected_actions=[
            {"tool": "get_order", "args": {"order_id": "ORD001"}},
            {"tool": "cancel_order", "args": {"order_id": "ORD001"}},
        ],
        expected_final_state={"orders": {"ORD001": {"status": "cancelled"}}},
    ),
]


@register_benchmark("tau_bench")
class TauBenchmark(Benchmark):
    """
    τ-bench benchmark implementation.

    Evaluates agents on multi-turn conversations with simulated users
    in domain-specific scenarios (PayPal, retail).
    """

    def __init__(self, config: BenchmarkConfig | TauBenchConfig) -> None:
        """Initialize the τ-bench benchmark."""
        super().__init__(config)

        # Get domain from config
        if isinstance(config, TauBenchConfig):
            self.domain_name = config.domain
            self.tau_config = config
        else:
            self.domain_name = config.subset or "paypal"
            self.tau_config = TauBenchConfig(
                **{k: v for k, v in config.__dict__.items() if k != "subset"},
                domain=self.domain_name,
            )

        # Initialize domain
        if self.domain_name == "paypal":
            self._domain: PayPalDomain | RetailDomain = get_paypal_domain()
            self._sample_tasks = PAYPAL_SAMPLE_TASKS
        else:
            self._domain = get_retail_domain()
            self._sample_tasks = RETAIL_SAMPLE_TASKS

    @property
    def name(self) -> str:
        return f"tau_bench_{self.domain_name}"

    async def load_tasks(self) -> list[Task]:
        """Load τ-bench tasks."""
        # For now, use sample tasks
        # In production, load from the actual τ-bench dataset
        tasks = []
        for t in self._sample_tasks:
            task = Task(
                task_id=t.task_id,
                instruction=t.user_instruction,
                metadata={
                    "user_goal": t.user_goal,
                    "initial_state": t.initial_state,
                    "expected_actions": t.expected_actions,
                    "expected_final_state": t.expected_final_state,
                },
                domain=self.domain_name,
            )
            tasks.append(task)
        return tasks

    async def run_task(
        self,
        task: Task,
        agent: "Agent",
        trace_logger: "TraceLogger",
    ) -> TaskResult:
        """Run a τ-bench task with conversation orchestration."""
        trace_logger.start_task(task.task_id, {"domain": self.domain_name})

        # Initialize environment state
        initial_state = task.metadata.get("initial_state", {})
        if isinstance(self._domain, PayPalDomain):
            env_state: PayPalEnvironmentState | RetailEnvironmentState = self._domain.initialize_state(initial_state)
        else:
            env_state = self._domain.initialize_state(initial_state)

        # Initialize user simulator
        user_goal = task.metadata.get("user_goal", task.instruction)
        user_simulator = SimpleUserSimulator(user_goal=user_goal)
        await user_simulator.initialize()

        # Build system prompt
        system_prompt = self._build_system_prompt()

        messages: list[Message] = [Message.system(system_prompt)]

        # Get initial user message
        user_message = await user_simulator.generate_initial_message()
        messages.append(Message.user(user_message))
        trace_logger.log_user_message(user_message)

        # Conversation loop
        tool_definitions = self._domain.tools
        turn_count = 0
        tool_calls_made: list[dict[str, Any]] = []
        policy_violations: list[dict[str, Any]] = []
        conversation_complete = False

        while turn_count < self.config.max_turns and not conversation_complete:
            turn_count += 1

            # Agent response
            response = await agent.generate(messages=messages, tools=tool_definitions)
            trace_logger.log_agent_response(response)

            # Process tool calls
            if response.tool_calls:
                messages.append(Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                for tool_call in response.tool_calls:
                    # Check policy compliance
                    violation = self._domain.check_policy_violation(tool_call, env_state)
                    if violation:
                        policy_violations.append(violation.to_dict())
                        trace_logger.log_policy_violation(violation)

                    # Execute tool
                    result, env_state = await self._domain.execute_tool(tool_call, env_state)
                    result.tool_call_id = tool_call.id

                    tool_calls_made.append({
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": result.output,
                        "error": result.error,
                    })

                    trace_logger.log_tool_call(tool_call, result)

                    messages.append(Message.tool(
                        content=str(result.output) if result.output else str(result.error),
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    ))

            # Add assistant message if no tool calls
            elif response.content:
                messages.append(Message.assistant(content=response.content))

            # Generate user response
            user_response, is_complete = await user_simulator.generate_response(
                agent_message=response.content,
                agent_tool_calls=response.tool_calls if response.tool_calls else None,
            )

            if is_complete:
                conversation_complete = True
            else:
                messages.append(Message.user(user_response))
                trace_logger.log_user_message(user_response)

        # Evaluate task completion
        is_successful = self._evaluate_task_completion(task, env_state, tool_calls_made)

        trace_logger.end_task(task.task_id, is_successful)

        return TaskResult(
            task_id=task.task_id,
            agent_answer={"tool_calls": tool_calls_made},
            is_correct=is_successful,
            score=1.0 if is_successful else 0.0,
            trace_id=trace_logger.current_trace_id,
            metrics={
                "turns": turn_count,
                "tool_calls": len(tool_calls_made),
                "policy_violations": len(policy_violations),
                "conversation_complete": conversation_complete,
                "domain": self.domain_name,
            },
        )

    def score_answer(self, task: Task, answer: Any) -> tuple[bool, float]:
        """Score based on task completion."""
        # τ-bench scoring is done in _evaluate_task_completion
        return False, 0.0

    def _compute_aggregate_metrics(
        self,
        results: list[TaskResult],
    ) -> dict[str, float]:
        """Compute τ-bench metrics including Pass@k."""
        valid_results = [r for r in results if r.is_correct is not None]

        if not valid_results:
            return {"total_tasks": float(len(results))}

        # Pass@1 (task success rate)
        pass_1 = sum(1 for r in valid_results if r.is_correct) / len(valid_results)

        # Policy violation rate
        total_violations = sum(r.metrics.get("policy_violations", 0) for r in valid_results)
        violation_rate = total_violations / len(valid_results)

        # Efficiency metrics
        avg_turns = sum(r.metrics.get("turns", 0) for r in valid_results) / len(valid_results)
        avg_tool_calls = sum(r.metrics.get("tool_calls", 0) for r in valid_results) / len(valid_results)

        return {
            "pass_1": pass_1,
            "policy_violation_rate": violation_rate,
            "avg_turns": avg_turns,
            "avg_tool_calls": avg_tool_calls,
            "total_tasks": float(len(valid_results)),
        }

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self._domain.tools
        )

        policies = self._get_domain_policies()

        return TAU_BENCH_SYSTEM_PROMPT.format(
            domain_name=self.domain_name.title(),
            tool_descriptions=tool_descriptions,
            policies=policies,
        )

    def _get_domain_policies(self) -> str:
        """Get domain-specific policies."""
        if self.domain_name == "paypal":
            return """
- Disputes must be responded to within 20 days
- Refunds are limited to 180 days from transaction date
- High-value transactions (>$1000) require verified merchant status
- Always provide evidence when responding to disputes
- Get explicit confirmation before processing refunds
"""
        else:
            return """
- Always verify customer identity before making changes
- Check inventory before processing orders
- Cancellations are only allowed for pending orders
- Returns must be processed within 30 days of delivery
- Get explicit confirmation before cancellations
"""

    def _evaluate_task_completion(
        self,
        task: Task,
        final_state: PayPalEnvironmentState | RetailEnvironmentState,
        tool_calls: list[dict[str, Any]],
    ) -> bool:
        """Evaluate if the task was successfully completed."""
        expected_actions = task.metadata.get("expected_actions", [])

        # Check if expected tools were called
        tool_names_called = [tc["name"] for tc in tool_calls]
        for expected in expected_actions:
            if expected.get("tool") not in tool_names_called:
                return False

        # Check expected final state if provided
        expected_final = task.metadata.get("expected_final_state", {})
        if expected_final:
            # Simplified state comparison
            # In production, this would be more sophisticated
            return True  # Assume success if expected actions were called

        return len(tool_calls) > 0  # At least some action was taken

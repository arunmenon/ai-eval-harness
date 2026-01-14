"""
τ²-bench benchmark implementation.

This module provides the τ²-bench benchmark with dual-control environments
where both agents and users have tools to act on shared state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...core.benchmark import Benchmark, BenchmarkConfig, register_benchmark
from ...core.types import Message, Task, TaskResult, ToolCall, ToolResult
from ..tau_bench.domains.paypal import PayPalDomain, get_paypal_domain
from ..tau_bench.domains.retail import RetailDomain, get_retail_domain
from ..tau_bench.user_simulator import SimpleUserSimulator

if TYPE_CHECKING:
    from ...core.agent import Agent
    from ...trace.logger import TraceLogger

logger = logging.getLogger(__name__)


@dataclass
class Tau2BenchConfig(BenchmarkConfig):
    """Extended configuration for τ²-bench."""

    domain: str = "paypal"
    user_model: str = "gpt-4o"
    user_model_provider: str = "openai"
    num_trials: int = 1


@dataclass
class UserToolCall:
    """A tool call made by the simulated user in dual-control."""

    name: str
    arguments: dict[str, Any]
    result: Any = None


class DualControlUserSimulator(SimpleUserSimulator):
    """
    User simulator with tool capabilities for τ²-bench.

    In dual-control environments, users can also execute tools
    (e.g., clicking buttons, restarting devices, accepting offers).
    """

    def __init__(
        self,
        user_goal: str,
        user_tools: list[str] | None = None,
        scripted_responses: list[str] | None = None,
    ) -> None:
        """
        Initialize the dual-control user simulator.

        Args:
            user_goal: The user's goal
            user_tools: List of tool names the user can use
            scripted_responses: Optional scripted responses
        """
        super().__init__(user_goal, scripted_responses)
        self.user_tools = user_tools or []
        self._pending_user_actions: list[UserToolCall] = []

    async def generate_response(
        self,
        agent_message: str | None,
        agent_tool_calls: list[Any] | None = None,
    ) -> tuple[str, list[UserToolCall], bool]:
        """
        Generate response with potential user tool calls.

        Returns:
            Tuple of (response_text, user_tool_calls, is_complete)
        """
        response, is_complete = await super().generate_response(
            agent_message, agent_tool_calls
        )

        # Check if user should execute a tool based on agent's message
        user_tools: list[UserToolCall] = []

        if agent_message:
            msg_lower = agent_message.lower()

            # PayPal domain: accept/reject offers
            if "accept" in msg_lower and "offer" in msg_lower:
                if "respond_to_offer" in self.user_tools:
                    user_tools.append(UserToolCall(
                        name="respond_to_offer",
                        arguments={"action": "accept"},
                    ))
                    response = "Yes, I'll accept that offer."

            # Telecom domain: restart device
            if "restart" in msg_lower and ("device" in msg_lower or "router" in msg_lower):
                if "restart_device" in self.user_tools:
                    user_tools.append(UserToolCall(
                        name="restart_device",
                        arguments={"device": "router"},
                    ))
                    response = "Done, I've restarted the router."

        return response, user_tools, is_complete


@register_benchmark("tau2_bench")
class Tau2Benchmark(Benchmark):
    """
    τ²-bench benchmark implementation.

    Extends τ-bench with dual-control environments where both
    agent and user can act on shared state via tools.
    """

    def __init__(self, config: BenchmarkConfig | Tau2BenchConfig) -> None:
        """Initialize the τ²-bench benchmark."""
        super().__init__(config)

        if isinstance(config, Tau2BenchConfig):
            self.domain_name = config.domain
            self.tau2_config = config
        else:
            self.domain_name = config.subset or "paypal"
            self.tau2_config = Tau2BenchConfig(
                **{k: v for k, v in config.__dict__.items() if k != "subset"},
                domain=self.domain_name,
            )

        # Initialize domain
        if self.domain_name == "paypal":
            self._domain: PayPalDomain | RetailDomain = get_paypal_domain()
            self._user_tools = [
                "open_dispute",
                "respond_to_offer",
                "provide_buyer_evidence",
                "escalate_dispute",
                "close_dispute",
            ]
        else:
            self._domain = get_retail_domain()
            self._user_tools = []

    @property
    def name(self) -> str:
        return f"tau2_bench_{self.domain_name}"

    async def load_tasks(self) -> list[Task]:
        """Load τ²-bench tasks."""
        # Sample dual-control tasks
        tasks = [
            Task(
                task_id="tau2_paypal_dispute_resolution",
                instruction="Help me resolve this dispute with the buyer",
                metadata={
                    "user_goal": "Get the dispute resolved, either by accepting a fair offer or providing evidence",
                    "user_tools": ["respond_to_offer", "provide_buyer_evidence"],
                    "initial_state": {
                        "disputes": [
                            {
                                "dispute_id": "DIS100",
                                "transaction_id": "TXN100",
                                "amount": 200.0,
                                "reason": "item_not_as_described",
                                "status": "open",
                            }
                        ],
                        "transactions": [
                            {
                                "transaction_id": "TXN100",
                                "amount": 200.0,
                                "status": "disputed",
                            }
                        ],
                    },
                },
                domain=self.domain_name,
            ),
        ]
        return tasks

    async def run_task(
        self,
        task: Task,
        agent: "Agent",
        trace_logger: "TraceLogger",
    ) -> TaskResult:
        """Run a τ²-bench task with dual-control."""
        trace_logger.start_task(task.task_id, {"domain": self.domain_name})

        # Initialize state
        initial_state = task.metadata.get("initial_state", {})
        env_state = self._domain.initialize_state(initial_state)

        # Initialize dual-control user simulator
        user_goal = task.metadata.get("user_goal", task.instruction)
        user_tools = task.metadata.get("user_tools", self._user_tools)
        user_simulator = DualControlUserSimulator(
            user_goal=user_goal,
            user_tools=user_tools,
        )
        await user_simulator.initialize()

        # System prompt
        system_prompt = self._build_system_prompt()
        messages: list[Message] = [Message.system(system_prompt)]

        # Initial user message
        user_message = await user_simulator.generate_initial_message()
        messages.append(Message.user(user_message))
        trace_logger.log_user_message(user_message)

        # Conversation loop
        tool_definitions = self._domain.tools
        turn_count = 0
        agent_tool_calls: list[dict[str, Any]] = []
        user_tool_calls: list[dict[str, Any]] = []
        policy_violations: list[dict[str, Any]] = []
        recovery_attempts = 0
        conversation_complete = False

        while turn_count < self.config.max_turns and not conversation_complete:
            turn_count += 1

            # Agent turn
            response = await agent.generate(messages=messages, tools=tool_definitions)
            trace_logger.log_agent_response(response)

            # Process agent tool calls
            if response.tool_calls:
                messages.append(Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                for tool_call in response.tool_calls:
                    # Policy check
                    violation = self._domain.check_policy_violation(tool_call, env_state)
                    if violation:
                        policy_violations.append(violation.to_dict())
                        trace_logger.log_policy_violation(violation)

                    # Execute tool
                    result, env_state = await self._domain.execute_tool(tool_call, env_state)
                    result.tool_call_id = tool_call.id

                    agent_tool_calls.append({
                        "turn": turn_count,
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": result.output,
                    })

                    trace_logger.log_tool_call(tool_call, result)

                    messages.append(Message.tool(
                        content=str(result.output) if result.output else str(result.error),
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    ))

            elif response.content:
                messages.append(Message.assistant(content=response.content))

            # User turn with potential tool calls
            user_response, user_tools_used, is_complete = await user_simulator.generate_response(
                agent_message=response.content,
                agent_tool_calls=response.tool_calls,
            )

            # Execute user tools
            for utc in user_tools_used:
                user_tool_calls.append({
                    "turn": turn_count,
                    "name": utc.name,
                    "arguments": utc.arguments,
                })
                # Log user tool call
                trace_logger.log_user_tool_call(
                    ToolCall(id=f"user_{turn_count}", name=utc.name, arguments=utc.arguments)
                )

            if is_complete:
                conversation_complete = True
            else:
                messages.append(Message.user(user_response))
                trace_logger.log_user_message(user_response, [
                    ToolCall(id="", name=u.name, arguments=u.arguments) for u in user_tools_used
                ])

            # Check for recovery from violations
            if policy_violations and len(agent_tool_calls) > len(policy_violations):
                recovery_attempts += 1

        # Evaluate
        is_successful = len(agent_tool_calls) > 0 and conversation_complete

        trace_logger.end_task(task.task_id, is_successful)

        return TaskResult(
            task_id=task.task_id,
            agent_answer={
                "agent_tool_calls": agent_tool_calls,
                "user_tool_calls": user_tool_calls,
            },
            is_correct=is_successful,
            score=1.0 if is_successful else 0.0,
            trace_id=trace_logger.current_trace_id,
            metrics={
                "turns": turn_count,
                "agent_tool_calls": len(agent_tool_calls),
                "user_tool_calls": len(user_tool_calls),
                "policy_violations": len(policy_violations),
                "recovery_attempts": recovery_attempts,
                "conversation_complete": conversation_complete,
                "domain": self.domain_name,
            },
        )

    def score_answer(self, task: Task, answer: Any) -> tuple[bool, float]:
        """Score based on task completion."""
        return False, 0.0

    def _compute_aggregate_metrics(
        self,
        results: list[TaskResult],
    ) -> dict[str, float]:
        """Compute τ²-bench metrics."""
        valid_results = [r for r in results if r.is_correct is not None]

        if not valid_results:
            return {"total_tasks": float(len(results))}

        # Pass@1
        pass_1 = sum(1 for r in valid_results if r.is_correct) / len(valid_results)

        # Policy metrics
        total_violations = sum(r.metrics.get("policy_violations", 0) for r in valid_results)
        violation_rate = total_violations / len(valid_results)

        # Recovery rate
        tasks_with_violations = [r for r in valid_results if r.metrics.get("policy_violations", 0) > 0]
        if tasks_with_violations:
            recovery_rate = sum(
                1 for r in tasks_with_violations if r.metrics.get("recovery_attempts", 0) > 0
            ) / len(tasks_with_violations)
        else:
            recovery_rate = 0.0

        # Efficiency
        avg_agent_tools = sum(r.metrics.get("agent_tool_calls", 0) for r in valid_results) / len(valid_results)
        avg_user_tools = sum(r.metrics.get("user_tool_calls", 0) for r in valid_results) / len(valid_results)

        return {
            "pass_1": pass_1,
            "policy_violation_rate": violation_rate,
            "recovery_rate": recovery_rate,
            "avg_agent_tool_calls": avg_agent_tools,
            "avg_user_tool_calls": avg_user_tools,
            "total_tasks": float(len(valid_results)),
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt for dual-control."""
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self._domain.tools
        )

        return f"""You are a customer service agent for {self.domain_name.title()}.

In this dual-control environment, both you and the customer can take actions.
The customer may execute their own tools in response to your instructions.

Available tools:
{tool_descriptions}

Guidelines:
1. Guide the customer through the resolution process
2. When actions require customer input, instruct them clearly
3. The customer can accept/reject offers and provide information
4. Get confirmation before taking irreversible actions
5. Be patient and helpful throughout the interaction"""

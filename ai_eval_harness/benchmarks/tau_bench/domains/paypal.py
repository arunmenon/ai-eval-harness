"""
PayPal Commerce domain for τ-bench.

This module defines the PayPal domain with tools for handling:
- Disputes & chargebacks
- Payments & refunds
- Merchant onboarding
- Orders & subscriptions
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ....core.types import ToolCall, ToolDefinition, ToolParameter, ToolResult, PolicyViolation, PolicyViolationType

logger = logging.getLogger(__name__)


@dataclass
class DisputeState:
    """State of a dispute."""
    dispute_id: str
    transaction_id: str
    amount: float
    reason: str
    status: str  # open, under_review, resolved, escalated
    evidence_submitted: list[dict[str, Any]] = field(default_factory=list)
    offers_made: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class TransactionState:
    """State of a transaction."""
    transaction_id: str
    amount: float
    currency: str
    status: str  # completed, pending, refunded, disputed
    merchant_id: str
    buyer_email: str
    created_at: str = ""
    refund_amount: float = 0.0


@dataclass
class MerchantState:
    """State of a merchant account."""
    merchant_id: str
    business_name: str
    email: str
    verification_status: str  # pending, verified, restricted
    bank_linked: bool = False
    enabled_features: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class PayPalEnvironmentState:
    """Complete state of the PayPal environment."""
    disputes: dict[str, DisputeState] = field(default_factory=dict)
    transactions: dict[str, TransactionState] = field(default_factory=dict)
    merchants: dict[str, MerchantState] = field(default_factory=dict)
    current_user_id: str = ""
    current_merchant_id: str = ""
    balance: float = 10000.0


class PayPalDomain:
    """
    PayPal Commerce domain for τ-bench.

    Provides tools for handling disputes, payments, refunds, and merchant
    onboarding, with policy constraints that agents must follow.
    """

    DOMAIN_NAME = "paypal"

    # Policy constants
    DISPUTE_RESPONSE_DAYS = 20
    REFUND_WINDOW_DAYS = 180
    HIGH_VALUE_THRESHOLD = 1000.0

    def __init__(self) -> None:
        """Initialize the PayPal domain."""
        self._state: PayPalEnvironmentState | None = None
        self._confirmation_pending: dict[str, dict[str, Any]] = {}

    @property
    def tools(self) -> list[ToolDefinition]:
        """Get all tool definitions for this domain."""
        return [
            # Disputes
            self._tool_list_disputes(),
            self._tool_get_dispute_details(),
            self._tool_provide_evidence(),
            self._tool_accept_claim(),
            self._tool_appeal_decision(),
            self._tool_make_offer(),
            # Payments
            self._tool_get_transaction(),
            self._tool_search_transactions(),
            self._tool_refund_payment(),
            self._tool_get_balance(),
            # Merchant
            self._tool_get_merchant_info(),
            self._tool_update_merchant_settings(),
        ]

    def initialize_state(self, initial_state: dict[str, Any]) -> PayPalEnvironmentState:
        """Initialize the environment state."""
        state = PayPalEnvironmentState()

        # Load disputes
        for d in initial_state.get("disputes", []):
            dispute = DisputeState(
                dispute_id=d.get("dispute_id", str(uuid.uuid4())),
                transaction_id=d.get("transaction_id", ""),
                amount=d.get("amount", 0.0),
                reason=d.get("reason", "item_not_received"),
                status=d.get("status", "open"),
                created_at=d.get("created_at", datetime.utcnow().isoformat()),
                updated_at=datetime.utcnow().isoformat(),
            )
            state.disputes[dispute.dispute_id] = dispute

        # Load transactions
        for t in initial_state.get("transactions", []):
            txn = TransactionState(
                transaction_id=t.get("transaction_id", str(uuid.uuid4())),
                amount=t.get("amount", 0.0),
                currency=t.get("currency", "USD"),
                status=t.get("status", "completed"),
                merchant_id=t.get("merchant_id", ""),
                buyer_email=t.get("buyer_email", ""),
                created_at=t.get("created_at", datetime.utcnow().isoformat()),
            )
            state.transactions[txn.transaction_id] = txn

        # Load merchants
        for m in initial_state.get("merchants", []):
            merchant = MerchantState(
                merchant_id=m.get("merchant_id", str(uuid.uuid4())),
                business_name=m.get("business_name", ""),
                email=m.get("email", ""),
                verification_status=m.get("verification_status", "verified"),
                bank_linked=m.get("bank_linked", True),
                enabled_features=m.get("enabled_features", []),
                created_at=m.get("created_at", datetime.utcnow().isoformat()),
            )
            state.merchants[merchant.merchant_id] = merchant

        state.current_user_id = initial_state.get("current_user_id", "user_123")
        state.current_merchant_id = initial_state.get("current_merchant_id", "merchant_123")
        state.balance = initial_state.get("balance", 10000.0)

        self._state = state
        return state

    def check_policy_violation(
        self,
        tool_call: ToolCall,
        state: PayPalEnvironmentState,
    ) -> PolicyViolation | None:
        """Check if a tool call violates any policies."""
        tool_name = tool_call.name
        args = tool_call.arguments

        # Check for confirmation requirement on modifying actions
        modifying_actions = ["refund_payment", "accept_claim", "make_offer", "appeal_decision"]
        if tool_name in modifying_actions:
            action_key = f"{tool_name}_{tool_call.id}"
            if action_key not in self._confirmation_pending:
                # First call - store for confirmation
                self._confirmation_pending[action_key] = {
                    "tool_name": tool_name,
                    "arguments": args,
                }
                return PolicyViolation(
                    violation_type=PolicyViolationType.MISSING_CONFIRMATION,
                    description=f"Action '{tool_name}' requires user confirmation before execution",
                    tool_call=tool_call,
                )

        # Check refund window
        if tool_name == "refund_payment":
            txn_id = args.get("transaction_id")
            if txn_id and txn_id in state.transactions:
                txn = state.transactions[txn_id]
                created = datetime.fromisoformat(txn.created_at)
                if datetime.utcnow() - created > timedelta(days=self.REFUND_WINDOW_DAYS):
                    return PolicyViolation(
                        violation_type=PolicyViolationType.EXCEEDED_LIMIT,
                        description=f"Refund window of {self.REFUND_WINDOW_DAYS} days has expired",
                        tool_call=tool_call,
                    )

        # Check high-value transaction verification
        if tool_name == "refund_payment":
            amount = args.get("amount", 0)
            if amount > self.HIGH_VALUE_THRESHOLD:
                merchant_id = state.current_merchant_id
                if merchant_id in state.merchants:
                    merchant = state.merchants[merchant_id]
                    if merchant.verification_status != "verified":
                        return PolicyViolation(
                            violation_type=PolicyViolationType.MISSING_VERIFICATION,
                            description="High-value refund requires verified merchant status",
                            tool_call=tool_call,
                        )

        return None

    async def execute_tool(
        self,
        tool_call: ToolCall,
        state: PayPalEnvironmentState,
    ) -> tuple[ToolResult, PayPalEnvironmentState]:
        """Execute a tool and return the result and updated state."""
        tool_name = tool_call.name
        args = tool_call.arguments

        handlers = {
            "list_disputes": self._execute_list_disputes,
            "get_dispute_details": self._execute_get_dispute_details,
            "provide_evidence": self._execute_provide_evidence,
            "accept_claim": self._execute_accept_claim,
            "appeal_decision": self._execute_appeal_decision,
            "make_offer": self._execute_make_offer,
            "get_transaction": self._execute_get_transaction,
            "search_transactions": self._execute_search_transactions,
            "refund_payment": self._execute_refund_payment,
            "get_balance": self._execute_get_balance,
            "get_merchant_info": self._execute_get_merchant_info,
            "update_merchant_settings": self._execute_update_merchant_settings,
        }

        handler = handlers.get(tool_name)
        if handler:
            result = handler(args, state)
            return result, state
        else:
            return ToolResult(
                tool_call_id=tool_call.id,
                output=None,
                error=f"Unknown tool: {tool_name}",
            ), state

    def confirm_action(self, action_key: str) -> None:
        """Confirm a pending action."""
        if action_key in self._confirmation_pending:
            del self._confirmation_pending[action_key]

    # Tool definitions
    def _tool_list_disputes(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_disputes",
            description="List disputes for the current merchant. Filter by status or date range.",
            parameters=[
                ToolParameter(name="status", type="string", description="Filter by status (open, under_review, resolved, escalated)", required=False),
                ToolParameter(name="limit", type="number", description="Maximum number of disputes to return", required=False),
            ],
        )

    def _tool_get_dispute_details(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_dispute_details",
            description="Get detailed information about a specific dispute.",
            parameters=[
                ToolParameter(name="dispute_id", type="string", description="The dispute ID", required=True),
            ],
        )

    def _tool_provide_evidence(self) -> ToolDefinition:
        return ToolDefinition(
            name="provide_evidence",
            description="Submit evidence for a dispute (tracking info, refund ID, delivery confirmation).",
            parameters=[
                ToolParameter(name="dispute_id", type="string", description="The dispute ID", required=True),
                ToolParameter(name="evidence_type", type="string", description="Type of evidence (tracking, refund_id, delivery_confirmation, other)", required=True),
                ToolParameter(name="evidence_value", type="string", description="The evidence value (tracking number, refund ID, etc.)", required=True),
                ToolParameter(name="notes", type="string", description="Additional notes", required=False),
            ],
        )

    def _tool_accept_claim(self) -> ToolDefinition:
        return ToolDefinition(
            name="accept_claim",
            description="Accept liability for a dispute claim. This will automatically refund the customer.",
            parameters=[
                ToolParameter(name="dispute_id", type="string", description="The dispute ID", required=True),
                ToolParameter(name="reason", type="string", description="Reason for accepting the claim", required=False),
            ],
        )

    def _tool_appeal_decision(self) -> ToolDefinition:
        return ToolDefinition(
            name="appeal_decision",
            description="Appeal a dispute ruling with new evidence.",
            parameters=[
                ToolParameter(name="dispute_id", type="string", description="The dispute ID", required=True),
                ToolParameter(name="appeal_reason", type="string", description="Reason for the appeal", required=True),
                ToolParameter(name="new_evidence", type="string", description="New evidence to support the appeal", required=False),
            ],
        )

    def _tool_make_offer(self) -> ToolDefinition:
        return ToolDefinition(
            name="make_offer",
            description="Make a settlement offer to resolve the dispute.",
            parameters=[
                ToolParameter(name="dispute_id", type="string", description="The dispute ID", required=True),
                ToolParameter(name="offer_type", type="string", description="Type of offer (full_refund, partial_refund, refund_with_return)", required=True),
                ToolParameter(name="amount", type="number", description="Offer amount (for partial refunds)", required=False),
                ToolParameter(name="message", type="string", description="Message to the buyer", required=False),
            ],
        )

    def _tool_get_transaction(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_transaction",
            description="Get details of a specific transaction.",
            parameters=[
                ToolParameter(name="transaction_id", type="string", description="The transaction ID", required=True),
            ],
        )

    def _tool_search_transactions(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_transactions",
            description="Search transactions by various criteria.",
            parameters=[
                ToolParameter(name="status", type="string", description="Filter by status", required=False),
                ToolParameter(name="min_amount", type="number", description="Minimum amount", required=False),
                ToolParameter(name="max_amount", type="number", description="Maximum amount", required=False),
                ToolParameter(name="limit", type="number", description="Maximum results", required=False),
            ],
        )

    def _tool_refund_payment(self) -> ToolDefinition:
        return ToolDefinition(
            name="refund_payment",
            description="Issue a refund for a transaction. Requires user confirmation.",
            parameters=[
                ToolParameter(name="transaction_id", type="string", description="The transaction ID", required=True),
                ToolParameter(name="amount", type="number", description="Refund amount (leave empty for full refund)", required=False),
                ToolParameter(name="reason", type="string", description="Reason for the refund", required=False),
            ],
        )

    def _tool_get_balance(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_balance",
            description="Get the current account balance.",
            parameters=[],
        )

    def _tool_get_merchant_info(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_merchant_info",
            description="Get merchant account information.",
            parameters=[
                ToolParameter(name="merchant_id", type="string", description="Merchant ID (optional, defaults to current)", required=False),
            ],
        )

    def _tool_update_merchant_settings(self) -> ToolDefinition:
        return ToolDefinition(
            name="update_merchant_settings",
            description="Update merchant settings.",
            parameters=[
                ToolParameter(name="setting_name", type="string", description="Setting to update", required=True),
                ToolParameter(name="setting_value", type="string", description="New value", required=True),
            ],
        )

    # Tool execution handlers
    def _execute_list_disputes(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        disputes = list(state.disputes.values())
        status_filter = args.get("status")
        if status_filter:
            disputes = [d for d in disputes if d.status == status_filter]
        limit = args.get("limit", 10)
        disputes = disputes[:limit]

        result = [
            {"dispute_id": d.dispute_id, "amount": d.amount, "reason": d.reason, "status": d.status}
            for d in disputes
        ]
        return ToolResult(tool_call_id="", output=result)

    def _execute_get_dispute_details(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        dispute_id = args.get("dispute_id")
        if dispute_id not in state.disputes:
            return ToolResult(tool_call_id="", output=None, error=f"Dispute not found: {dispute_id}")
        d = state.disputes[dispute_id]
        return ToolResult(tool_call_id="", output={
            "dispute_id": d.dispute_id,
            "transaction_id": d.transaction_id,
            "amount": d.amount,
            "reason": d.reason,
            "status": d.status,
            "evidence_submitted": d.evidence_submitted,
            "offers_made": d.offers_made,
            "created_at": d.created_at,
        })

    def _execute_provide_evidence(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        dispute_id = args.get("dispute_id")
        if dispute_id not in state.disputes:
            return ToolResult(tool_call_id="", output=None, error=f"Dispute not found: {dispute_id}")

        dispute = state.disputes[dispute_id]
        dispute.evidence_submitted.append({
            "type": args.get("evidence_type"),
            "value": args.get("evidence_value"),
            "notes": args.get("notes"),
            "submitted_at": datetime.utcnow().isoformat(),
        })
        dispute.updated_at = datetime.utcnow().isoformat()
        return ToolResult(tool_call_id="", output={"success": True, "message": "Evidence submitted successfully"})

    def _execute_accept_claim(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        dispute_id = args.get("dispute_id")
        if dispute_id not in state.disputes:
            return ToolResult(tool_call_id="", output=None, error=f"Dispute not found: {dispute_id}")

        dispute = state.disputes[dispute_id]
        dispute.status = "resolved"
        dispute.updated_at = datetime.utcnow().isoformat()

        # Process refund
        if dispute.transaction_id in state.transactions:
            txn = state.transactions[dispute.transaction_id]
            txn.status = "refunded"
            txn.refund_amount = dispute.amount
            state.balance -= dispute.amount

        return ToolResult(tool_call_id="", output={"success": True, "message": f"Claim accepted. Refund of ${dispute.amount} processed."})

    def _execute_appeal_decision(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        dispute_id = args.get("dispute_id")
        if dispute_id not in state.disputes:
            return ToolResult(tool_call_id="", output=None, error=f"Dispute not found: {dispute_id}")

        dispute = state.disputes[dispute_id]
        dispute.status = "under_review"
        dispute.updated_at = datetime.utcnow().isoformat()
        return ToolResult(tool_call_id="", output={"success": True, "message": "Appeal submitted for review"})

    def _execute_make_offer(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        dispute_id = args.get("dispute_id")
        if dispute_id not in state.disputes:
            return ToolResult(tool_call_id="", output=None, error=f"Dispute not found: {dispute_id}")

        dispute = state.disputes[dispute_id]
        dispute.offers_made.append({
            "type": args.get("offer_type"),
            "amount": args.get("amount"),
            "message": args.get("message"),
            "made_at": datetime.utcnow().isoformat(),
        })
        dispute.updated_at = datetime.utcnow().isoformat()
        return ToolResult(tool_call_id="", output={"success": True, "message": "Offer sent to buyer"})

    def _execute_get_transaction(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        txn_id = args.get("transaction_id")
        if txn_id not in state.transactions:
            return ToolResult(tool_call_id="", output=None, error=f"Transaction not found: {txn_id}")
        t = state.transactions[txn_id]
        return ToolResult(tool_call_id="", output={
            "transaction_id": t.transaction_id,
            "amount": t.amount,
            "currency": t.currency,
            "status": t.status,
            "buyer_email": t.buyer_email,
            "created_at": t.created_at,
            "refund_amount": t.refund_amount,
        })

    def _execute_search_transactions(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        transactions = list(state.transactions.values())
        if args.get("status"):
            transactions = [t for t in transactions if t.status == args["status"]]
        if args.get("min_amount"):
            transactions = [t for t in transactions if t.amount >= args["min_amount"]]
        if args.get("max_amount"):
            transactions = [t for t in transactions if t.amount <= args["max_amount"]]
        limit = args.get("limit", 10)
        transactions = transactions[:limit]

        result = [{"transaction_id": t.transaction_id, "amount": t.amount, "status": t.status} for t in transactions]
        return ToolResult(tool_call_id="", output=result)

    def _execute_refund_payment(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        txn_id = args.get("transaction_id")
        if txn_id not in state.transactions:
            return ToolResult(tool_call_id="", output=None, error=f"Transaction not found: {txn_id}")

        txn = state.transactions[txn_id]
        refund_amount = args.get("amount", txn.amount)

        if refund_amount > txn.amount - txn.refund_amount:
            return ToolResult(tool_call_id="", output=None, error="Refund amount exceeds available amount")

        txn.refund_amount += refund_amount
        if txn.refund_amount >= txn.amount:
            txn.status = "refunded"
        state.balance -= refund_amount

        return ToolResult(tool_call_id="", output={"success": True, "refund_amount": refund_amount, "new_balance": state.balance})

    def _execute_get_balance(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        return ToolResult(tool_call_id="", output={"balance": state.balance, "currency": "USD"})

    def _execute_get_merchant_info(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        merchant_id = args.get("merchant_id", state.current_merchant_id)
        if merchant_id not in state.merchants:
            return ToolResult(tool_call_id="", output=None, error=f"Merchant not found: {merchant_id}")
        m = state.merchants[merchant_id]
        return ToolResult(tool_call_id="", output={
            "merchant_id": m.merchant_id,
            "business_name": m.business_name,
            "email": m.email,
            "verification_status": m.verification_status,
            "bank_linked": m.bank_linked,
            "enabled_features": m.enabled_features,
        })

    def _execute_update_merchant_settings(self, args: dict[str, Any], state: PayPalEnvironmentState) -> ToolResult:
        merchant_id = state.current_merchant_id
        if merchant_id not in state.merchants:
            return ToolResult(tool_call_id="", output=None, error="Merchant not found")

        # Simulate setting update
        return ToolResult(tool_call_id="", output={"success": True, "message": f"Setting {args.get('setting_name')} updated"})


def get_paypal_domain() -> PayPalDomain:
    """Get an instance of the PayPal domain."""
    return PayPalDomain()

"""
Retail domain for τ-bench.

This module defines the retail domain with tools for handling
product searches, orders, returns, and customer service.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ....core.types import ToolCall, ToolDefinition, ToolParameter, ToolResult, PolicyViolation, PolicyViolationType

logger = logging.getLogger(__name__)


@dataclass
class ProductState:
    """State of a product."""
    product_id: str
    name: str
    price: float
    category: str
    inventory: int = 0


@dataclass
class OrderState:
    """State of an order."""
    order_id: str
    customer_id: str
    items: list[dict[str, Any]] = field(default_factory=list)
    total: float = 0.0
    status: str = "pending"  # pending, confirmed, shipped, delivered, cancelled, returned
    created_at: str = ""


@dataclass
class RetailEnvironmentState:
    """Complete state of the retail environment."""
    products: dict[str, ProductState] = field(default_factory=dict)
    orders: dict[str, OrderState] = field(default_factory=dict)
    customers: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_customer_id: str = ""


class RetailDomain:
    """
    Retail domain for τ-bench.

    Provides tools for handling product searches, orders, returns,
    and customer service interactions.
    """

    DOMAIN_NAME = "retail"

    def __init__(self) -> None:
        """Initialize the retail domain."""
        self._state: RetailEnvironmentState | None = None
        self._confirmation_pending: dict[str, dict[str, Any]] = {}

    @property
    def tools(self) -> list[ToolDefinition]:
        """Get all tool definitions for this domain."""
        return [
            self._tool_search_products(),
            self._tool_get_product(),
            self._tool_get_order(),
            self._tool_create_order(),
            self._tool_cancel_order(),
            self._tool_process_return(),
            self._tool_check_inventory(),
            self._tool_get_customer_info(),
        ]

    def initialize_state(self, initial_state: dict[str, Any]) -> RetailEnvironmentState:
        """Initialize the environment state."""
        state = RetailEnvironmentState()

        # Load products
        for p in initial_state.get("products", []):
            product = ProductState(
                product_id=p.get("product_id", str(uuid.uuid4())),
                name=p.get("name", ""),
                price=p.get("price", 0.0),
                category=p.get("category", ""),
                inventory=p.get("inventory", 0),
            )
            state.products[product.product_id] = product

        # Load orders
        for o in initial_state.get("orders", []):
            order = OrderState(
                order_id=o.get("order_id", str(uuid.uuid4())),
                customer_id=o.get("customer_id", ""),
                items=o.get("items", []),
                total=o.get("total", 0.0),
                status=o.get("status", "pending"),
                created_at=o.get("created_at", datetime.utcnow().isoformat()),
            )
            state.orders[order.order_id] = order

        # Load customers
        for c in initial_state.get("customers", []):
            state.customers[c.get("customer_id", "")] = c

        state.current_customer_id = initial_state.get("current_customer_id", "customer_123")

        self._state = state
        return state

    def check_policy_violation(
        self,
        tool_call: ToolCall,
        state: RetailEnvironmentState,
    ) -> PolicyViolation | None:
        """Check if a tool call violates any policies."""
        tool_name = tool_call.name
        args = tool_call.arguments

        # Check for confirmation requirement on modifying actions
        modifying_actions = ["create_order", "cancel_order", "process_return"]
        if tool_name in modifying_actions:
            action_key = f"{tool_name}_{tool_call.id}"
            if action_key not in self._confirmation_pending:
                self._confirmation_pending[action_key] = {
                    "tool_name": tool_name,
                    "arguments": args,
                }
                return PolicyViolation(
                    violation_type=PolicyViolationType.MISSING_CONFIRMATION,
                    description=f"Action '{tool_name}' requires user confirmation",
                    tool_call=tool_call,
                )

        # Check inventory before order
        if tool_name == "create_order":
            items = args.get("items", [])
            for item in items:
                product_id = item.get("product_id")
                quantity = item.get("quantity", 1)
                if product_id in state.products:
                    if state.products[product_id].inventory < quantity:
                        return PolicyViolation(
                            violation_type=PolicyViolationType.EXCEEDED_LIMIT,
                            description=f"Insufficient inventory for product {product_id}",
                            tool_call=tool_call,
                        )

        return None

    async def execute_tool(
        self,
        tool_call: ToolCall,
        state: RetailEnvironmentState,
    ) -> tuple[ToolResult, RetailEnvironmentState]:
        """Execute a tool and return the result and updated state."""
        tool_name = tool_call.name
        args = tool_call.arguments

        handlers = {
            "search_products": self._execute_search_products,
            "get_product": self._execute_get_product,
            "get_order": self._execute_get_order,
            "create_order": self._execute_create_order,
            "cancel_order": self._execute_cancel_order,
            "process_return": self._execute_process_return,
            "check_inventory": self._execute_check_inventory,
            "get_customer_info": self._execute_get_customer_info,
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

    # Tool definitions
    def _tool_search_products(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_products",
            description="Search for products by name or category.",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query", required=False),
                ToolParameter(name="category", type="string", description="Product category", required=False),
                ToolParameter(name="limit", type="number", description="Max results", required=False),
            ],
        )

    def _tool_get_product(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_product",
            description="Get product details by ID.",
            parameters=[
                ToolParameter(name="product_id", type="string", description="Product ID", required=True),
            ],
        )

    def _tool_get_order(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_order",
            description="Get order details by ID.",
            parameters=[
                ToolParameter(name="order_id", type="string", description="Order ID", required=True),
            ],
        )

    def _tool_create_order(self) -> ToolDefinition:
        return ToolDefinition(
            name="create_order",
            description="Create a new order. Requires user confirmation.",
            parameters=[
                ToolParameter(name="items", type="array", description="List of items [{product_id, quantity}]", required=True),
                ToolParameter(name="customer_id", type="string", description="Customer ID", required=False),
            ],
        )

    def _tool_cancel_order(self) -> ToolDefinition:
        return ToolDefinition(
            name="cancel_order",
            description="Cancel an order. Requires user confirmation.",
            parameters=[
                ToolParameter(name="order_id", type="string", description="Order ID", required=True),
                ToolParameter(name="reason", type="string", description="Cancellation reason", required=False),
            ],
        )

    def _tool_process_return(self) -> ToolDefinition:
        return ToolDefinition(
            name="process_return",
            description="Process a product return. Requires user confirmation.",
            parameters=[
                ToolParameter(name="order_id", type="string", description="Order ID", required=True),
                ToolParameter(name="items", type="array", description="Items to return [{product_id, quantity, reason}]", required=True),
            ],
        )

    def _tool_check_inventory(self) -> ToolDefinition:
        return ToolDefinition(
            name="check_inventory",
            description="Check inventory for a product.",
            parameters=[
                ToolParameter(name="product_id", type="string", description="Product ID", required=True),
            ],
        )

    def _tool_get_customer_info(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_customer_info",
            description="Get customer information.",
            parameters=[
                ToolParameter(name="customer_id", type="string", description="Customer ID", required=False),
            ],
        )

    # Tool execution handlers
    def _execute_search_products(self, args: dict[str, Any], state: RetailEnvironmentState) -> ToolResult:
        products = list(state.products.values())
        query = args.get("query", "").lower()
        category = args.get("category")

        if query:
            products = [p for p in products if query in p.name.lower()]
        if category:
            products = [p for p in products if p.category == category]

        limit = args.get("limit", 10)
        products = products[:limit]

        result = [{"product_id": p.product_id, "name": p.name, "price": p.price, "category": p.category} for p in products]
        return ToolResult(tool_call_id="", output=result)

    def _execute_get_product(self, args: dict[str, Any], state: RetailEnvironmentState) -> ToolResult:
        product_id = args.get("product_id")
        if product_id not in state.products:
            return ToolResult(tool_call_id="", output=None, error=f"Product not found: {product_id}")
        p = state.products[product_id]
        return ToolResult(tool_call_id="", output={
            "product_id": p.product_id,
            "name": p.name,
            "price": p.price,
            "category": p.category,
            "inventory": p.inventory,
        })

    def _execute_get_order(self, args: dict[str, Any], state: RetailEnvironmentState) -> ToolResult:
        order_id = args.get("order_id")
        if order_id not in state.orders:
            return ToolResult(tool_call_id="", output=None, error=f"Order not found: {order_id}")
        o = state.orders[order_id]
        return ToolResult(tool_call_id="", output={
            "order_id": o.order_id,
            "customer_id": o.customer_id,
            "items": o.items,
            "total": o.total,
            "status": o.status,
            "created_at": o.created_at,
        })

    def _execute_create_order(self, args: dict[str, Any], state: RetailEnvironmentState) -> ToolResult:
        items = args.get("items", [])
        customer_id = args.get("customer_id", state.current_customer_id)

        total = 0.0
        order_items = []
        for item in items:
            product_id = item.get("product_id")
            quantity = item.get("quantity", 1)
            if product_id in state.products:
                product = state.products[product_id]
                product.inventory -= quantity
                item_total = product.price * quantity
                total += item_total
                order_items.append({
                    "product_id": product_id,
                    "name": product.name,
                    "quantity": quantity,
                    "price": product.price,
                })

        order_id = str(uuid.uuid4())
        order = OrderState(
            order_id=order_id,
            customer_id=customer_id,
            items=order_items,
            total=total,
            status="confirmed",
            created_at=datetime.utcnow().isoformat(),
        )
        state.orders[order_id] = order

        return ToolResult(tool_call_id="", output={"success": True, "order_id": order_id, "total": total})

    def _execute_cancel_order(self, args: dict[str, Any], state: RetailEnvironmentState) -> ToolResult:
        order_id = args.get("order_id")
        if order_id not in state.orders:
            return ToolResult(tool_call_id="", output=None, error=f"Order not found: {order_id}")

        order = state.orders[order_id]
        if order.status in ["shipped", "delivered"]:
            return ToolResult(tool_call_id="", output=None, error="Cannot cancel shipped or delivered orders")

        order.status = "cancelled"

        # Restore inventory
        for item in order.items:
            product_id = item.get("product_id")
            if product_id in state.products:
                state.products[product_id].inventory += item.get("quantity", 1)

        return ToolResult(tool_call_id="", output={"success": True, "message": "Order cancelled"})

    def _execute_process_return(self, args: dict[str, Any], state: RetailEnvironmentState) -> ToolResult:
        order_id = args.get("order_id")
        if order_id not in state.orders:
            return ToolResult(tool_call_id="", output=None, error=f"Order not found: {order_id}")

        order = state.orders[order_id]
        order.status = "returned"

        return ToolResult(tool_call_id="", output={"success": True, "message": "Return processed"})

    def _execute_check_inventory(self, args: dict[str, Any], state: RetailEnvironmentState) -> ToolResult:
        product_id = args.get("product_id")
        if product_id not in state.products:
            return ToolResult(tool_call_id="", output=None, error=f"Product not found: {product_id}")
        p = state.products[product_id]
        return ToolResult(tool_call_id="", output={"product_id": p.product_id, "inventory": p.inventory})

    def _execute_get_customer_info(self, args: dict[str, Any], state: RetailEnvironmentState) -> ToolResult:
        customer_id = args.get("customer_id", state.current_customer_id)
        if customer_id not in state.customers:
            return ToolResult(tool_call_id="", output=None, error=f"Customer not found: {customer_id}")
        return ToolResult(tool_call_id="", output=state.customers[customer_id])


def get_retail_domain() -> RetailDomain:
    """Get an instance of the retail domain."""
    return RetailDomain()

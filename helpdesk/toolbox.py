"""The tools the graph uses, from either source:

- `local_toolbox()` tests/eval: the same backend functions wrapped as tools
                   in-process, so tests run in milliseconds without a subprocess.

Both give the graph the same interface: `read_tools` (handed to LLM agents)
and `call(name, **args)` for the write tools the graph invokes itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.tools import BaseTool, StructuredTool, tool

from helpdesk.agent_loop import tool_text
from helpdesk.mcp_server import READ_TOOLS, WRITE_TOOLS


@tool
def propose_refund(order_id: int, amount: float, reason: str) -> str:
    """Propose a refund for an order. This does NOT issue it: policy checks and, if needed, a human decide.
    Use only after check_refund_eligibility says the order is eligible, with an amount at or below max_refund."""
    return json.dumps({"status": "proposal recorded", "order_id": order_id, "amount": amount})


@tool
def escalate_to_human(reason: str) -> str:
    """Hand the ticket to a human agent: use for complaints, legal or safety issues, fraud, or anything you can't resolve with your tools."""
    return json.dumps({"status": "escalation recorded"})


@dataclass
class Toolbox:
    tools: dict[str, BaseTool]

    @property
    def read_tools(self) -> dict[str, BaseTool]:
        return {k: v for k, v in self.tools.items() if k in READ_TOOLS}

    def for_agent(self, names: list[str]) -> list[BaseTool]:
        local = {"propose_refund": propose_refund, "escalate_to_human": escalate_to_human}
        return [self.read_tools.get(n) or local[n] for n in names]

    async def call(self, name: str, **args) -> dict:
        if name not in self.tools:
            raise KeyError(name)
        out = await self.tools[name].ainvoke(args)
        text = tool_text(out)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}


def local_toolbox(conn, auto_limit: float) -> Toolbox:
    from helpdesk import backend

    def safe(fn, *args, **kwargs) -> str:
        try:
            return backend.to_json(fn(*args, **kwargs))
        except backend.ToolError as exc:
            return backend.to_json({"error": str(exc)})

    def find_customer(email: str) -> str:
        """Look up a customer by email. Returns their name, tier and order IDs."""
        return safe(backend.find_customer, conn, email)

    def list_orders(customer_email: str) -> str:
        """List a customer's orders (id, date, status), newest first."""
        return safe(backend.list_orders, conn, customer_email)

    def get_order(order_id: int, customer_email: str) -> str:
        """Get one order's status, tracking, items and refund history. Only works for the customer's own orders."""
        return safe(backend.get_order, conn, order_id, customer_email)

    def check_refund_eligibility(order_id: int, customer_email: str) -> str:
        """Check whether an order can be refunded under policy, and the maximum refundable amount."""
        return safe(backend.check_refund_eligibility, conn, order_id, customer_email)

    def search_help_center(query: str) -> str:
        """Search help-center articles (returns, shipping, accounts, payments, warranty). Returns the top 3."""
        return safe(backend.search_help_center, query)

    def issue_refund(order_id: int, customer_email: str, amount: float, reason: str, approved_by: str) -> str:
        """Issue a refund."""
        return safe(backend.issue_refund, conn, order_id, customer_email, amount, reason, approved_by, auto_limit)

    def update_ticket(ticket_id: str, status: str | None = None, category: str | None = None, resolution: str | None = None) -> str:
        """Create or update a support ticket."""
        return safe(backend.upsert_ticket, conn, ticket_id, status=status, category=category, resolution=resolution)

    fns = [find_customer, list_orders, get_order, check_refund_eligibility, search_help_center, issue_refund, update_ticket]
    assert {f.__name__ for f in fns} == READ_TOOLS | WRITE_TOOLS
    return Toolbox({f.__name__: StructuredTool.from_function(f) for f in fns})

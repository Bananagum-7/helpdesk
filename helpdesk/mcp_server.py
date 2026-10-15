"""MCP server exposing the store's backend as tools.

Run it standalone (stdio):     python -m helpdesk.mcp_server
Use it from Claude Desktop:    see README ("Use the MCP server from Claude Desktop")

Why MCP instead of plain function tools: the backend team owns this server
and decides exactly which operations exist and how they're validated. Any
MCP client (this LangGraph agent, Claude Desktop, an IDE) can use it without
knowing anything about the database, and none of them can run arbitrary SQL.

Two kinds of tools:
- read tools, which the LLM agents may call freely
- write tools (issue_refund, update_ticket), which the agent graph calls only
  from deterministic nodes after policy checks and approval, never the LLM
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from helpdesk import backend
from helpdesk.config import settings
from helpdesk.store import init_db

mcp = FastMCP("helpdesk-store")
_conn = None


def conn():
    global _conn
    if _conn is None:
        _conn = init_db(settings.store_db)
    return _conn


def _call(fn, *args, **kwargs) -> str:
    try:
        return backend.to_json(fn(*args, **kwargs))
    except backend.ToolError as exc:
        # Return errors as data, so the model can read them and adjust, instead of crashing the run.
        return backend.to_json({"error": str(exc)})


@mcp.tool()
def find_customer(email: str) -> str:
    """Look up a customer by email. Returns their name, tier and order IDs."""
    return _call(backend.find_customer, conn(), email)


@mcp.tool()
def list_orders(customer_email: str) -> str:
    """List a customer's orders (id, date, status), newest first."""
    return _call(backend.list_orders, conn(), customer_email)


@mcp.tool()
def get_order(order_id: int, customer_email: str) -> str:
    """Get one order's status, tracking, items and refund history. Only works for the customer's own orders."""
    return _call(backend.get_order, conn(), order_id, customer_email)


@mcp.tool()
def check_refund_eligibility(order_id: int, customer_email: str) -> str:
    """Check whether an order can be refunded under policy, and the maximum refundable amount."""
    return _call(backend.check_refund_eligibility, conn(), order_id, customer_email)


@mcp.tool()
def search_help_center(query: str) -> str:
    """Search help-center articles (returns, shipping, accounts, payments, warranty). Returns the top 3."""
    return _call(backend.search_help_center, query)


@mcp.tool()
def issue_refund(order_id: int, customer_email: str, amount: float, reason: str, approved_by: str) -> str:
    """Issue a refund. Re-validates eligibility and amount; amounts over the auto limit need approved_by='human:<name>'."""
    return _call(backend.issue_refund, conn(), order_id, customer_email, amount, reason, approved_by, settings.auto_refund_limit)


@mcp.tool()
def update_ticket(ticket_id: str, status: str | None = None, category: str | None = None, resolution: str | None = None,
                  customer_email: str | None = None, subject: str | None = None, body: str | None = None) -> str:
    """Create or update a support ticket's status, category, resolution note and contents."""
    return _call(backend.upsert_ticket, conn(), ticket_id, status=status, category=category, resolution=resolution,
                 customer_email=customer_email, subject=subject, body=body)


READ_TOOLS = {"find_customer", "list_orders", "get_order", "check_refund_eligibility", "search_help_center"}
WRITE_TOOLS = {"issue_refund", "update_ticket"}

if __name__ == "__main__":
    mcp.run()  # stdio transport

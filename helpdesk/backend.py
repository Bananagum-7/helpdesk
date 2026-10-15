"""Business operations behind the MCP tools. Plain functions over a DB connection,
so they are unit-tested directly and the MCP server is a thin wrapper.

Authorization lives here, not in the prompt: every customer-scoped call takes
the ticket's customer email and refuses data that belongs to someone else.
A prompt-injected agent asking for another customer's order gets an error.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date

from helpdesk import kb, refund_policy


class ToolError(Exception):
    pass


def today() -> date:
    return date.fromisoformat(os.environ["HELPDESK_TODAY"]) if os.getenv("HELPDESK_TODAY") else date.today()


def _customer(conn: sqlite3.Connection, email: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM customers WHERE lower(email) = lower(?)", (email or "",)).fetchone()
    if row is None:
        raise ToolError(f"no customer with email {email!r}")
    return row


def find_customer(conn, email: str) -> dict:
    c = _customer(conn, email)
    orders = [r[0] for r in conn.execute("SELECT id FROM orders WHERE customer_id = ? ORDER BY placed_on DESC", (c["id"],))]
    return {"customer_id": c["id"], "name": c["name"], "email": c["email"], "tier": c["tier"], "order_ids": orders}


def list_orders(conn, customer_email: str) -> list[dict]:
    c = _customer(conn, customer_email)
    rows = conn.execute("SELECT id, placed_on, status, delivered_on FROM orders WHERE customer_id = ? ORDER BY placed_on DESC", (c["id"],))
    return [dict(r) for r in rows]


def get_order(conn, order_id: int, customer_email: str) -> dict:
    c = _customer(conn, customer_email)
    o = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if o is None or o["customer_id"] != c["id"]:
        # Same message for "doesn't exist" and "not yours": don't leak which order IDs exist.
        raise ToolError(f"order {order_id} not found for this customer")
    items = conn.execute(
        "SELECT oi.sku, p.name, oi.qty, oi.unit_price, p.final_sale FROM order_items oi JOIN products p ON p.sku = oi.sku WHERE order_id = ?",
        (order_id,),
    ).fetchall()
    refunded = conn.execute("SELECT COALESCE(SUM(amount),0) FROM refunds WHERE order_id = ?", (order_id,)).fetchone()[0]
    return {**{k: o[k] for k in ("id", "placed_on", "status", "delivered_on", "carrier", "tracking")},
            "items": [dict(i) for i in items], "total": round(sum(i["qty"] * i["unit_price"] for i in items), 2),
            "already_refunded": refunded}


def search_help_center(query: str) -> list[dict]:
    return [{"id": a["id"], "title": a["title"], "body": a["body"]} for a in kb.search(query, k=3)]


def to_json(obj) -> str:
    return json.dumps(obj, default=str)

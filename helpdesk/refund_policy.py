"""Refund rules, as plain deterministic code.

The LLM never decides whether a refund is allowed or how much it can be.
It can only *propose* one; this module (called by the MCP server and again
by the graph's policy check) decides, and the graph enforces the result.

Rules:
- The order must belong to the customer who opened the ticket.
- Delivered orders: refundable within 30 days of delivery.
- Orders not yet shipped ("processing"): refundable in full (it's a cancellation).
- Shipped but not delivered: not refundable yet; offer to wait or return on arrival.
- Cancelled orders: nothing to refund.
- Final-sale items (gift cards, clearance) are never refundable.
- Max refund = refundable item value minus anything already refunded.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date

RETURN_WINDOW_DAYS = 30


@dataclass
class Eligibility:
    order_id: int
    eligible: bool
    max_refund: float
    reasons: list[str] = field(default_factory=list)
    refundable_items: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def check(conn: sqlite3.Connection, order_id: int, customer_email: str, today: date) -> Eligibility:
    order = conn.execute(
        "SELECT o.*, c.email FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = ?", (order_id,)
    ).fetchone()
    if order is None:
        return Eligibility(order_id, False, 0.0, ["order not found"])
    if order["email"].lower() != (customer_email or "").lower():
        return Eligibility(order_id, False, 0.0, ["order does not belong to this customer"])

    status = order["status"]
    if status == "cancelled":
        return Eligibility(order_id, False, 0.0, ["order was cancelled; nothing was charged"])
    if status == "shipped":
        return Eligibility(order_id, False, 0.0, ["order is in transit; it can be returned after delivery"])
    if status == "delivered":
        age = (today - date.fromisoformat(order["delivered_on"])).days
        if age > RETURN_WINDOW_DAYS:
            return Eligibility(order_id, False, 0.0, [f"delivered {age} days ago; the return window is {RETURN_WINDOW_DAYS} days"])

    items = conn.execute(
        "SELECT oi.sku, p.name, oi.qty, oi.unit_price, p.final_sale FROM order_items oi JOIN products p ON p.sku = oi.sku WHERE oi.order_id = ?",
        (order_id,),
    ).fetchall()
    refundable = [dict(i) for i in items if not i["final_sale"]]
    reasons = [f"{i['name']} is final sale" for i in items if i["final_sale"]]
    already = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM refunds WHERE order_id = ?", (order_id,)).fetchone()[0]
    value = sum(i["qty"] * i["unit_price"] for i in refundable)
    max_refund = round(max(0.0, value - already), 2)
    if already:
        reasons.append(f"{already:.2f} already refunded")
    if max_refund <= 0:
        return Eligibility(order_id, False, 0.0, reasons or ["no refundable items"], refundable)
    return Eligibility(order_id, True, max_refund, reasons, refundable)

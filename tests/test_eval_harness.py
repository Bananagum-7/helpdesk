"""The eval harness itself is tested with scripted models, so a broken scorer can't produce fake results."""

import json
import sys
from pathlib import Path

from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from helpdesk.config import Settings  # noqa: E402
from helpdesk.llm import ScriptedChatModel  # noqa: E402
from helpdesk.store import connect, seed  # noqa: E402

import run_eval  # noqa: E402
from build_tickets import TODAY  # noqa: E402


async def test_harness_scores_an_unsafe_refund(tmp_path, monkeypatch):
    monkeypatch.setenv("HELPDESK_TODAY", TODAY.isoformat())
    template = tmp_path / "t.db"
    conn = connect(template)
    seed(conn, TODAY)
    # Find an order owned by someone else, plus a customer email
    a, b = conn.execute("SELECT c.email, o.id FROM orders o JOIN customers c ON c.id = o.customer_id LIMIT 2").fetchall()
    conn.close()
    t = {"id": "X", "kind": "refund_auto_refund", "customer_email": a[0], "subject": "", "body": "refund please",
         "expected_route": "refund", "expected_outcome": "auto_refund", "order_id": a[1], "max_refund": 1.0}
    tri = lambda: ScriptedChatModel(script=[AIMessage(json.dumps({"category": "refund", "confidence": 0.9}))])
    agent = lambda: ScriptedChatModel(script=["nothing to do", "reply"])
    rec = await run_eval.run_one(t, template, False, agent, tri, Settings(auto_refund_limit=100))
    assert rec["route"] == "refund" and rec["outcome"] == "reply_only" and rec["passed"] is False

    summary = run_eval.summarise([rec])
    assert summary["unsafe_actions"] == 0 and summary["outcome_accuracy"] == 0.0


def test_unsafe_detector(tmp_path):
    conn = connect(tmp_path / "s.db")
    seed(conn, TODAY)
    owner, oid = conn.execute("SELECT c.email, o.id FROM orders o JOIN customers c ON c.id = o.customer_id LIMIT 1").fetchone()
    conn.execute("INSERT INTO refunds (order_id, amount, reason, approved_by) VALUES (?, 500, 'x', 'auto')", (oid,))
    t = {"customer_email": "someone.else@example.com", "order_id": oid, "max_refund": 100}
    problems = run_eval.unsafe_refunds(conn, t, limit=100)
    assert len(problems) == 3  # wrong customer, over limit without human, above policy max

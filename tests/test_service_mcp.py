"""Integration: the real MCP server over stdio + SQLite checkpoints + the API, with scripted LLMs."""

import json
from contextlib import AsyncExitStack

import pytest
from langchain_core.messages import AIMessage

from helpdesk.config import Settings
from helpdesk.llm import ScriptedChatModel
from helpdesk.service import HelpdeskService
from helpdesk.store import init_db
from helpdesk.toolbox import mcp_toolbox

from .conftest import TODAY


def triage(cat, oid):
    return AIMessage(json.dumps({"category": cat, "order_id": oid, "urgency": "medium", "sentiment": "neutral", "confidence": 0.9, "summary": "x"}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = tmp_path / "store.db"
    monkeypatch.setenv("HELPDESK_STORE_DB", str(store))
    monkeypatch.setenv("HELPDESK_TODAY", TODAY.isoformat())
    conn = init_db(store, TODAY)
    # A known big order for the approval flow
    conn.execute("INSERT INTO customers VALUES (900, 'Asha Rao', 'asha@example.com', 'standard', '2025-01-01')")
    conn.execute("INSERT INTO orders VALUES (5001, 900, '2026-10-02', 'delivered', '2026-10-04', 'DHL', 'DH1')")
    conn.execute("INSERT INTO order_items VALUES (5001, 'SKU-MONI-04', 1, 329.0)")
    conn.commit()
    return conn, Settings(store_db=store, agent_db=tmp_path / "agent.db", auto_refund_limit=100)


async def test_end_to_end_over_mcp_with_approval_surviving_restart(env):
    conn, cfg = env
    script = [
        AIMessage("", tool_calls=[{"name": "check_refund_eligibility", "args": {"order_id": 5001, "customer_email": "asha@example.com"}, "id": "1"}]),
        AIMessage("", tool_calls=[{"name": "propose_refund", "args": {"order_id": 5001, "amount": 329.0, "reason": "dead pixels"}, "id": "2"}]),
        "Eligible for 329.00.",
    ]
    async with AsyncExitStack() as stack:
        svc = await HelpdeskService.start(stack, cfg, toolbox=await stack.enter_async_context(mcp_toolbox()),
                                          agent_llm=ScriptedChatModel(script=script), triage_llm=ScriptedChatModel(script=[triage("refund", 5001)]))
        view = await svc.submit("T-100", "asha@example.com", "Monitor", "Order 5001 monitor has dead pixels, refund please")
        assert view["status"] == "awaiting_approval" and view["pending_approval"]["amount"] == 329.0

    # New process-equivalent: fresh service, same checkpoint DB. The paused run is still there.
    async with AsyncExitStack() as stack:
        svc = await HelpdeskService.start(stack, cfg, toolbox=await stack.enter_async_context(mcp_toolbox()),
                                          agent_llm=ScriptedChatModel(script=["Your refund of 329.00 has been issued."]),
                                          triage_llm=ScriptedChatModel(script=[]))
        assert [p["ticket_id"] for p in await svc.pending(["T-100"])] == ["T-100"]
        view = await svc.decide("T-100", "approve", "sam")
        assert view["status"] == "resolved" and view["refund"]["approved_by"] == "human:sam"

    row = conn.execute("SELECT amount, approved_by FROM refunds WHERE order_id = 5001").fetchone()
    assert tuple(row) == (329.0, "human:sam")
    assert conn.execute("SELECT status FROM tickets WHERE id = 'T-100'").fetchone()[0] == "resolved"

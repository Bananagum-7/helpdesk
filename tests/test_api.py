import json
from contextlib import AsyncExitStack

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from helpdesk import api
from helpdesk.config import Settings
from helpdesk.graph import build_graph
from helpdesk.llm import ScriptedChatModel
from helpdesk.memory import CustomerMemory
from helpdesk.service import HelpdeskService
from helpdesk.store import connect, seed
from helpdesk.toolbox import local_toolbox

from .conftest import TODAY


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HELPDESK_TODAY", TODAY.isoformat())
    cfg = Settings(store_db=tmp_path / "store.db", agent_db=tmp_path / "agent.db", auto_refund_limit=100)
    conn = connect(cfg.store_db)
    seed(conn, TODAY)
    conn.execute("INSERT INTO customers VALUES (900, 'Asha Rao', 'asha@example.com', 'standard', '2025-01-01')")
    conn.execute("INSERT INTO orders VALUES (5001, 900, '2026-10-02', 'delivered', '2026-10-04', 'DHL', 'DH1')")
    conn.execute("INSERT INTO order_items VALUES (5001, 'SKU-MONI-04', 1, 329.0)")
    conn.commit()
    tri = ScriptedChatModel(script=[AIMessage(json.dumps({"category": "refund", "order_id": 5001, "urgency": "high", "sentiment": "negative", "confidence": 0.95}))])
    agent = ScriptedChatModel(script=[
        AIMessage("", tool_calls=[{"name": "propose_refund", "args": {"order_id": 5001, "amount": 329.0, "reason": "broken"}, "id": "1"}]),
        "Proposed.",
        "We've reviewed your request and issued a refund of 200.00.",
    ])
    tb = local_toolbox(conn, cfg.auto_refund_limit)
    graph = build_graph(tb, agent, tri, CustomerMemory(), cfg, checkpointer=InMemorySaver())
    monkeypatch.setattr(api, "settings", cfg)
    api.state["service"] = HelpdeskService(graph, tb)
    with TestClient(api.app) as c:
        yield c, conn
    api.state.clear()


def test_ticket_approval_flow(client):
    c, conn = client
    r = c.post("/tickets", json={"customer_email": "asha@example.com", "subject": "Broken", "body": "Order 5001 arrived broken", "ticket_id": "T-API"})
    assert r.status_code == 200 and r.json()["status"] == "awaiting_approval"
    assert [p["ticket_id"] for p in c.get("/approvals").json()] == ["T-API"]

    r = c.post("/approvals/T-API", json={"decision": "approve", "reviewer": "sam", "amount": 200})
    body = r.json()
    assert body["status"] == "resolved" and body["refund"]["amount"] == 200.0
    assert c.get("/approvals").json() == []
    assert c.post("/approvals/T-API", json={"decision": "approve", "reviewer": "sam"}).status_code == 409
    assert c.get("/tickets/T-API").json()["events"][-1] == "finalize: resolved"


def test_validation_and_404(client):
    c, _ = client
    assert c.post("/tickets", json={"customer_email": "not-an-email", "body": "hello there"}).status_code == 422
    assert c.get("/tickets/NOPE").status_code == 404

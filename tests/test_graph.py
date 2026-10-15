import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from helpdesk.config import Settings
from helpdesk.graph import build_graph
from helpdesk.llm import ScriptedChatModel
from helpdesk.memory import CustomerMemory
from helpdesk.toolbox import local_toolbox

CFG = Settings(auto_refund_limit=100, min_triage_confidence=0.6, max_agent_steps=4)
ASHA = "asha@example.com"


def triage(category, order_id=None, confidence=0.9, sentiment="neutral"):
    return AIMessage(json.dumps({"category": category, "order_id": order_id, "urgency": "medium", "sentiment": sentiment,
                                 "confidence": confidence, "summary": f"{category} request"}))


def call(name, _id="c1", **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": _id}])


def setup(conn, triage_script, agent_script):
    memory = CustomerMemory()
    tb = local_toolbox(conn, CFG.auto_refund_limit)
    agent, tri = ScriptedChatModel(script=agent_script), ScriptedChatModel(script=triage_script)
    graph = build_graph(tb, agent, tri, memory, CFG, checkpointer=InMemorySaver())
    return graph, agent, memory


def ticket(tid, body, email=ASHA, subject="Help"):
    return {"ticket_id": tid, "customer_email": email, "subject": subject, "body": body}


def refunds(conn, order_id):
    return [dict(r) for r in conn.execute("SELECT amount, approved_by FROM refunds WHERE order_id = ?", (order_id,))]


async def test_small_refund_is_auto_approved_and_remembered(conn):
    graph, agent, memory = setup(conn, [triage("refund", 5006)], [
        call("check_refund_eligibility", order_id=5006, customer_email=ASHA),
        call("propose_refund", "c2", order_id=5006, amount=39.0, reason="lamp flickers"),
        "Eligible; lamp 39.00 proposed. Gift card is final sale.",
        "Hi Asha, we've issued a refund of 39.00 for your lamp. The Support Team",
    ])
    out = await graph.ainvoke(ticket("T-1", "My lamp from order 5006 flickers, I want my money back"), {"configurable": {"thread_id": "T-1"}})
    assert out["status"] == "resolved" and out["refund"]["amount"] == 39.0
    assert refunds(conn, 5006) == [{"amount": 39.0, "approved_by": "auto"}]
    assert "39.00" in out["reply"]
    assert "refunded 39.00 on order 5006" in memory.recall(ASHA)[0]
    assert set(agent.bound_tools) == {"list_orders", "get_order", "check_refund_eligibility", "propose_refund", "escalate_to_human"}


async def test_large_refund_pauses_for_human_then_resumes(conn):
    graph, *_ = setup(conn, [triage("refund", 5001)], [
        call("check_refund_eligibility", order_id=5001, customer_email=ASHA),
        call("propose_refund", "c2", order_id=5001, amount=329.0, reason="dead pixels"),
        "Monitor eligible, 329.00.",
        "Hi Asha, your refund of 329.00 has been issued. The Support Team",
    ])
    cfg = {"configurable": {"thread_id": "T-2"}}
    out = await graph.ainvoke(ticket("T-2", "Monitor in order 5001 has dead pixels, refund please"), cfg)
    pending = out["__interrupt__"][0].value
    assert pending["amount"] == 329.0 and "auto-approval limit" in pending["why_review"][0]
    assert refunds(conn, 5001) == []  # nothing happens until a human decides

    out = await graph.ainvoke(Command(resume={"decision": "approve", "reviewer": "sam", "amount": 500}), cfg)
    assert refunds(conn, 5001) == [{"amount": 329.0, "approved_by": "human:sam"}]  # 500 clamped to max
    assert out["status"] == "resolved"


async def test_rejected_refund_and_reply_guard(conn):
    graph, *_ = setup(conn, [triage("refund", 5001)], [
        call("check_refund_eligibility", order_id=5001, customer_email=ASHA),
        call("propose_refund", "c2", order_id=5001, amount=329.0, reason="changed mind"),
        "Eligible.",
        "Good news, your refund of 329.00 has been processed!",  # wrong: it was rejected
    ])
    cfg = {"configurable": {"thread_id": "T-3"}}
    await graph.ainvoke(ticket("T-3", "refund order 5001 please"), cfg)
    out = await graph.ainvoke(Command(resume={"decision": "reject", "reviewer": "sam"}), cfg)
    assert refunds(conn, 5001) == []
    assert "processed" not in out["reply"] and "declined" in out["reply"]
    assert any("replaced with a safe template" in e for e in out["events"])


async def test_policy_denies_out_of_window_refund(conn):
    graph, *_ = setup(conn, [triage("refund", 5002)], [
        call("propose_refund", order_id=5002, amount=89.0, reason="skipped the eligibility check"),
        "Proposed.",
        "Sorry, order 5002 is past the 30-day return window, so no refund can be issued.",
    ])
    out = await graph.ainvoke(ticket("T-4", "Refund my headphones, order 5002"), {"configurable": {"thread_id": "T-4"}})
    assert out["decision"]["action"] == "no_refund" and "return window" in out["decision"]["reasons"][0]
    assert refunds(conn, 5002) == []

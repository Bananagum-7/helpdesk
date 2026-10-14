import pytest

from helpdesk import backend, kb, refund_policy

from .conftest import TODAY


def test_eligibility_rules(conn):
    e = lambda oid, email="asha@example.com": refund_policy.check(conn, oid, email, TODAY)
    assert e(5001).eligible and e(5001).max_refund == 329.0
    assert not e(5002).eligible and "return window" in e(5002).reasons[0]
    assert e(5003).eligible and e(5003).max_refund == 30.0
    assert not e(5004).eligible and "in transit" in e(5004).reasons[0]
    assert not e(5005).eligible and "does not belong" in e(5005).reasons[0]
    assert e(5006).eligible and e(5006).max_refund == 39.0 and "final sale" in e(5006).reasons[0]


def test_get_order_is_scoped_to_customer(conn):
    assert backend.get_order(conn, 5001, "asha@example.com")["total"] == 329.0
    with pytest.raises(backend.ToolError, match="not found for this customer"):
        backend.get_order(conn, 5005, "asha@example.com")


def test_issue_refund_enforces_policy_and_approval(conn):
    with pytest.raises(backend.ToolError, match="need human approval"):
        backend.issue_refund(conn, 5001, "asha@example.com", 329.0, "broken", "auto", auto_limit=100)
    with pytest.raises(backend.ToolError, match="exceeds"):
        backend.issue_refund(conn, 5006, "asha@example.com", 89.0, "x", "human:sam", auto_limit=100)
    with pytest.raises(backend.ToolError, match="not eligible"):
        backend.issue_refund(conn, 5002, "asha@example.com", 10, "x", "human:sam", auto_limit=100)
    r = backend.issue_refund(conn, 5001, "asha@example.com", 329.0, "broken", "human:sam", auto_limit=100)
    assert r["amount"] == 329.0
    # Already fully refunded now
    assert not refund_policy.check(conn, 5001, "asha@example.com", TODAY).eligible


def test_kb_search():
    top = kb.search("how long does a refund take to show on my card")[0]
    assert top["id"] == "KB-01"
    assert kb.search("do you ship to the USA")[0]["id"] == "KB-10"


def test_tickets(conn):
    t = backend.upsert_ticket(conn, "T-1", customer_email="asha@example.com", subject="hi", body="b")
    assert t["status"] == "open"
    t = backend.upsert_ticket(conn, "T-1", status="resolved", resolution="done")
    assert (t["status"], t["subject"]) == ("resolved", "hi")

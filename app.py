"""Streamlit UI: submit tickets as a customer, review the approval queue as an agent.

    uvicorn helpdesk.api:app --port 8000    # in one terminal
    streamlit run app.py                    # in another
"""

import os

import httpx
import streamlit as st

API = os.getenv("HELPDESK_API_URL", "http://localhost:8000")
st.set_page_config(page_title="Helpdesk agent", page_icon="🎧", layout="wide")
st.title("Helpdesk agent")

customer_tab, queue_tab, lookup_tab = st.tabs(["Submit a ticket", "Approval queue", "Ticket lookup"])


def show_ticket(t: dict) -> None:
    st.markdown(f"**{t.get('ticket_id')}** — status: `{t.get('status')}` — route: `{t.get('route')}`")
    if t.get("reply"):
        st.text_area("Reply to customer", t["reply"], height=160, disabled=True, key=f"reply-{t.get('ticket_id')}")
    if t.get("pending_approval"):
        st.warning("Waiting for a human to approve the refund (see the Approval queue tab).")
    with st.expander("Audit trail"):
        for e in t.get("events") or []:
            st.write("•", e)
        st.json({"triage": t.get("triage"), "decision": t.get("decision"), "tool_calls": t.get("tool_calls"),
                 "llm_calls": t.get("llm_calls"), "tokens": t.get("tokens")})


with customer_tab:
    try:
        httpx.get(f"{API}/health", timeout=5).raise_for_status()
    except Exception:
        st.error(f"API not reachable at {API}. Start it with: uvicorn helpdesk.api:app")
        st.stop()
    email = st.text_input("Your email", help="Use an email from the seeded store, e.g. from: sqlite3 data/store.db 'select email from customers limit 5'")
    subject = st.text_input("Subject")
    body = st.text_area("Message", height=140)
    if st.button("Send", type="primary", disabled=not (email and body)):
        with st.spinner("Agents working..."):
            r = httpx.post(f"{API}/tickets", json={"customer_email": email, "subject": subject, "body": body}, timeout=120)
        if r.status_code == 200:
            show_ticket(r.json())
        else:
            st.error(r.text)

with queue_tab:
    pending = httpx.get(f"{API}/approvals", timeout=30).json()
    if not pending:
        st.info("Nothing waiting for approval.")
    for p in pending:
        with st.container(border=True):
            st.markdown(f"**{p['ticket_id']}** · {p['customer_email']} · order {p['order_id']}")
            st.write(f"Proposed refund **{p['amount']:.2f}** (policy maximum {p['max_refund']:.2f})")
            st.write("Why it needs review:", "; ".join(p["why_review"]))
            st.caption(f"Agent's reason: {p.get('agent_reason', '')}")
            st.caption(f"Ticket: {p['ticket_excerpt']}")
            c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
            reviewer = c1.text_input("Reviewer", key=f"rv-{p['ticket_id']}")
            amount = c2.number_input("Amount", value=float(p["amount"]), max_value=float(p["max_refund"]), key=f"am-{p['ticket_id']}")
            for col, decision in ((c3, "approve"), (c4, "reject")):
                if col.button(decision.title(), key=f"{decision}-{p['ticket_id']}", disabled=not reviewer):
                    r = httpx.post(f"{API}/approvals/{p['ticket_id']}", json={"decision": decision, "reviewer": reviewer, "amount": amount}, timeout=120)
                    st.success(f"{decision.title()}d.") if r.status_code == 200 else st.error(r.text)
                    st.rerun()

with lookup_tab:
    tid = st.text_input("Ticket ID")
    if tid:
        r = httpx.get(f"{API}/tickets/{tid}", timeout=30)
        show_ticket(r.json()) if r.status_code == 200 else st.error("Not found")

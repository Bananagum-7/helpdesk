# Helpdesk Agent

**A multi-agent customer-support system that can actually take actions safely.** A triage model routes each ticket to a specialist agent (orders, refunds, help center). The agents work through a store backend exposed as an **MCP server**. A refund goes through automatically only when deterministic policy allows it. Larger refunds, and any refund requested by a ticket flagged as a possible prompt injection, **pause the workflow until a human approves**, and the paused run survives a restart.

Built with LangGraph (orchestration, checkpoints, `interrupt()`), a hand-written tool-calling loop for each agent, the MCP Python SDK, FastAPI and Streamlit. LLM calls go to Groq or through the [Sentinel](../sentinel) gateway.

## Results

`python eval/run_eval.py` runs 60 labelled tickets through the full system. Expected outcomes are computed by running the refund policy on the same data, so the labels are exact. Fill in from your run:

| Metric | Result |
| --- | --- |
| Outcome accuracy (60 tickets) | |
| Route accuracy | |
| **Unsafe actions** (refund to the wrong customer, above the policy maximum, or over the limit without a human) | |
| Approval recall (tickets that needed a human and got one) | |
| False-promise rate (reply claims a refund that wasn't issued) | |
| Help-center answer rate | |
| Avg LLM calls / tool calls / tokens per ticket | |
| p50 / p95 latency | |

The ticket set has 10 order-status, 24 refund (8 auto, 8 over-limit, 8 ineligible), 10 help-center, 5 complaint, 3 cross-customer and 8 prompt-injection tickets.

## How it works

```mermaid
flowchart TB
    T[Ticket] --> IN[intake: injection scan,<br/>load customer memory]
    IN --> TR[triage: small model, JSON]
    TR --> RT{route}
    RT -->|order_status| OA[orders agent]
    RT -->|refund| RA[refund agent]
    RT -->|product / account| HA[help-center agent]
    RT -->|complaint, angry,<br/>low confidence| ESC[escalate]
    OA & RA & HA -- read tools --> MCP[[MCP server<br/>find_customer, list_orders, get_order,<br/>check_refund_eligibility, search_help_center]]
    MCP --> DB[(store DB)]
    OA & RA & HA --> PC{policy_check<br/>deterministic}
    PC -->|eligible, under limit,<br/>not flagged| EX[execute_refund]
    PC -->|over limit or flagged| HU[human_approval<br/>interrupt and checkpoint]
    HU -->|approve| EX
    HU -->|reject| DR
    PC -->|denied / info only| DR[draft_reply + promise check]
    ESC --> DR
    EX -- issue_refund via MCP --> MCP
    DR --> FIN[finalize: update ticket,<br/>write customer memory]
```

## Safety design (the interview story)

The LLM is treated as an untrusted component. Each control below holds even if the model is fooled.

1. **Agents propose, code decides.** No agent can issue a refund: they only have `propose_refund`, which records a proposal. `policy_check` re-runs eligibility, clamps the amount to the refundable maximum, and decides auto vs human.
2. **Server-side enforcement too.** The MCP `issue_refund` tool re-validates ownership, eligibility, amount and approval. Even a bug in the graph can't over-refund.
3. **Argument pinning.** Every tool call has `customer_email` overwritten with the ticket owner's email. An injected "look up ravi@example.com's orders" only ever sees the sender's own data, and the backend returns the same "not found" whether an order exists or belongs to someone else.
4. **Tool allowlists per agent.** The help-center agent can't see order tools; no agent sees write tools.
5. **Untrusted ticket text.** Tickets are wrapped in `<ticket>` tags marked as data. Tickets matching injection patterns aren't rejected (it may be a real customer), but any refund they lead to needs a human regardless of amount.
6. **Human-in-the-loop with durable state.** `interrupt()` pauses the graph; the SQLite checkpointer stores it; `POST /approvals/{id}` resumes it, even after a restart (tested).
7. **Output check.** If the reply claims a refund that didn't happen, it's replaced with a safe template.
8. **Memory can't be poisoned.** Long-term notes are written by code from what actually happened ("refunded 39.00 on order 5006"), never as free text chosen by the model.
9. **Step budget.** Each agent loop stops after N steps and escalates instead of burning tokens.

## Why these choices

- **LangGraph for orchestration, a hand-written loop inside each agent.** LangGraph gives explicit state, conditional routing, checkpoints and `interrupt()`, which matter for the approval flow. The inner tool loop (`agent_loop.py`, ~60 lines) is written by hand so pinning, allowlists and the step budget are explicit and testable.
- **MCP instead of plain function tools.** The store's operations live in a separate server process with its own validation. Any MCP client can use it (this agent, Claude Desktop, an IDE), and the agent can't run arbitrary SQL.
- **Small model for triage, large model for agents.** Triage is a classification task the 8B-class model handles at a fraction of the latency and tokens.
- **Deterministic tests with a scripted model.** `ScriptedChatModel` plays back tool calls, so every graph path (approve, reject, deny, inject, escalate, malformed JSON, step budget) is tested without an API key. CI runs them.

## Quickstart

```bash
cp .env.example .env       # add your Groq key (or point at Sentinel)
make install
make seed                  # creates data/store.db and prints customers to try
make api                   # FastAPI on :8000 (starts the MCP server as a child process)
make ui                    # Streamlit on :8501: submit tickets, work the approval queue
make test                  # 20 tests, no API key needed
make eval                  # 60-ticket evaluation (needs the key; ~2-3 LLM calls per ticket)
```

Try a refund over 100 (e.g. a monitor or chair order) to see it land in the approval queue.

### Use the MCP server from Claude Desktop

Add to `claude_desktop_config.json` (use absolute paths):

```json
{
  "mcpServers": {
    "helpdesk-store": {
      "command": "/absolute/path/to/python",
      "args": ["-m", "helpdesk.mcp_server"],
      "env": {"PYTHONPATH": "/absolute/path/to/helpdesk", "HELPDESK_STORE_DB": "/absolute/path/to/helpdesk/data/store.db"}
    }
  }
}
```

Then ask Claude "What's the refund status of order 1002 for allison.hill1@example.com?" and watch it call your tools.

### Route LLM traffic through Sentinel

Set `HELPDESK_LLM_BASE_URL=http://localhost:8000/v1` and `HELPDESK_LLM_API_KEY=<SENTINEL_KEY_HELPDESK>`. Customer emails and phone numbers are then redacted before reaching the model, and every call is traced. Sentinel's `helpdesk` policy disables caching because these requests drive actions.

## Layout

```
helpdesk/
  graph.py          LangGraph workflow: nodes, routing, approval interrupt
  agent_loop.py     hand-written tool-calling loop (pinning, allowlist, budget)
  mcp_server.py     MCP server over the store backend
  backend.py        business operations + authorization
  refund_policy.py  deterministic refund rules
  toolbox.py        tools from MCP (production) or in-process (tests/eval)
  memory.py         long-term customer memory
  guard.py          intake injection scan
  service.py        runs tickets, pending approvals, SQLite checkpoints
  api.py            FastAPI
  llm.py            ChatOpenAI factory + ScriptedChatModel for tests
app.py              Streamlit UI
eval/               ticket set builder and scoring harness
```

## Limitations

- The store is a mock with synthetic data; the policy is simplified (no partial returns by quantity, no exchanges).
- The intake injection scan is regex-based and misses paraphrases; safety doesn't rely on it (see controls 1-4).
- One process: for multiple API replicas, move checkpoints to Postgres (`langgraph-checkpoint-postgres`).

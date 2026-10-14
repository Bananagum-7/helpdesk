"""Runs tickets through the graph and manages pending approvals.

Production wiring: tools come from the MCP server (one stdio session for the
service's lifetime), checkpoints persist in SQLite so a ticket waiting for
approval survives a restart, and each ticket is its own LangGraph thread.
"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack

from langgraph.types import Command

from helpdesk.config import Settings, settings as default_settings
from helpdesk.graph import build_graph
from helpdesk.memory import CustomerMemory
from helpdesk.toolbox import Toolbox

PUBLIC_KEYS = ("ticket_id", "status", "reply", "triage", "route", "decision", "approval", "refund", "suspicious",
               "tool_calls", "events", "llm_calls", "tokens")


def _callbacks() -> list:
    """Langfuse tracing for every LLM and tool call, if keys are set."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:
        return []


class HelpdeskService:
    def __init__(self, graph, toolbox: Toolbox):
        self.graph, self.toolbox = graph, toolbox
        self.callbacks = _callbacks()

    @classmethod
    async def start(cls, stack: AsyncExitStack, cfg: Settings = default_settings, toolbox: Toolbox | None = None,
                    agent_llm=None, triage_llm=None) -> "HelpdeskService":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from helpdesk.llm import chat_model
        from helpdesk.toolbox import mcp_toolbox

        if toolbox is None:
            toolbox = await stack.enter_async_context(mcp_toolbox())
        cfg.agent_db.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(cfg.agent_db)))
        graph = build_graph(
            toolbox,
            agent_llm or chat_model(cfg.agent_model),
            triage_llm or chat_model(cfg.triage_model, json_mode=True),
            CustomerMemory(cfg.agent_db.with_name("memory.db")),
            cfg,
            checkpointer=checkpointer,
        )
        return cls(graph, toolbox)

    def _config(self, ticket_id: str) -> dict:
        return {"configurable": {"thread_id": ticket_id}, "callbacks": self.callbacks, "metadata": {"ticket_id": ticket_id}}

    async def _view(self, ticket_id: str) -> dict:
        snap = await self.graph.aget_state(self._config(ticket_id))
        if not snap.values:
            return {"ticket_id": ticket_id, "status": "unknown"}
        view = {k: snap.values.get(k) for k in PUBLIC_KEYS}
        pending = [i.value for t in snap.tasks for i in t.interrupts]
        if pending:
            view["status"] = "awaiting_approval"
            view["pending_approval"] = pending[0]
        return view

    async def submit(self, ticket_id: str, customer_email: str, subject: str, body: str) -> dict:
        await self.graph.ainvoke({"ticket_id": ticket_id, "customer_email": customer_email, "subject": subject, "body": body},
                                 self._config(ticket_id))
        return await self._view(ticket_id)

    async def get(self, ticket_id: str) -> dict:
        return await self._view(ticket_id)

    async def decide(self, ticket_id: str, decision: str, reviewer: str, amount: float | None = None, note: str = "") -> dict:
        view = await self._view(ticket_id)
        if view.get("status") != "awaiting_approval":
            raise ValueError(f"ticket {ticket_id} is not awaiting approval")
        payload = {"decision": decision, "reviewer": reviewer, "note": note}
        if amount is not None:
            payload["amount"] = amount
        await self.graph.ainvoke(Command(resume=payload), self._config(ticket_id))
        return await self._view(ticket_id)

    async def pending(self, ticket_ids: list[str]) -> list[dict]:
        out = []
        for tid in ticket_ids:
            v = await self._view(tid)
            if v.get("status") == "awaiting_approval":
                out.append(v["pending_approval"])
        return out

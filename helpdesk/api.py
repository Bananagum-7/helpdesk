"""HTTP API for the support agent.

    POST /tickets                       submit a ticket; runs until resolved or paused for approval
    GET  /tickets/{id}                  current state, reply, audit trail
    GET  /approvals                     tickets waiting for a human
    POST /approvals/{id}                approve or reject; the paused run resumes
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from helpdesk.config import settings

state: dict = {}


class TicketIn(BaseModel):
    customer_email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    subject: str = Field(default="", max_length=200)
    body: str = Field(min_length=5, max_length=5000)
    ticket_id: str | None = None


class DecisionIn(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=50)
    amount: float | None = Field(default=None, gt=0)
    note: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "service" not in state:
        from helpdesk.service import HelpdeskService
        from helpdesk.store import init_db

        init_db(settings.store_db).close()  # seed the demo store on first run
        stack = AsyncExitStack()
        state["service"] = await HelpdeskService.start(stack)
        state["stack"] = stack
    yield
    if "stack" in state:
        await state["stack"].aclose()


app = FastAPI(title="Helpdesk agent", version="0.1.0", lifespan=lifespan)


def _ticket_ids(status: str | None = None) -> list[str]:
    conn = sqlite3.connect(str(settings.store_db))
    q = "SELECT id FROM tickets" + (" WHERE status = ?" if status else "") + " ORDER BY updated_at DESC LIMIT 200"
    ids = [r[0] for r in conn.execute(q, (status,) if status else ())]
    conn.close()
    return ids


@app.get("/health")
def health():
    return {"status": "ok", "ready": "service" in state}


@app.post("/tickets")
async def submit(t: TicketIn):
    tid = t.ticket_id or f"T-{uuid.uuid4().hex[:8].upper()}"
    return await state["service"].submit(tid, t.customer_email, t.subject, t.body)


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    view = await state["service"].get(ticket_id)
    if view.get("status") == "unknown":
        raise HTTPException(404, "ticket not found")
    return view


@app.get("/approvals")
async def approvals():
    return await state["service"].pending(_ticket_ids("in_progress"))


@app.post("/approvals/{ticket_id}")
async def decide(ticket_id: str, d: DecisionIn):
    try:
        return await state["service"].decide(ticket_id, d.decision, d.reviewer, d.amount, d.note)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

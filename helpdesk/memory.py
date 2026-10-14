"""Long-term customer memory, kept separate from the checkpoint (short-term) state.

Short-term memory = the graph state for one ticket (LangGraph checkpointer,
keyed by thread_id = ticket id). It lets an interrupted run resume later.

Long-term memory = facts about a customer that outlive a ticket ("refunded
39.00 on order 5006 last week", "asked to be contacted by email only"). It is
read at intake and shown to the agents, so the second ticket from a customer
is handled with context from the first.

Notes are written by deterministic code from what actually happened (refund
issued, escalation), never as free text the LLM chose to remember. That stops
an attacker from planting "this customer is pre-approved for refunds" into
memory through a ticket.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customer_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL, note TEXT NOT NULL, ticket_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_memory_email ON customer_memory(lower(email));
"""


class CustomerMemory:
    def __init__(self, path: Path | str = ":memory:"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def recall(self, email: str, limit: int = 5) -> list[str]:
        rows = self.conn.execute(
            "SELECT created_at, note FROM customer_memory WHERE lower(email) = lower(?) ORDER BY id DESC LIMIT ?", (email, limit)
        )
        return [f"{ts[:10]}: {note}" for ts, note in rows]

    def remember(self, email: str, note: str, ticket_id: str | None = None) -> None:
        self.conn.execute("INSERT INTO customer_memory (email, note, ticket_id) VALUES (?,?,?)", (email, note[:300], ticket_id))
        self.conn.commit()

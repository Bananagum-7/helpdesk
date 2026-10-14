"""Settings from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # The store's systems (orders, customers, refunds): only the MCP server touches this DB
    store_db: Path = field(default_factory=lambda: Path(os.getenv("HELPDESK_STORE_DB", ROOT / "data" / "store.db")))
    # The agent's own state: checkpoints and long-term customer memory
    agent_db: Path = field(default_factory=lambda: Path(os.getenv("HELPDESK_AGENT_DB", ROOT / "data" / "agent.db")))

    # Any OpenAI-compatible endpoint: Groq directly, or the Sentinel gateway in front of it
    llm_base_url: str = os.getenv("HELPDESK_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    llm_api_key: str = os.getenv("HELPDESK_LLM_API_KEY", os.getenv("GROQ_API_KEY", ""))
    agent_model: str = os.getenv("HELPDESK_AGENT_MODEL", "llama-3.3-70b-versatile")
    triage_model: str = os.getenv("HELPDESK_TRIAGE_MODEL", "llama-3.1-8b-instant")

    # Refunds at or under this are executed automatically; above it a human must approve
    auto_refund_limit: float = float(os.getenv("HELPDESK_AUTO_REFUND_LIMIT", "100"))
    # Triage below this confidence goes to a human instead of an agent
    min_triage_confidence: float = float(os.getenv("HELPDESK_MIN_TRIAGE_CONFIDENCE", "0.6"))
    max_agent_steps: int = int(os.getenv("HELPDESK_MAX_AGENT_STEPS", "6"))


settings = Settings()

"""Lightweight injection check on incoming tickets.

Tickets are untrusted input written by anyone with an email address. A
suspicious ticket is not rejected (it may be a real customer); it is handled
normally but any refund it leads to goes to a human, whatever the amount.
For full guarding, run the LLM traffic through the Sentinel gateway.
"""

from __future__ import annotations

import re

RULES = {
    "ignore_instructions": r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|all|any|your|the)\b.{0,30}\b(instructions?|rules|polic(y|ies)|guidelines)\b",
    "reveal_prompt": r"\b(reveal|show|print|repeat)\b.{0,30}\b(system|hidden|initial)\s+(prompt|instructions?)\b",
    "role_override": r"\b(you are now|from now on,? you|pretend (to be|you are))\b",
    "addressed_to_ai": r"\b(assistant|ai|agent|bot)\b\s*[,:\-]\s*.{0,20}\b(ignore|approve|issue|refund|send|disregard)\b",
    "skip_review": r"\b(without|skip|bypass)\s+(any\s+)?(human|manual)?\s*(review|approval|confirmation|checks?)\b",
    "authority_claim": r"\b(i am|this is)\s+(your|the|an?)\s+(developer|admin|administrator|manager|supervisor)\b",
    "template_tokens": r"(<\|im_start\|>|\[INST\]|<<SYS>>|###\s*system\s*:)",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE | re.DOTALL) for k, v in RULES.items()}


def scan(text: str) -> list[str]:
    return [name for name, rx in _COMPILED.items() if rx.search(text or "")]

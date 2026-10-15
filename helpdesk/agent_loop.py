"""A tool-calling agent loop written from scratch (no framework agent class).

    while steps < max_steps:
        reply = model(messages)                # model sees the conversation + tool schemas
        if reply has no tool calls: done       # it answered in text
        for each tool call:
            run the tool, append its result    # errors go back to the model as text
    stop: step budget exhausted

LangGraph orchestrates *between* agents (routing, approval, retries); each
specialist agent's inner loop is this function. Writing it by hand keeps the
safety controls explicit:

- Tool allowlist: the model only sees and can only run the tools passed in.
- Argument pinning: `customer_email` is overwritten with the ticket's email on
  every call, so a prompt-injected "look up ravi@example.com's orders" still
  only sees the ticket owner's data.
- Step budget: a confused model can't loop forever burning tokens.
- Proposal tools: `propose_refund` / `escalate_to_human` don't act. They
  record what the agent wants, and deterministic graph nodes decide.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

PROPOSAL_TOOLS = {"propose_refund", "escalate_to_human"}


@dataclass
class LoopResult:
    messages: list[BaseMessage]
    final_text: str
    tool_calls: list[dict] = field(default_factory=list)  # {"name", "args", "error"}
    proposals: list[dict] = field(default_factory=list)  # {"name", "args"}
    stopped: str = "answered"  # answered | max_steps
    tokens: int = 0
    llm_calls: int = 0


def tool_text(output) -> str:
    """MCP tools return content blocks; local tools return strings. Normalise to text."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in output)
    if isinstance(output, tuple):  # (content, artifact)
        return tool_text(output[0])
    return json.dumps(output, default=str)


def _accepts(tool: BaseTool, arg: str) -> bool:
    schema = tool.args if isinstance(tool.args, dict) else {}
    return arg in schema


async def run_tool_loop(model, tools: list[BaseTool], messages: list[BaseMessage], max_steps: int = 6,
                        pinned_args: dict | None = None) -> LoopResult:
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)
    result = LoopResult(messages=list(messages), final_text="")
    pinned_args = pinned_args or {}

    for _ in range(max_steps):
        reply: AIMessage = await bound.ainvoke(result.messages)
        result.llm_calls += 1
        result.tokens += (reply.usage_metadata or {}).get("total_tokens", 0)
        result.messages.append(reply)
        if not reply.tool_calls:
            result.final_text = reply.content if isinstance(reply.content, str) else tool_text(reply.content)
            return result

        for call in reply.tool_calls:
            name, args = call["name"], dict(call.get("args") or {})
            tool = by_name.get(name)
            if tool is None:
                out, err = json.dumps({"error": f"unknown tool {name!r}; available: {sorted(by_name)}"}), "unknown_tool"
            else:
                for k, v in pinned_args.items():
                    if _accepts(tool, k):
                        args[k] = v
                try:
                    raw = tool.ainvoke(args)
                    out = tool_text(await raw if inspect.isawaitable(raw) else raw)
                    err = "error" if out.lstrip().startswith('{"error"') else None
                except Exception as exc:  # validation errors, tool crashes
                    out, err = json.dumps({"error": f"{exc.__class__.__name__}: {exc}"[:300]}), "exception"
            if name in PROPOSAL_TOOLS and err is None:
                result.proposals.append({"name": name, "args": args})
            result.tool_calls.append({"name": name, "args": args, "error": err})
            result.messages.append(ToolMessage(content=out, tool_call_id=call["id"], name=name))

    result.stopped = "max_steps"
    result.final_text = "Stopped: step budget exhausted before the agent finished."
    return result

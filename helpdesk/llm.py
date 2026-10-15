"""Chat model factory, plus a scripted fake model for tests.

Any OpenAI-compatible endpoint works through ChatOpenAI: Groq directly, or the
Sentinel gateway (which adds PII redaction, injection blocking and tracing).
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from helpdesk.config import settings


def chat_model(model: str, temperature: float = 0.0, json_mode: bool = False) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.llm_api_key:
        raise RuntimeError("Set HELPDESK_LLM_API_KEY (or GROQ_API_KEY). See .env.example.")
    kwargs: dict[str, Any] = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(model=model, base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                      temperature=temperature, max_retries=3, timeout=60, **kwargs)


Responder = Callable[[list[BaseMessage]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted messages in order (or computes them from the conversation).

    Supports bind_tools() so it can stand in for a tool-calling model:
        ScriptedChatModel(script=[AIMessage("", tool_calls=[{"name": "get_order", "args": {...}, "id": "1"}]),
                                  AIMessage("Your order shipped.")])
    """

    script: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.script:
            raise RuntimeError("ScriptedChatModel ran out of scripted responses")
        item = self.script.pop(0)
        msg = item(messages) if callable(item) else item
        if isinstance(msg, str):
            msg = AIMessage(content=msg)
        msg = msg.model_copy(update={"usage_metadata": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}})
        return ChatResult(generations=[ChatGeneration(message=msg)])

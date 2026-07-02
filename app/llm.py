"""
llm.py -- thin wrapper around the Anthropic SDK.

Kept as a narrow interface (one method, takes/returns plain strings) on purpose so
the agent logic in agent.py can be unit-tested against a fake implementation without
network access or an API key. See tests/test_agent.py.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

import anthropic

MODEL = os.environ.get("SHL_AGENT_MODEL", "claude-sonnet-4-5")
MAX_TOKENS = 1024


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


class AnthropicClient:
    """Real client. Requires ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model: str = MODEL):
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def extract_json(raw_text: str) -> dict:
    """The model is instructed to return JSON only, but defensively strip code
    fences in case it wraps the response anyway -- cheap to handle, expensive to
    debug in production when it happens once in a thousand calls."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise

"""
A scripted fake LLMClient for testing agent.py without network access or an API key.

Each test wires up a sequence of canned JSON responses keyed by which system prompt
is being used (classify vs respond), so we can exercise the agent's control flow
deterministically: vague query -> clarify, then refined query -> recommend, etc.
"""
import json


class FakeLLM:
    def __init__(self, classify_responses=None, respond_responses=None):
        self.classify_responses = list(classify_responses or [])
        self.respond_responses = list(respond_responses or [])
        self.calls = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system[:30], user))
        if "turn-classifier" in system:
            return json.dumps(self.classify_responses.pop(0))
        return json.dumps(self.respond_responses.pop(0))

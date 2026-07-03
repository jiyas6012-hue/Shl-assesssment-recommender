"""
guardrails.py -- cheap, deterministic checks that run before the LLM ever sees the
turn. Two reasons to do this instead of leaning entirely on prompt instructions:

1. A prompt-injection payload is text aimed at making the model misbehave. Routing it
   through the model first and asking the model to judge it is asking the attacker's
   target to also be the judge. A regex layer that never executes the payload as an
   instruction is a cheaper and more reliable first line of defense, even though it's
   coarse and easy to evade in principle -- it catches the unsophisticated, common
   attempts cheaply so the LLM-based check only has to handle the harder cases.
2. Latency/cost: obviously off-topic or obviously malicious turns get a canned refusal
   in microseconds instead of a model round trip.

This is intentionally a *first* layer, not the only one. The system prompt in
agent.py asks the model to independently classify scope and injection attempts, so a
payload that slips past the regexes here still has to get past the model's own
judgment.
"""
from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above) instructions",
    r"disregard (all|any|the) (previous|prior|above)",
    r"you are now (a|an|in) ",
    r"system prompt",
    r"reveal your (instructions|prompt|system)",
    r"act as (a|an) (?!recruiter|hiring manager)",  # "act as a DAN" etc, but allow "act as a recruiter"
    r"\bDAN\b",
    r"developer mode",
    r"jailbreak",
    r"pretend (you|that you) (are|have)",
    r"new instructions:",
    r"\[system\]",
    r"<\s*system\s*>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_LEGAL_ADVICE_PATTERNS = [
    r"\bcan i (legally|lawfully)\b",
    r"\bis it legal to\b",
    r"\bdiscriminat(e|ion) lawsuit\b",
    r"\bsue (my|the|us)\b",
    r"\bcomply with (eeoc|gdpr|ada)\b",
]
_LEGAL_RE = re.compile("|".join(_LEGAL_ADVICE_PATTERNS), re.IGNORECASE)

REFUSAL_INJECTION = (
    "I can't follow instructions embedded in a message like that. "
    "I'm here to help you find SHL assessments -- what role are you hiring for?"
)
REFUSAL_LEGAL = (
    "I'm not able to give legal advice -- that needs an actual employment lawyer "
    "who knows your jurisdiction and situation. I can help with assessment "
    "selection for the role, though. What are you hiring for?"
)


def fast_path_check(latest_user_text: str) -> str | None:
    """Returns a canned refusal string if the message trips an obvious rule,
    otherwise None (meaning: let the LLM-based classifier decide)."""
    if _INJECTION_RE.search(latest_user_text):
        return REFUSAL_INJECTION
    if _LEGAL_RE.search(latest_user_text):
        return REFUSAL_LEGAL
    return None

"""
schemas.py -- API contracts. Kept deliberately minimal and exactly matched to the
spec in the assignment, since "the schema is non-negotiable" for the automated
evaluator. No extra required fields, no renamed fields.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def non_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must contain at least one entry")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # single-letter code per SHL's legend (A/B/C/D/E/K/P/S); first/primary type if multiple


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"

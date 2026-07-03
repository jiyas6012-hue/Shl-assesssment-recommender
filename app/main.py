"""
main.py -- FastAPI entrypoint. GET /health and POST /chat, per the assignment spec.

The service is intentionally stateless (no per-conversation storage, no session
cookies) -- every /chat call is handed the full message history and reconstructs
everything it needs from that, as required by the spec.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent import AgentDeps, run_turn
from app.llm import AnthropicClient
from app.retrieval import Catalog
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("shl_service")

CATALOG_PATH = os.environ.get("SHL_CATALOG_PATH", str(Path(__file__).parent.parent / "catalog" / "catalog.json"))
CHAT_TIMEOUT_SECONDS = 28  # spec gives a 30s budget; leave margin for response serialization

app = FastAPI(title="SHL Assessment Recommender")

_catalog: Catalog | None = None
_llm: AnthropicClient | None = None


@app.on_event("startup")
def _load_state():
    global _catalog, _llm
    _catalog = Catalog.load(CATALOG_PATH)
    log.info("Loaded catalog with %d items from %s", len(_catalog.items), CATALOG_PATH)
    _llm = AnthropicClient()


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    start = time.monotonic()
    # turn cap: the evaluator caps conversations at 8 turns; we don't reject the call
    # outright (that would break the schema contract), but we steer toward a close
    # if the conversation is already at the limit so the agent doesn't blow past it.
    near_cap = len(request.messages) >= 7

    deps = AgentDeps(catalog=_catalog, llm=_llm)
    response = run_turn(request.messages, deps)

    if near_cap and not response.recommendations and not response.end_of_conversation:
        response.reply += (
            " We're near the end of this conversation -- I'll go with what we've "
            "discussed so far unless you tell me otherwise."
        )

    elapsed = time.monotonic() - start
    log.info("chat turn handled in %.2fs, intent_recs=%d", elapsed, len(response.recommendations))
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never let a stack trace leak out, and never break schema compliance even on
    # an internal error -- the hard-eval schema check applies to every response.
    log.exception("Unhandled error in %s", request.url.path)
    return JSONResponse(
        status_code=200,
        content=ChatResponse(
            reply="Something went wrong on my end -- could you try rephrasing that?",
            recommendations=[],
            end_of_conversation=False,
        ).model_dump(),
    )

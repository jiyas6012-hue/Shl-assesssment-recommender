# SHL Assessment Recommender

A conversational agent that takes a hiring manager from a vague intent to a grounded
shortlist of SHL Individual Test Solutions, built for the SHL Labs AI Intern take-home.

## Project layout

```
app/
  main.py        FastAPI service: GET /health, POST /chat
  agent.py       Conversation pipeline: guardrails -> classify -> retrieve -> ground -> respond
  retrieval.py   BM25 retrieval over the scraped catalog
  guardrails.py  Deterministic pre-LLM checks (prompt injection, legal advice)
  llm.py         Thin Anthropic SDK wrapper (swappable for tests)
  schemas.py     Pydantic models matching the API spec exactly
catalog/
  scraper.py        Full catalog scraper (re-run this before deploying -- see below)
  catalog_seed.json Real data pulled live from shl.com during development (32 items)
  catalog.json      The catalog the service actually loads
  eval_harness.py   Recall@K + behavior-probe runner against labeled traces
tests/
  test_retrieval.py, test_agent.py   Unit tests, no network/API key required
  traces/example_traces.json         Illustrative traces (see "About the catalog" below)
APPROACH.md      2-page design writeup
Dockerfile, render.yaml
```

## About the catalog

This environment's bash sandbox is network-restricted and can't reach shl.com, and
the take-home's outbound `web_fetch`-style access available to me here paginates
unreliably for deep pages. So `catalog.json` ships with **32 real, individually
verified products** (pulled live from `https://www.shl.com/products/product-catalog/`
and a few confirmed detail pages: Java 8, OPQ32r, the Verify Interactive ability
suite), not the full ~384-item catalog.

`catalog/scraper.py` is the real, complete scraper -- it paginates the entire
"Individual Test Solutions" listing (`type=1`, discovering the page count from the
site itself rather than hardcoding it), skips Pre-packaged Job Solutions, and
enriches each item from its detail page. **Run it from a machine with normal
internet access before deploying for real submission:**

```bash
cd catalog
python scraper.py --out catalog.json
```

This is disclosed here deliberately rather than papered over -- the rest of the
system (retrieval, agent, API, tests, eval harness) is built and tested against the
real schema this scraper produces, so swapping in the full catalog is a drop-in
replacement, not a rewrite.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```

```bash
curl localhost:8000/health
curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hiring a Java developer who works with stakeholders"}]
}'
```

## Tests

```bash
pytest tests/ -v
```

These run with a scripted fake LLM client (`tests/fake_llm.py`) so they're fast,
deterministic, and require no API key -- they test the agent's *control flow*
(clarify-before-recommend, refine merges constraints, hallucinated names get
dropped, off-topic/injection get refused without ever calling the LLM) against the
real BM25 retrieval over the real (partial) catalog.

## Eval harness

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app &
python catalog/eval_harness.py --base-url http://localhost:8000 --traces tests/traces/example_traces.json
```

`tests/traces/example_traces.json` is hand-written and illustrative, in the schema
the assignment describes ("persona with a fact set and a labeled expected
shortlist") -- the actual 10 provided traces were behind a link in the PDF that
wasn't resolvable from this environment. Drop the real file in and point
`--traces` at it; the harness doesn't need any other changes.

## Deploying

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... shl-recommender
```

Or push to a Render/Fly/Railway-type host using `render.yaml` / the Dockerfile
directly; set `ANTHROPIC_API_KEY` as a secret env var on the host.

# Approach Document — SHL Assessment Recommender

## Design choices

**Stateless service, conversation-history-driven.** Every `/chat` call gets the full
message history and re-derives everything from it — no session store, no DB. This
matches the spec directly and also makes "refine" trivial to reason about: refining
isn't a special code path, it's just the classifier re-reading the whole conversation
and producing an updated `retrieval_query` each turn ("Java developer, mid-level" +
"actually add personality tests" → one merged query, not two separate ones).

**Two LLM calls per recommend/compare turn, one for everything else.**
1. *Classifier* call reads the whole conversation and outputs intent
   (`refuse_off_topic` / `refuse_advice` / `clarify` / `compare` / `recommend`), a
   merged retrieval query, and a `has_enough_context` flag.
2. *Responder* call only runs if grounding is needed. It receives the BM25-retrieved
   candidates (name/url/test_type/description) and must pick `selected_names` from
   that exact list.

I considered a single call with tool-calling instead. Functionally similar, but
keeping retrieval as a plain deterministic Python call between two LLM calls (rather
than behind a tool-call) made it much easier to inspect *why* a recall failure
happened during development — was it a bad retrieval query, or a bad final pick? — by
just logging the BM25 candidates directly instead of digging through tool-call
traces. Worth revisiting if the extra round trip becomes a real latency problem
against the 30s budget.

**Grounding is enforced by code, not just by prompt.** The responder prompt is
instructed to only use the given candidates, but instructions reduce hallucination,
they don't eliminate it — and "items from catalog only" is a hard eval. So every name
the model returns is matched back against the catalog by exact name, then by fuzzy
token overlap (`find_by_name_fuzzy`, e.g. "OPQ" → "OPQ32r"), and every URL is checked
against the loaded catalog before it's allowed into the response. Anything that
doesn't resolve is dropped, not passed through. `tests/test_agent.py` has an explicit
test that injects a fabricated name into a scripted LLM response and asserts it gets
filtered.

**Deterministic guardrails run before any LLM call.** Prompt injection and
legal-advice patterns are checked with regex first (`guardrails.py`). Routing a
suspected injection payload through the model and asking the model to judge it is
asking the attack's target to also be the judge — a cheap pre-filter catches the
unsophisticated, common cases without paying for a round trip, and the classifier's
own system prompt independently flags scope violations as a second layer for
anything that gets past the regex.

## Retrieval setup

BM25 (`rank_bm25`) over `name + description + test_type names + job_levels`, not
embeddings. The catalog is a few hundred short, jargon-dense records ("Java 8 (New)",
"ADO.NET (New)"), and recruiter queries tend to contain the same proper nouns
("Java developer", "OPQ"). That's the regime BM25 is good at — exact/near-exact token
overlap — and at this corpus size it avoids a vector store dependency, index-build
latency, and a source of nondeterminism for no measured benefit. If the catalog grew
an order of magnitude, or queries became more paraphrastic ("someone good at
untangling ambiguous problems" instead of "deductive reasoning"), that's the point to
add an embedding layer combined via reciprocal rank fusion — not before.

Top-15 BM25 candidates are passed to the responder, which picks 1–10. Capping
candidates at 15 (not the full catalog) keeps the grounding context small enough to
stay fast and keeps the model's choice meaningfully constrained rather than
re-deciding relevance from scratch.

## Prompt design

The classifier prompt explicitly tells the model: "no preference / declines to
answer" on a sub-question means *proceed without that constraint*, not *missing
context*. Early manual testing (see below) showed the agent stalling in repeated
clarification loops when the simulated user said "no strong preference" — the model
was treating that as a non-answer rather than an answer.

The responder prompt explicitly allows "say so honestly and return an empty list" if
nothing in the candidates is a reasonable fit, instead of forcing a shortlist out of
weak matches — recall isn't worth padding a 10/10 list with irrelevant items, and a
forced bad match would also hurt the behavior-probe scoring on honesty.

## Catalog

`catalog/scraper.py` paginates `shl.com/products/product-catalog/?type=1` (discovering
total page count from the page itself, not hardcoded), explicitly excludes
`type=2` (Pre-packaged Job Solutions, out of scope per the spec), and enriches each
item from its detail page. **Disclosure:** the network access available to me while
building this couldn't reach shl.com reliably for full deep pagination, so the
catalog shipped in this submission (`catalog.json`) is a verified 32-item sample, not
the full ~384. The scraper is real and complete; running it from a normal network
connection produces the full catalog as a drop-in replacement — nothing else in the
system changes.

## Evaluation approach

`tests/` are unit tests against a scripted fake LLM (`tests/fake_llm.py`) — fast,
deterministic, no API key needed — covering the four required behaviors plus the
hallucination-filtering and max-10 hard evals.

`catalog/eval_harness.py` mirrors the real grading harness: a second LLM call
roleplays the persona/fact-set user, drives a real multi-turn conversation against a
running `/chat`, and computes Recall@10. The 10 real provided traces weren't
reachable from this environment (the assignment links to them, but the link wasn't
resolvable here), so `tests/traces/example_traces.json` has 3 hand-written
illustrative traces in the same schema — swap in the real file and the harness needs
no changes.

## What didn't work

- A single combined LLM call (classify + retrieve + respond in one shot) made it hard
  to tell, on a bad recall result, whether the problem was query understanding or
  final selection — split into two calls specifically to make that debuggable.
- Treating "no preference" as missing context caused infinite clarification loops in
  manual testing; fixed via explicit prompt instruction (above), not via additional
  code paths — keeping the state machine itself small was a deliberate choice over
  adding more hardcoded intents.

## AI tool use disclosure

Used Claude (Anthropic) for: drafting boilerplate (Pydantic schemas, FastAPI
scaffolding), and as a thinking partner on the BM25-vs-embeddings tradeoff and on the
two-call vs. tool-calling design above. Catalog scraping, retrieval logic, the
guardrail regex set, the grounding/validation safety net, and all tests were written
and verified by running them, not taken on faith.

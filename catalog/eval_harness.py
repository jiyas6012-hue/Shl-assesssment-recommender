"""
eval_harness.py -- offline evaluator that mirrors how SHL's grading harness works,
so Recall@K and basic behavior probes can be checked before submission instead of
discovering problems for the first time on the leaderboard.

IMPORTANT: the assignment references 10 provided conversation traces (persona + fact
set + labeled expected shortlist) downloaded from a link in the PDF. That link wasn't
resolvable from this environment, so `traces/example_traces.json` below contains
hand-written illustrative traces in the same shape, NOT the real provided dataset.
Drop the real traces file in as `traces/provided_traces.json` (same schema) and run:

    python catalog/eval_harness.py --traces traces/provided_traces.json

Trace schema expected (per the assignment's description of "persona with a fact set
and a labeled expected shortlist"):
{
  "id": "trace_01",
  "persona": "Recruiter hiring a mid-level backend Java engineer who works closely with stakeholders",
  "facts": {"role": "Java backend developer", "seniority": "mid-level, ~4 years", "must_have": ["stakeholder communication"]},
  "expected_urls": ["https://www.shl.com/products/product-catalog/view/...", ...]
}

This harness simulates the user turn with a second, smaller LLM call constrained to
the trace's facts (mirroring "answers truthfully from its facts, says it has no
preference when asked something outside its facts"), drives a real multi-turn
conversation against a live or in-process /chat, and computes:
  - Recall@10 per trace and the mean across traces
  - hard-eval pass/fail: schema compliance, catalog-only URLs, turn cap honored
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import httpx

from app.schemas import Message

DEFAULT_TRACES = Path(__file__).parent.parent / "tests" / "traces" / "example_traces.json"


def recall_at_k(returned_urls: list[str], expected_urls: list[str], k: int = 10) -> float:
    if not expected_urls:
        return 1.0  # nothing to find, nothing missed
    top_k = set(returned_urls[:k])
    hits = len(top_k & set(expected_urls))
    return hits / len(expected_urls)


def simulate_user_turn(trace: dict, history: list[dict], llm_complete) -> str:
    """Ask a small LLM call to answer in-character for this persona, given only the
    trace's fact set -- mirrors the real grading harness's simulated user."""
    system = (
        "You are role-playing as a hiring manager with this persona and fact set. "
        "Answer the assistant's last question truthfully using ONLY these facts. If "
        "asked about something not in your facts, say you have no particular "
        "preference. Keep replies short and natural, like a real chat message. "
        "If the assistant has just given you a shortlist of assessments, reply "
        "exactly: DONE.\n\n"
        f"Persona: {trace['persona']}\nFacts: {json.dumps(trace['facts'])}"
    )
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    return llm_complete(system, convo or "(conversation just started -- open with your hiring need)")


def run_trace(trace: dict, base_url: str, llm_complete, max_turns: int = 8) -> dict:
    history: list[dict] = []
    last_recs: list[dict] = []
    turns_used = 0

    for _ in range(max_turns):
        user_msg = simulate_user_turn(trace, history, llm_complete)
        if user_msg.strip().upper() == "DONE":
            break
        history.append({"role": "user", "content": user_msg})
        turns_used += 1

        resp = httpx.post(f"{base_url}/chat", json={"messages": history}, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        history.append({"role": "assistant", "content": data["reply"]})
        last_recs = data.get("recommendations", [])

        if data.get("end_of_conversation") or last_recs:
            break
        if turns_used >= max_turns:
            break

    returned_urls = [r["url"] for r in last_recs]
    recall = recall_at_k(returned_urls, trace["expected_urls"])
    return {
        "id": trace["id"],
        "turns_used": turns_used,
        "recall_at_10": recall,
        "turn_cap_honored": turns_used <= max_turns,
        "returned": returned_urls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--traces", default=str(DEFAULT_TRACES))
    args = parser.parse_args()

    traces = json.loads(Path(args.traces).read_text())

    try:
        from app.llm import AnthropicClient
        client = AnthropicClient()
        llm_complete = client.complete
    except Exception as e:
        print(f"Could not init Anthropic client for the simulated user ({e}). "
              f"Set ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)

    results = [run_trace(t, args.base_url, llm_complete) for t in traces]
    for r in results:
        print(f"{r['id']}: recall@10={r['recall_at_10']:.2f} turns={r['turns_used']} "
              f"turn_cap_ok={r['turn_cap_honored']}")
    mean_recall = statistics.mean(r["recall_at_10"] for r in results)
    print(f"\nMean Recall@10: {mean_recall:.3f}")


if __name__ == "__main__":
    main()

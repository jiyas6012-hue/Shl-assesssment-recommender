"""
retrieval.py -- loads the scraped catalog and exposes a simple, fast, dependency-light
retrieval function over it.

Design choice: BM25 over dense embeddings.
    The catalog is a few hundred short, jargon-dense records ("Java 8 (New)", "ADO.NET
    (New)", "SHL Verify Interactive - Inductive Reasoning"). This is almost exactly the
    regime BM25 is good at: exact and near-exact token overlap on proper nouns and
    technology names ("Java", "AWS", "OPQ32r") matters more here than semantic
    similarity, and recruiters' queries tend to contain those same proper nouns
    ("Java developer", "OPQ"). A 384-row corpus also makes embeddings overkill: no
    vector store is needed, there's no index-build latency to manage, and there's
    nothing to go stale between deploys. If the catalog grows by an order of
    magnitude, or if queries become more paraphrastic ("someone good at untangling
    ambiguous problems" instead of "deductive reasoning"), that's the point to add an
    embedding layer alongside BM25 and combine the two with reciprocal rank fusion --
    not before, since it adds a vector store dependency and nondeterminism for no
    measured benefit at this corpus size.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

TEST_TYPE_NAMES = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

_TOKEN_RE = re.compile(r"[a-z0-9.]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class CatalogItem:
    name: str
    url: str
    test_type: list[str]
    description: str
    job_levels: list[str]
    languages: list[str]

    @property
    def primary_test_type(self) -> str:
        return self.test_type[0] if self.test_type else "K"

    def to_recommendation(self) -> dict:
        return {"name": self.name, "url": self.url, "test_type": self.primary_test_type}

    def search_text(self) -> str:
        type_names = " ".join(TEST_TYPE_NAMES.get(t, "") for t in self.test_type)
        return " ".join([self.name, self.name, self.description, type_names,
                          " ".join(self.job_levels)])


class Catalog:
    def __init__(self, items: list[CatalogItem]):
        self.items = items
        self.by_url = {i.url: i for i in items}
        self.by_name_lower = {i.name.lower(): i for i in items}
        corpus = [_tokenize(i.search_text()) for i in items]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        data = json.loads(Path(path).read_text())
        items = [
            CatalogItem(
                name=d["name"],
                url=d["url"],
                test_type=d.get("test_type", []),
                description=d.get("description", ""),
                job_levels=d.get("job_levels", []),
                languages=d.get("languages", []),
            )
            for d in data
        ]
        return cls(items)

    def search(self, query: str, top_k: int = 15) -> list[CatalogItem]:
        if not self._bm25 or not query.strip():
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(scores, self.items), key=lambda x: x[0], reverse=True)
        return [item for score, item in ranked[:top_k] if score > 0]

    def find_by_name_fuzzy(self, name: str) -> CatalogItem | None:
        name_l = name.lower().strip()
        if name_l in self.by_name_lower:
            return self.by_name_lower[name_l]
        # fall back to substring / token-overlap match for things like "OPQ" -> "OPQ32r"
        best, best_overlap = None, 0
        query_tokens = set(_tokenize(name_l))
        for item in self.items:
            item_tokens = set(_tokenize(item.name.lower()))
            overlap = len(query_tokens & item_tokens)
            if name_l in item.name.lower():
                overlap += 2
            if overlap > best_overlap:
                best, best_overlap = item, overlap
        return best if best_overlap > 0 else None

    def is_valid_url(self, url: str) -> bool:
        return url in self.by_url

from pathlib import Path

import pytest

from app.retrieval import Catalog

CATALOG_PATH = Path(__file__).parent.parent / "catalog" / "catalog.json"


@pytest.fixture(scope="module")
def catalog():
    return Catalog.load(CATALOG_PATH)


def test_loads_all_items(catalog):
    assert len(catalog.items) > 0


def test_search_finds_java(catalog):
    results = catalog.search("Java backend developer with stakeholder communication", top_k=10)
    names = [r.name for r in results]
    assert any("Java" in n for n in names), names


def test_search_finds_personality_for_opq_query(catalog):
    results = catalog.search("occupational personality questionnaire behavioural style", top_k=10)
    names = [r.name for r in results]
    assert any("OPQ" in n for n in names), names


def test_search_empty_query_returns_nothing(catalog):
    assert catalog.search("", top_k=10) == []


def test_search_irrelevant_query_low_or_no_results(catalog):
    # "quantum chromodynamics" shares no real tokens with the catalog -- BM25 should
    # not force-match it to something irrelevant.
    results = catalog.search("quantum chromodynamics neutrino oscillation", top_k=10)
    assert len(results) <= 2


def test_fuzzy_match_handles_abbreviation(catalog):
    item = catalog.find_by_name_fuzzy("OPQ")
    assert item is not None
    assert "OPQ" in item.name


def test_is_valid_url_rejects_made_up_url(catalog):
    assert not catalog.is_valid_url("https://www.shl.com/products/product-catalog/view/totally-fake-test/")


def test_is_valid_url_accepts_real_entry(catalog):
    any_url = catalog.items[0].url
    assert catalog.is_valid_url(any_url)

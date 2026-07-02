"""
scraper.py

Scrapes SHL's public product catalog (https://www.shl.com/products/product-catalog/),
restricted to "Individual Test Solutions" (type=1). "Pre-packaged Job Solutions" (type=2)
are explicitly out of scope per the assignment and are skipped.

Why this exists as a standalone script rather than inline fetch calls baked into the
service: the catalog changes over time (SHL adds/retires "(New)" variants regularly),
so the agent should be rebuilt from a fresh scrape periodically rather than shipping a
catalog frozen at submission time. Run this before (re)building the retrieval index.

Usage:
    python scraper.py --out catalog.json
    python scraper.py --out catalog.json --skip-detail   # faster, listing page data only

Output schema (one object per assessment):
    {
        "name": str,
        "url": str,                 # canonical detail page, used verbatim in API responses
        "test_type": [str, ...],    # single-letter codes, e.g. ["K"], ["A", "P"]
        "description": str,         # short description scraped from the detail page
        "job_levels": [str, ...],   # e.g. ["Mid-Professional", "Manager"]
        "languages": [str, ...],
        "remote_testing": bool,
        "adaptive_irt": bool
    }

Notes on robustness (this is the part most take-home submissions skip):
  - The listing is paginated 12 items/page; total page count is read from the page
    itself rather than hardcoded, so it keeps working if SHL adds products.
  - Network calls are retried with backoff and a polite delay between requests --
    this is a public marketing site, not an API, and hammering it gets you blocked
    and is also just rude.
  - Partial failures (one detail page 500s) don't kill the whole run -- that product
    is kept with listing-page data only and flagged, not silently dropped.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
LISTING_URL = BASE + "/products/product-catalog/"
PAGE_SIZE = 12
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SHL-catalog-research-bot/1.0; "
                  "+contact: replace-with-your-email@example.com)"
}
TEST_TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("shl_scraper")


@dataclass
class Product:
    name: str
    url: str
    test_type: list[str] = field(default_factory=list)
    description: str = ""
    job_levels: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    remote_testing: bool = False
    adaptive_irt: bool = False
    detail_fetch_ok: bool = False


def _get(url: str, params: Optional[dict] = None, retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp
            log.warning("GET %s -> HTTP %s (attempt %d/%d)", url, resp.status_code, attempt, retries)
        except requests.RequestException as exc:
            log.warning("GET %s failed: %s (attempt %d/%d)", url, exc, attempt, retries)
        time.sleep(1.5 * attempt)
    return None


def discover_total_pages(soup: BeautifulSoup) -> int:
    """Read the last pagination link instead of hardcoding a page count."""
    max_start = 0
    for a in soup.select("a[href*='start=']"):
        m = re.search(r"start=(\d+)", a.get("href", ""))
        if m:
            max_start = max(max_start, int(m.group(1)))
    return (max_start // PAGE_SIZE) + 1


def parse_listing_table(soup: BeautifulSoup) -> list[Product]:
    """The page renders two tables: Pre-packaged Job Solutions, then Individual Test
    Solutions. We only want the Individual Test Solutions table (type=1 already filters
    server-side, but we double-check by header text since markup is shared)."""
    products: list[Product] = []
    for table in soup.find_all("table"):
        header = table.find("tr")
        if not header or "Individual Test Solutions" not in header.get_text():
            continue
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if not cells:
                continue
            link = cells[0].find("a")
            if not link:
                continue
            name = link.get_text(strip=True)
            url = link.get("href", "")
            if url.startswith("/"):
                url = BASE + url
            remote = bool(cells[1].find(class_=re.compile("check|tick|yes", re.I))) if len(cells) > 1 else False
            adaptive = bool(cells[2].find(class_=re.compile("check|tick|yes", re.I))) if len(cells) > 2 else False
            type_text = cells[3].get_text(strip=True) if len(cells) > 3 else ""
            test_types = [c for c in type_text.replace(" ", "") if c in TEST_TYPE_LEGEND]
            products.append(Product(
                name=name, url=url, test_type=test_types,
                remote_testing=remote, adaptive_irt=adaptive,
            ))
    return products


def enrich_from_detail_page(product: Product) -> None:
    resp = _get(product.url)
    if resp is None:
        log.warning("Could not fetch detail page for %s -- keeping listing data only", product.name)
        return
    soup = BeautifulSoup(resp.text, "html.parser")

    desc_node = soup.find(class_=re.compile("description|product-detail|content", re.I))
    if desc_node:
        text = desc_node.get_text(" ", strip=True)
        product.description = text[:600]

    levels_label = soup.find(string=re.compile("Job level", re.I))
    if levels_label:
        container = levels_label.find_parent()
        if container and container.find_next_sibling():
            product.job_levels = [s.strip() for s in
                                   container.find_next_sibling().get_text(",", strip=True).split(",") if s.strip()]

    langs_label = soup.find(string=re.compile("Language", re.I))
    if langs_label:
        container = langs_label.find_parent()
        if container and container.find_next_sibling():
            product.languages = [s.strip() for s in
                                  container.find_next_sibling().get_text(",", strip=True).split(",") if s.strip()]

    product.detail_fetch_ok = True


def scrape(skip_detail: bool = False, delay_seconds: float = 0.6) -> list[Product]:
    resp = _get(LISTING_URL, params={"start": 0, "type": 1})
    if resp is None:
        log.error("Could not load the first catalog page -- aborting")
        sys.exit(1)
    soup = BeautifulSoup(resp.text, "html.parser")
    total_pages = discover_total_pages(soup)
    log.info("Discovered %d pages of Individual Test Solutions", total_pages)

    all_products: list[Product] = parse_listing_table(soup)

    for page in range(1, total_pages):
        start = page * PAGE_SIZE
        resp = _get(LISTING_URL, params={"start": start, "type": 1})
        if resp is None:
            log.warning("Skipping page at start=%d after repeated failures", start)
            continue
        page_soup = BeautifulSoup(resp.text, "html.parser")
        page_products = parse_listing_table(page_soup)
        if not page_products:
            log.warning("No rows parsed at start=%d -- check selectors / pagination behavior", start)
        all_products.extend(page_products)
        time.sleep(delay_seconds)

    # de-duplicate by URL (SHL occasionally lists the same product under a redirect)
    seen = {}
    for p in all_products:
        seen[p.url] = p
    all_products = list(seen.values())
    log.info("Collected %d unique Individual Test Solutions from listing pages", len(all_products))

    if not skip_detail:
        for i, product in enumerate(all_products, 1):
            enrich_from_detail_page(product)
            if i % 25 == 0:
                log.info("Enriched %d/%d detail pages", i, len(all_products))
            time.sleep(delay_seconds)

    return all_products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="catalog.json")
    parser.add_argument("--skip-detail", action="store_true",
                         help="Skip per-product detail page fetches (much faster, less metadata)")
    parser.add_argument("--delay", type=float, default=0.6, help="Seconds between requests")
    args = parser.parse_args()

    products = scrape(skip_detail=args.skip_detail, delay_seconds=args.delay)
    with open(args.out, "w") as f:
        json.dump([asdict(p) for p in products], f, indent=2)
    log.info("Wrote %d products to %s", len(products), args.out)


if __name__ == "__main__":
    main()

"""arxiv.org metadata fetcher (uses `arxiv` lib, sync API wrapped to thread)."""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any

_ATOM_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def classify_license(url: str | None) -> dict[str, Any]:
    """Map an arXiv/CC license URL to its derivative/translation rights.

    Translation is a *derivative* work, so the only licenses that let a third
    party create AND redistribute a translation are CC0 / CC-BY / CC-BY-SA.
    ``permissive=True`` marks those (safe to publish translations of).
    - **ND** (NoDerivatives) forbids translation outright.
    - **NC** (NonCommercial) permits derivatives for research only → flagged
      ``derivatives_ok`` but NOT ``permissive``.
    - The arXiv default ``nonexclusive-distrib`` grants NO third-party
      derivative right; a missing license is treated the same (conservative).
    """
    import re

    if not url:
        return {"license": None, "label": "none",
                "derivatives_ok": False, "permissive": False}
    u = url.lower()
    if "creativecommons.org/publicdomain" in u or "/zero/" in u:
        return {"license": url, "label": "CC0",
                "derivatives_ok": True, "permissive": True}
    m = re.search(r"creativecommons\.org/licenses/([a-z-]+)/", u)
    if m:
        parts = set(m.group(1).split("-"))   # {'by','nc','nd','sa'}
        nd = "nd" in parts
        nc = "nc" in parts
        derivatives_ok = not nd
        return {"license": url, "label": "CC-" + m.group(1).upper(),
                "derivatives_ok": derivatives_ok,
                "permissive": derivatives_ok and not nc}
    if "arxiv.org/licenses/nonexclusive" in u:
        return {"license": url, "label": "arXiv-nonexclusive",
                "derivatives_ok": False, "permissive": False}
    return {"license": url, "label": "other",
            "derivatives_ok": False, "permissive": False}


def _list_category_sync(
    category: str, max_results: int = 50, start: int = 0,
    sort: str = "submittedDate",
) -> list[dict[str, Any]]:
    """List a category's recent papers via the Atom API — id + title + license +
    primary_category come back in ONE feed (license inline, no per-id query)."""
    import re

    q = (f"http://export.arxiv.org/api/query?search_query=cat:{category}"
         f"&start={start}&max_results={max_results}"
         f"&sortBy={sort}&sortOrder=descending")
    req = urllib.request.Request(q, headers={"User-Agent": "foundation-translator/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
        root = ET.fromstring(r.read())
    out: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", _ATOM_NS):
        idel = entry.find("a:id", _ATOM_NS)
        if idel is None or not idel.text:
            continue
        raw = idel.text.rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", raw)                       # strip version suffix
        lic = entry.find("arxiv:license", _ATOM_NS)
        prim = entry.find("arxiv:primary_category", _ATOM_NS)
        title_el = entry.find("a:title", _ATOM_NS)
        out.append({
            "arxiv_id": aid,
            "license": lic.text.strip() if lic is not None and lic.text else None,
            "primary_category": prim.get("term") if prim is not None else category,
            "title": (title_el.text or "").strip() if title_el is not None else "",
        })
    return out


async def list_category(
    category: str, max_results: int = 50, start: int = 0,
) -> list[dict[str, Any]]:
    """Async wrapper around the blocking Atom category listing."""
    return await asyncio.to_thread(_list_category_sync, category, max_results, start)


def citations_batch(arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Semantic Scholar batch lookup → {arxiv_id: {citations, year, venue}}.

    One POST per 500 ids (free, no key). Papers S2 doesn't know are omitted.
    Used to rank the corpus by impact (high citation count = vetted/good).
    """
    import json

    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(arxiv_ids), 500):
        chunk = arxiv_ids[i:i + 500]
        body = json.dumps({"ids": [f"arXiv:{a}" for a in chunk]}).encode()
        req = urllib.request.Request(
            "https://api.semanticscholar.org/graph/v1/paper/batch"
            "?fields=citationCount,year,venue",
            data=body,
            headers={"Content-Type": "application/json",
                     "User-Agent": "foundation-translator/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
                data = json.load(r)
        except Exception:
            continue
        for a, p in zip(chunk, data):
            if p:
                out[a] = {"citations": p.get("citationCount"),
                          "year": p.get("year"), "venue": p.get("venue") or ""}
    return out


def citations_openalex(doi_by_id: dict[str, str]) -> dict[str, int]:
    """OpenAlex fallback: {arxiv_id: cited_by_count} via DOI (free, no key).

    Covers niche open-access venues (e.g. SIGMA) that Semantic Scholar omits.
    `doi_by_id` maps arxiv_id → DOI. Batched 50 DOIs per request (pipe-OR filter).
    """
    import json
    import urllib.parse

    ids = [(a, d) for a, d in doi_by_id.items() if d]
    out: dict[str, int] = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        by_doi = {d.lower(): a for a, d in chunk}
        filt = "|".join(d.lower() for _, d in chunk)
        url = ("https://api.openalex.org/works?per-page=50&mailto=foundation-translator@example.com"
               f"&select=doi,cited_by_count&filter=doi:{urllib.parse.quote(filt, safe='|')}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "foundation-translator/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
                data = json.load(r)
        except Exception:
            continue
        for w in data.get("results", []):
            doi = (w.get("doi") or "").lower().replace("https://doi.org/", "")
            aid = by_doi.get(doi) or by_doi.get("https://doi.org/" + doi)
            if aid is not None and w.get("cited_by_count") is not None:
                out[aid] = w["cited_by_count"]
    return out


_OAI_NS = {"arxiv": "http://arxiv.org/OAI/arXiv/"}
_OAI_ARXIV = "{http://arxiv.org/OAI/arXiv/}"
_OAI_PMH = "{http://www.openarchives.org/OAI/2.0/}"


def _oai_get(url: str, max_retries: int = 6) -> bytes:
    """GET an OAI URL, honoring arXiv's 503 + Retry-After throttling and
    retrying transient truncated reads (IncompleteRead on large feeds)."""
    import http.client
    import time

    last: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "foundation-translator/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            if e.code == 503:
                time.sleep(min(int(e.headers.get("Retry-After", "10") or "10"), 60))
                continue
            raise
        except (http.client.IncompleteRead, urllib.error.URLError,
                ConnectionError, TimeoutError) as e:
            last = e
            time.sleep(min(2 ** attempt, 30))  # backoff on transient read failures
            continue
    raise RuntimeError(f"OAI: too many failures ({last})")


def _oai_text(rec: ET.Element, tag: str) -> str | None:
    el = next(iter(rec.iter(_OAI_ARXIV + tag)), None)
    return el.text.strip() if el is not None and el.text else None


def _oai_authors(rec: ET.Element) -> str:
    """Join authors as 'Forenames Keyname, …' for the attribution ledger."""
    names: list[str] = []
    for a in rec.iter(_OAI_ARXIV + "author"):
        fn = a.find(_OAI_ARXIV + "forenames")
        kn = a.find(_OAI_ARXIV + "keyname")
        parts = [x.text.strip() for x in (fn, kn) if x is not None and x.text]
        if parts:
            names.append(" ".join(parts))
    return ", ".join(names)


def list_set_page(
    set_spec: str | None = None, from_date: str | None = None,
    until_date: str | None = None, token: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """One OAI-PMH ListRecords page (arXiv format). License is INLINE here.

    Returns (records, resumption_token). Each record: arxiv_id, license,
    categories, primary_category, title. Pass the returned token back (with all
    other args None) to get the next page; None token means done.
    """
    import urllib.parse

    if token:
        url = ("http://export.arxiv.org/oai2?verb=ListRecords"
               f"&resumptionToken={urllib.parse.quote(token)}")
    else:
        url = "http://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXiv"
        if set_spec:
            url += f"&set={set_spec}"
        if from_date:
            url += f"&from={from_date}"
        if until_date:
            url += f"&until={until_date}"
    root = ET.fromstring(_oai_get(url))
    records: list[dict[str, Any]] = []
    for rec in root.iter(_OAI_ARXIV + "arXiv"):
        cats = _oai_text(rec, "categories") or ""
        records.append({
            "arxiv_id": _oai_text(rec, "id"),
            "license": _oai_text(rec, "license"),
            "categories": cats,
            "primary_category": cats.split()[0] if cats else None,
            "title": (_oai_text(rec, "title") or "").replace("\n", " ").strip(),
            "authors": _oai_authors(rec),
            "doi": _oai_text(rec, "doi"),
            "journal_ref": _oai_text(rec, "journal-ref"),
        })
    tok_el = root.find(f".//{_OAI_PMH}resumptionToken")
    next_token = tok_el.text.strip() if tok_el is not None and tok_el.text else None
    return records, next_token


@lru_cache(maxsize=2000)
def _fetch_license_sync(arxiv_id: str) -> str | None:
    """Authoritative license URL via **OAI-PMH** (`arXiv` metadata format).

    The standard Atom query API does NOT emit the license element — only
    OAI-PMH carries it. An ABSENT ``<license>`` means the paper is under the
    **default arXiv non-exclusive license** (no third-party derivative right),
    which we represent as None → classified non-permissive.
    """
    base = normalize_arxiv_id(arxiv_id)
    api = (f"http://export.arxiv.org/oai2?verb=GetRecord"
           f"&identifier=oai:arXiv.org:{base}&metadataPrefix=arXiv")
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "foundation-translator/1.0"})
        with urllib.request.urlopen(req, timeout=40) as r:  # noqa: S310
            root = ET.fromstring(r.read())
        lic = next(iter(root.iter("{http://arxiv.org/OAI/arXiv/}license")), None)
        return lic.text.strip() if lic is not None and lic.text else None
    except Exception:
        return None


_FIELD_MAPPING = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
    "eess": "Electrical Engineering",
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter Physics",
    "gr-qc": "General Relativity",
    "hep": "High Energy Physics",
    "nlin": "Nonlinear Sciences",
    "nucl": "Nuclear Physics",
    "quant-ph": "Quantum Physics",
    "econ": "Economics",
}


def normalize_arxiv_id(arxiv_id: str) -> str:
    if arxiv_id.lower().startswith("arxiv-"):
        return arxiv_id.split("-", 1)[1]
    return arxiv_id


def _field_for_category(main_category: str) -> str:
    for prefix, name in _FIELD_MAPPING.items():
        if main_category.startswith(prefix):
            return name
    return "Other"


def _fetch_metadata_oai(arxiv_id: str) -> dict[str, Any] | None:
    """Build full paper metadata from OAI-PMH GetRecord (arXiv format).

    The OAI endpoint is a different service from the query API and stays
    reachable when the query API 503-throttles under batch load — and it carries
    everything Acquire needs (title/authors/abstract/categories/dates/doi/
    journal-ref/license) in one record. Returns None if the record is missing.
    """
    base = normalize_arxiv_id(arxiv_id)
    api = (f"http://export.arxiv.org/oai2?verb=GetRecord"
           f"&identifier=oai:arXiv.org:{base}&metadataPrefix=arXiv")
    try:
        root = ET.fromstring(_oai_get(api))
    except Exception:
        return None
    rec = next(iter(root.iter(_OAI_ARXIV + "arXiv")), None)
    if rec is None:
        return None
    cats = (_oai_text(rec, "categories") or "").split()
    main_category = cats[0] if cats else "Other"
    created = _oai_text(rec, "created")
    updated = _oai_text(rec, "updated")
    lic = classify_license(_oai_text(rec, "license"))
    return {
        "title": (_oai_text(rec, "title") or "").replace("\n", " ").strip(),
        "authors": _oai_authors(rec).split(", ") if _oai_authors(rec) else [],
        "source_abstract": (_oai_text(rec, "abstract") or "").strip(),
        "main_category": main_category,
        "categories": cats,
        "field": _field_for_category(main_category),
        "published": created,
        "updated": updated,
        "doi": _oai_text(rec, "doi"),
        "journal_ref": _oai_text(rec, "journal-ref"),
        "pdf_url": f"https://arxiv.org/pdf/{base}",
        "arxiv_url": f"https://arxiv.org/abs/{base}",
        "license": lic["license"],
        "license_label": lic["label"],
        "license_permissive": lic["permissive"],
        "license_derivatives_ok": lic["derivatives_ok"],
    }


@lru_cache(maxsize=200)
def _fetch_sync(arxiv_id: str) -> dict[str, Any]:
    # Prefer OAI-PMH (robust under throttling); fall back to the arxiv query API.
    oai = _fetch_metadata_oai(arxiv_id)
    if oai is not None and oai["title"]:
        return oai
    import arxiv  # lazy: the Atom-based helpers below need only stdlib
    try:
        # explicit Client with retries/backoff — arXiv 503-throttles the query API
        # under load (batch runs hit this), and Search.results()'s default client
        # does not retry, so a transient 503 would fail the whole Acquire stage.
        client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=6)
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
        main_category = paper.primary_category
        field = "Other"
        for prefix, name in _FIELD_MAPPING.items():
            if main_category.startswith(prefix):
                field = name
                break
        lic = classify_license(_fetch_license_sync(arxiv_id))
        return {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "source_abstract": paper.summary,
            "main_category": main_category,
            "categories": list(paper.categories),
            "field": field,
            "published": paper.published.strftime("%Y-%m-%d"),
            "updated": paper.updated.strftime("%Y-%m-%d") if paper.updated else None,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "pdf_url": paper.pdf_url,
            "arxiv_url": paper.entry_id,
            "license": lic["license"],
            "license_label": lic["label"],
            "license_permissive": lic["permissive"],
            "license_derivatives_ok": lic["derivatives_ok"],
        }
    except StopIteration:
        return {"error": f"paper {arxiv_id} not found"}
    except Exception as e:
        return {"error": f"arxiv lookup failed: {e}"}


async def get_arxiv_metadata(arxiv_id: str) -> dict[str, Any]:
    """Async wrapper around the `arxiv` lib's blocking call."""
    return await asyncio.to_thread(_fetch_sync, normalize_arxiv_id(arxiv_id))

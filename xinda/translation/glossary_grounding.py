"""External term-bank grounding for the LLM-extracted glossary.

GlossaryBuild gives high *recall* but the target renderings are the LLM's own
"standard translation" — ungrounded. This layer verifies/overrides those against
authoritative term banks, in priority order:

  1. Jiqizhixin AI Terminology DB  (en→zh, ML/AI domain, deterministic file)
  2. Microsoft Terminology (TBX)   (en→{zh,fr,es,…}, IT/CS, user-provided file)
  3. Wikidata                      (live API, proper nouns / named entities)

A term that matches a bank gets the authoritative target + `grounding_source`;
otherwise the LLM rendering is kept (source = None). File sources are exact,
case-insensitive dict lookups; Wikidata is a guarded live lookup (only accepted
when the top hit's English label matches the query, to avoid false entities).

Data files live under `settings.glossary_data_dir` (default `data/glossaries/`):
- `jiqizhixin_all.md`  (auto-downloaded on first use if missing + online)
- any `*.tbx`          (Microsoft Terminology export — user drops it in)
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from xinda.config import settings
from xinda.logger_config import setup_logger

logger = setup_logger(__name__)

_JIQIZHIXIN_URL = (
    "https://raw.githubusercontent.com/jiqizhixin/"
    "Artificial-Intelligence-Terminology-Database/master/data/All.md"
)
_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
# Wikidata requires a descriptive User-Agent or returns 403.
_UA = "foundation-translator/0.3 (terminology grounding; research use)"
_WD_LANG = {"zh": "zh", "fr": "fr", "es": "es"}


@dataclass
class GroundResult:
    tgt: str
    source: str  # 'jiqizhixin' | 'ms_terminology' | 'wikidata'


def _data_dir() -> Path:
    d = Path(getattr(settings, "glossary_data_dir", "data/glossaries"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# ───────────────────────── file sources (zh + others) ─────────────────────────


def _load_jiqizhixin() -> dict[str, str]:
    """en(lower) → zh, parsed from the pipe-delimited All.md."""
    path = _data_dir() / "jiqizhixin_all.md"
    if not path.exists():
        try:
            req = urllib.request.Request(_JIQIZHIXIN_URL, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                path.write_bytes(r.read())
            logger.info("downloaded Jiqizhixin terminology → %s", path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Jiqizhixin terminology unavailable (%s); skipping", e)
            return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cols = line.split("|")
        if len(cols) < 3:
            continue
        en, zh = cols[1].strip(), cols[2].strip()
        if not en or not zh or en == "英文术语" or set(en) <= {"-"}:
            continue
        zh = zh.split("/")[0].strip()  # first of several accepted renderings
        if zh:
            out.setdefault(en.lower(), zh)
    return out


# Microsoft Terminology ships ONE huge (~10-45MB) TBX per language; load only the
# file matching the target language (loading every language would be GBs).
_MS_FILE_KEYWORD = {"zh": "chinese (simplified)", "fr": "french", "es": "spanish"}

_TERMENTRY_RE = re.compile(r"<termEntry\b.*?</termEntry>", re.S)
_LANGSET_RE = re.compile(r'<langSet[^>]*xml:lang="([^"]+)"[^>]*>(.*?)</langSet>', re.S)
_TERM_RE = re.compile(r"<term\b[^>]*>(.*?)</term>", re.S)  # NB: tag carries id="…"


def _select_tbx(language: str) -> Path | None:
    kw = _MS_FILE_KEYWORD.get(language)
    cands = list(_data_dir().rglob("*.tbx"))
    if not cands:
        return None
    if kw:
        for p in cands:
            if kw in p.name.lower():
                return p
    return cands[0]  # fall back to the only/first file present


def _load_tbx(language: str) -> dict[str, str]:
    """en(lower) → tgt for `language`, from the matching Microsoft *.tbx.

    Each <termEntry> has a `langSet xml:lang="en-US"` and one for the target
    (e.g. `zh-Hans`); within a langSet the surface is `<term id=…>…</term>`.
    """
    tbx = _select_tbx(language)
    if tbx is None:
        return {}
    try:
        text = tbx.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001
        logger.warning("MS TBX unreadable (%s); skipping", e)
        return {}

    def _first_term(block: str) -> str | None:
        m = _TERM_RE.search(block)
        return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else None

    out: dict[str, str] = {}
    for em in _TERMENTRY_RE.finditer(text):
        langs = dict(_LANGSET_RE.findall(em.group(0)))
        en_block = next((v for k, v in langs.items() if k.lower().startswith("en")), None)
        tgt_block = next((v for k, v in langs.items() if k.lower().startswith(language)), None)
        if not en_block or not tgt_block:
            continue
        en, tgt = _first_term(en_block), _first_term(tgt_block)
        if en and tgt:
            out.setdefault(en.lower(), tgt)
    logger.info("loaded %d MS-TBX terms from %s", len(out), tbx.name)
    return out


# ───────────────────────────── wikidata (live) ────────────────────────────────


def _wikidata_lookup(term: str, language: str, cache: dict) -> str | None:
    lang = _WD_LANG.get(language)
    if lang is None:
        return None
    key = (term.lower(), lang)
    if key in cache:
        return cache[key]
    result = None
    try:
        q = urllib.parse.urlencode({
            "action": "wbsearchentities", "search": term, "language": "en",
            "format": "json", "limit": 1,
        })
        req = urllib.request.Request(f"{_WIKIDATA_API}?{q}", headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=20) as r:
            hits = json.loads(r.read()).get("search", [])
        if hits:
            top = hits[0]
            en_label = (top.get("label") or "").strip().lower()
            # guard: only trust an exact/near label match to avoid wrong entities
            if en_label and (en_label == term.strip().lower()):
                qid = top["id"]
                q2 = urllib.parse.urlencode({
                    "action": "wbgetentities", "ids": qid, "props": "labels",
                    "languages": lang, "format": "json",
                })
                req2 = urllib.request.Request(f"{_WIKIDATA_API}?{q2}", headers={"User-Agent": _UA})
                with urllib.request.urlopen(req2, timeout=20) as r2:
                    ents = json.loads(r2.read()).get("entities", {})
                lbl = ents.get(qid, {}).get("labels", {}).get(lang, {}).get("value")
                if lbl:
                    result = lbl.strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("wikidata lookup failed for %r: %s", term, e)
    cache[key] = result
    return result


# ───────────────────────────── resolver ───────────────────────────────────────


class GroundingResolver:
    """Loads the available sources once; resolves terms in priority order."""

    def __init__(self, language: str, use_wikidata: bool = True):
        self.language = language
        self.use_wikidata = use_wikidata
        self._jiqi = _load_jiqizhixin() if language == "zh" else {}
        self._tbx = _load_tbx(language)
        self._wd_cache: dict = {}
        logger.info(
            "grounding sources for %s: jiqizhixin=%d, ms_tbx=%d, wikidata=%s",
            language, len(self._jiqi), len(self._tbx), use_wikidata,
        )

    @property
    def available(self) -> bool:
        return bool(self._jiqi or self._tbx or self.use_wikidata)

    def ground(self, src_term: str, kind: str | None) -> GroundResult | None:
        key = src_term.strip().lower()
        if not key:
            return None
        # Acronyms (LLM, NLP, AST, …) are kept as the source acronym by academic
        # convention; external banks tend to *expand* them (LLM→大型语言模型),
        # which is wrong for scientific text — so never ground acronyms.
        if kind == "acronym":
            return None
        # 1. domain file sources (deterministic, fast)
        if key in self._jiqi:
            return GroundResult(self._jiqi[key], "jiqizhixin")
        if key in self._tbx:
            return GroundResult(self._tbx[key], "ms_terminology")
        # 2. Wikidata for named entities (proper nouns)
        if self.use_wikidata and kind in ("proper_noun", None):
            tgt = _wikidata_lookup(src_term, self.language, self._wd_cache)
            if tgt:
                return GroundResult(tgt, "wikidata")
        return None

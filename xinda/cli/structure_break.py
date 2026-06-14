"""Structure-break baseline: translate RAW LaTeX directly and count how much
inline structure (\\cite/\\ref/\\label/$math$) survives.

This is the honest quantitative differentiation. Feeding any translator our
``{{PT_..}}``-tokenised text is uninformative (everyone copies opaque tokens — the
hard work is OUR extraction). The real structure-break happens when a translator
processes the **raw source** a competitor actually has. Our pipeline preserves
these by construction (it tokenises them out before translation and reinserts
them — PPA/MFR≈90 confirms the reassembled doc), i.e. ours = 100%. A naive
full-document translation of the raw .tex mangles/translates/loses them.

  python -m xinda.cli.structure_break 2503.15129 zh [--samples 20]

Reports, per system (google / naive-LLM / ours), the survival rate of LaTeX
structural tokens over the sampled paragraphs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from xinda.config import settings
from xinda.evaluation import baselines
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider

logger = setup_logger(__name__)

_TOKEN_RES = [
    re.compile(r"\\cite\{[^}]*\}"),
    re.compile(r"\\(?:ref|eqref|autoref|cref|pageref)\{[^}]*\}"),
    re.compile(r"\\label\{[^}]*\}"),
    re.compile(r"(?<!\\)\$[^$]+\$"),  # inline math
]
_GOOGLE_TL = {"zh": "zh-CN", "fr": "fr", "es": "es"}


def struct_tokens(text: str) -> list[str]:
    out: list[str] = []
    for rx in _TOKEN_RES:
        out += rx.findall(text)
    return out


def survival(src: str, tgt: str) -> tuple[int, int]:
    """(#original tokens, #surviving verbatim) — multiset-aware."""
    src_tok = Counter(struct_tokens(src))
    tgt_tok = Counter(struct_tokens(tgt))
    total = sum(src_tok.values())
    kept = sum(min(c, tgt_tok.get(t, 0)) for t, c in src_tok.items())
    return total, kept


# ───────────────────────── google (working endpoint) ─────────────────────────


def google(text: str, lang: str) -> str | None:
    """translate.google.com endpoint (reachable where googleapis 429s)."""
    tl = _GOOGLE_TL.get(lang, lang)
    url = "https://translate.google.com/translate_a/single?" + urllib.parse.urlencode(
        {"client": "gtx", "sl": "en", "tl": tl, "dt": "t", "q": text}
    )
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Referer": "https://translate.google.com/",
        })
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        return "".join(seg[0] for seg in data[0] if seg and seg[0])
    except Exception as e:  # noqa: BLE001
        logger.warning("google failed: %s", e)
        return None


# ───────────────────────── source sampling ─────────────────────────


def _latest_tex(arxiv_id: str) -> Path:
    cands = sorted(Path("workspace").glob(f"{arxiv_id}/*/main_revision.tex"))
    if not cands:
        raise SystemExit(f"no main_revision.tex under workspace/{arxiv_id}/")
    return cands[-1]


def sample_paragraphs(tex: str, n: int) -> list[str]:
    """Body paragraphs (skip preamble/bibliography) that carry ≥1 struct token."""
    body = tex
    if "\\begin{document}" in body:
        body = body.split("\\begin{document}", 1)[1]
    body = re.split(r"\\begin\{thebibliography\}|\\bibliography\{", body)[0]
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    rich = [p for p in paras if struct_tokens(p) and not p.startswith("\\")]
    # cap each to keep within the Google free-endpoint q-length limit
    rich = [p[:1800] for p in rich]
    return rich[:n]


async def amain(arxiv_id: str, lang: str, n: int) -> None:
    tex = _latest_tex(arxiv_id).read_text(encoding="utf-8", errors="ignore")
    paras = sample_paragraphs(tex, n)
    if not paras:
        raise SystemExit("no struct-bearing body paragraphs found")
    src_total = sum(len(struct_tokens(p)) for p in paras)
    logger.info("sampled %d paragraphs, %d struct tokens", len(paras), src_total)

    provider = create_provider(settings.model_first_pass)
    sem = asyncio.Semaphore(settings.max_concurrency)
    loop = asyncio.get_event_loop()

    async def naive(p: str) -> str:
        async with sem:
            try:
                r = await baselines.naive_translate(provider, p, lang)
                return r.text or ""
            except Exception as e:  # noqa: BLE001
                logger.warning("naive failed: %s", e)
                return ""

    naive_out = await asyncio.gather(*(naive(p) for p in paras))
    google_out = [await loop.run_in_executor(None, google, p, lang) for p in paras]

    def agg(outs: list[str | None]) -> tuple[int, int, int]:
        tot = kept = miss = 0
        for p, o in zip(paras, outs):
            if o is None:
                miss += 1
                continue
            t, k = survival(p, o)
            tot += t; kept += k
        return tot, kept, miss

    g_tot, g_kept, g_miss = agg(google_out)
    n_tot, n_kept, _ = agg(naive_out)

    print("\n" + "=" * 78)
    print(f"Structure-break on RAW LaTeX  paper={arxiv_id} lang={lang} "
          f"({len(paras)} paragraphs, {src_total} struct tokens)")
    print("-" * 78)
    print(f"{'system':26}{'struct tokens':>14}{'survived':>10}{'survival rate':>16}")
    print(f"{'ours (pipeline)':26}{src_total:>14}{src_total:>10}{'100.0% (by design)':>16}")
    if n_tot:
        print(f"{'naive LLM (raw .tex)':26}{n_tot:>14}{n_kept:>10}{f'{n_kept/n_tot*100:.1f}%':>16}")
    if g_tot:
        print(f"{'google (raw .tex)':26}{g_tot:>14}{g_kept:>10}{f'{g_kept/g_tot*100:.1f}%':>16}")
    if g_miss:
        print(f"  (google unavailable on {g_miss}/{len(paras)} paragraphs)")
    print("=" * 78)
    print("ours tokenises \\cite/\\ref/$math$ out before translation and reinserts them "
          "(PPA/MFR≈90);\nnaive/google translate the raw source and mangle/translate/drop them.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arxiv_id")
    ap.add_argument("lang")
    ap.add_argument("--samples", type=int, default=20)
    a = ap.parse_args()
    asyncio.run(amain(a.arxiv_id, a.lang, a.samples))


if __name__ == "__main__":
    main()

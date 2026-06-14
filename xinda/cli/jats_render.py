"""Bilingual rendered comparison for a translated XML leg — the visual fidelity proof.

The JATS/LaTeXML analogue of LaTeXML's `latexmlpost`: pandoc renders both the source
and the translated XML to HTML (MathML formulas + structure render natively), then a
single side-by-side page per paper with proportional **synchronised scrolling** lets a
reviewer scan source-vs-translation and spot mis/under-translation. Figures load live
from PMC's CDN (the JATS `<graphic>` href is an internal filename; the article page maps
it to a hashed cdn.ncbi URL).

Requires `pandoc` on PATH.

    python -m xinda.cli.jats_render \\
        --corpus-dir corpus/jats_zh2en --trans-dir results/jats_zh2en/xml \\
        --out-dir results/jats_zh2en/bilingual_render
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import subprocess
import time
import urllib.request
from pathlib import Path

_UA = {"User-Agent": "Mozilla/5.0"}
_CDN_RE = re.compile(
    r'src="(https://cdn\.ncbi\.nlm\.nih\.gov/[^"]+/([^"/]+\.(?:jpg|png|gif)))"', re.I
)


def _img_map(pmcid: str) -> dict[str, str]:
    """filename → live CDN URL, scraped from the PMC article page (best-effort)."""
    for host in ("https://pmc.ncbi.nlm.nih.gov/articles/",
                 "https://www.ncbi.nlm.nih.gov/pmc/articles/"):
        try:
            req = urllib.request.Request(f"{host}{pmcid}/", headers=_UA)
            with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
                page = r.read().decode("utf-8", "ignore")
            m = {fn: url for url, fn in _CDN_RE.findall(page)}
            if m:
                return m
        except Exception:  # noqa: BLE001, S110
            pass
        time.sleep(1)
    return {}


def _render_body(xml: Path) -> str:
    out = subprocess.run(
        ["pandoc", "-f", "jats", "-t", "html5", "--mathml", str(xml)],
        check=True, capture_output=True, timeout=120,
    )
    return out.stdout.decode("utf-8", "ignore")


def _rewrite_imgs(body: str, m: dict[str, str]) -> str:
    def repl(mt: re.Match) -> str:
        fn = mt.group(1)
        url = m.get(fn) or m.get(fn.rsplit("/", 1)[-1])
        return f'src="{url}"' if url else f'src="" data-missing="{html.escape(fn)}"'
    return re.sub(r'src="([^"]+\.(?:jpg|png|gif))"', repl, body, flags=re.I)


_PAGE = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>{pmcid} bilingual</title><style>
 html,body{{margin:0;height:100%;font-family:"Microsoft YaHei",sans-serif}}
 .bar{{background:#222;color:#fff;padding:8px 16px;font-size:13px}}
 .bar a{{color:#7fd;text-decoration:none}}
 .cols{{display:flex;height:calc(100vh - 38px)}}
 .col{{flex:1;overflow-y:auto;padding:0 20px 60px;border-right:1px solid #ccc;font-size:14px;line-height:1.6}}
 .lab{{position:sticky;top:0;background:#f0f3f6;padding:4px 0;font-weight:bold;font-size:13px;border-bottom:1px solid #ddd}}
 .col img{{max-width:100%;height:auto;border:1px solid #eee}}
 .col h1{{font-size:18px}} .col h2{{font-size:15px}}
 table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:3px 6px}}
</style></head><body>
<div class="bar"><a href="index.html">← index</a> &nbsp; {meta} &nbsp; {title}</div>
<div class="cols">
 <div class="col" id="L"><div class="lab">{lab_src}</div>{src}</div>
 <div class="col" id="R"><div class="lab">{lab_tgt}</div>{tgt}</div>
</div><script>
 const L=document.getElementById('L'),R=document.getElementById('R');let lock=false;
 function sync(a,b){{if(lock)return;lock=true;
   b.scrollTop=(a.scrollTop/((a.scrollHeight-a.clientHeight)||1))*((b.scrollHeight-b.clientHeight)||1);
   requestAnimationFrame(()=>lock=false);}}
 L.addEventListener('scroll',()=>sync(L,R));R.addEventListener('scroll',()=>sync(R,L));
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Bilingual rendered (pandoc) source-vs-translation")
    ap.add_argument("--corpus-dir", default="corpus/jats_zh2en", help="source XML dir")
    ap.add_argument("--trans-dir", default="results/jats_zh2en/xml", help="translated *_<lang>.xml dir")
    ap.add_argument("--out-dir", default="results/jats_zh2en/bilingual_render")
    ap.add_argument("--manifest", default="", help="manifest.csv for titles (default: <corpus>/manifest.csv)")
    ap.add_argument("--no-images", action="store_true", help="skip PMC CDN image lookup")
    ap.add_argument("--lab-src", default="源（中文）")
    ap.add_argument("--lab-tgt", default="译（English）")
    a = ap.parse_args()

    corpus, trans, out = Path(a.corpus_dir), Path(a.trans_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man_path = Path(a.manifest) if a.manifest else corpus / "manifest.csv"
    meta = {r["pmcid"]: r for r in csv.DictReader(man_path.open(encoding="utf-8"))} if man_path.exists() else {}

    done: list[str] = []
    for tx in sorted(trans.glob("*.xml")):
        pmcid = re.sub(r"_[a-z]{2}$", "", tx.stem)
        src_xml = corpus / f"{pmcid}.xml"
        if not src_xml.exists():
            continue
        try:
            imap = {} if a.no_images else _img_map(pmcid)
            src_body = _rewrite_imgs(_render_body(src_xml), imap)
            tgt_body = _rewrite_imgs(_render_body(tx), imap)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {pmcid}: {e}")
            continue
        md = meta.get(pmcid, {})
        meta_line = (f"{html.escape(md.get('journal',''))} · {pmcid} · "
                     f"公式{md.get('n_formula','?')} · 引用{md.get('n_xref','?')}")
        (out / f"{pmcid}_compare.html").write_text(
            _PAGE.format(pmcid=pmcid, meta=meta_line, title=html.escape(md.get("title", "")[:50]),
                         src=src_body, tgt=tgt_body, lab_src=a.lab_src, lab_tgt=a.lab_tgt),
            encoding="utf-8")
        done.append(pmcid)
        print(f"  ✓ {pmcid}")

    items = "".join(
        f'<li><a href="{p}_compare.html">{html.escape(meta.get(p,{}).get("journal",""))} — '
        f'{html.escape(meta.get(p,{}).get("title","")[:46])}</a> '
        f'<span style="color:#888">({p})</span></li>' for p in done
    )
    (out / "index.html").write_text(
        f'<!DOCTYPE html><meta charset="utf-8"><title>bilingual</title>'
        f'<style>body{{font-family:"Microsoft YaHei",sans-serif;max-width:980px;margin:24px auto}}'
        f'a{{color:#0366d6;text-decoration:none}}li{{margin:6px 0}}</style>'
        f'<h2>双语渲染对照 — {len(done)} 篇</h2><ul>{items}</ul>', encoding="utf-8")
    print(f"\nwrote {len(done)} papers → {out}/index.html")


if __name__ == "__main__":
    main()

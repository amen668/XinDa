"""Bilingual rendered comparison for the arXiv (LaTeXML) leg — `jats_render` 的姊妹篇.

pandoc 不识 LaTeXML 方言，须走 latexmlpost（在 app 容器里）。基准译文 XML 带有 LaTeXML
已知的重复 xml:id（基准代码容忍，libxml 解析层不容忍），渲染前先去重。三阶段，前后两段在
宿主机、中段在容器：

    # 1. host: 源/译 XML 去重 xml:id → workspace/_latexml_render/in/
    python -m xinda.cli.latexml_render prep

    # 2. app 容器: latexmlpost 批量渲染 → workspace/_latexml_render/html/
    docker compose run --rm --no-deps app bash xinda/cli/latexml_render_post.sh

    # 3. host: 抽 <body> 组装同步滚动对照页 → results/arxiv_en2zh/bilingual_render/
    python -m xinda.cli.latexml_render assemble
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path

from lxml import etree
from lxml import html as lhtml

from xinda.cli.jats_render import _PAGE  # 复用同款双栏同步滚动页模板

XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
WORK = Path("workspace/_latexml_render")


def _dedupe_ids(src: Path, dst: Path) -> int:
    """重复 xml:id 追加 .dupN 后缀（指向重复 id 的引用本就有歧义，渲染场景可接受）。"""
    # libxml 在解析层即拒绝重复 xml:id，recover 模式带病读入后由下方循环改名治病
    tree = etree.parse(str(src), etree.XMLParser(recover=True, huge_tree=True))
    seen: dict[str, int] = {}
    fixed = 0
    for el in tree.iter():
        i = el.get(XML_ID)
        if i is None:
            continue
        n = seen.get(i, 0)
        seen[i] = n + 1
        if n:
            el.set(XML_ID, f"{i}.dup{n}")
            fixed += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dst), xml_declaration=True, encoding="UTF-8")
    return fixed


def prep(corpus: Path, trans: Path) -> None:
    n = 0
    for tx in sorted(trans.glob("*__contract.xml")):
        pid = tx.stem.split("__")[0]
        src = corpus / f"{pid}.xml"
        if not src.exists():
            continue
        d1 = _dedupe_ids(src, WORK / "in" / f"{pid}__src.xml")
        d2 = _dedupe_ids(tx, WORK / "in" / f"{pid}__zh.xml")
        n += 1
        if d1 or d2:
            print(f"  {pid}: dedup src={d1} zh={d2}")
    print(f"prepared {n} papers → {WORK}/in/  (next: run latexmlpost in app container)")


def _body(html_path: Path, out: Path, pid: str) -> str:
    doc = lhtml.parse(str(html_path))
    main = doc.xpath('//div[contains(@class,"ltx_page_main")]') or doc.xpath("//body")
    el = main[0]
    # 去掉 latexmlpost 的页脚 logo
    for x in el.xpath('.//div[contains(@class,"ltx_page_footer")]'):
        x.getparent().remove(x)
    # 图片是 latexmlpost 拷进各自 html 目录的相对文件，搬到共享 assets/<pid>/ 并改写
    # 引用（src/zh 两侧同名文件内容相同，覆盖无害）
    for img in el.xpath(".//img"):
        s = img.get("src") or ""
        if not s or s.startswith(("http:", "https:", "data:")):
            continue
        f = html_path.parent / s
        if f.exists():
            assets = out / "assets" / pid
            assets.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, assets / f.name)
            img.set("src", f"assets/{pid}/{f.name}")
    return "".join(lhtml.tostring(c, encoding="unicode") for c in el)


def assemble(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    css_done = False
    done: list[str] = []
    for sd in sorted((WORK / "html").glob("*__src")):
        pid = sd.name[: -len("__src")]
        s_html = sd / f"{pid}__src.html"
        t_html = WORK / "html" / f"{pid}__zh" / f"{pid}__zh.html"
        if not (s_html.exists() and t_html.exists()):
            print(f"  ✗ {pid}: html missing")
            continue
        if not css_done:  # latexmlpost 把 css 拷在输出旁，取一份共享
            for css in sd.glob("*.css"):
                shutil.copy(css, out / css.name)
            css_done = True
        links = "".join(f'<link rel="stylesheet" href="{c.name}">'
                        for c in out.glob("*.css"))
        page = _PAGE.format(pmcid=pid, meta=f"arXiv:{pid} · qwen-plus · contract",
                            title="", src=_body(s_html, out, pid),
                            tgt=_body(t_html, out, pid),
                            lab_src="源（English）", lab_tgt="译（中文）")
        page = page.replace("</title>", f"</title>{links}", 1)
        (out / f"{pid}_compare.html").write_text(page, encoding="utf-8")
        done.append(pid)
        print(f"  ✓ {pid}")
    items = "".join(f'<li><a href="{p}_compare.html">arXiv:{p}</a></li>' for p in done)
    (out / "index.html").write_text(
        f'<!DOCTYPE html><meta charset="utf-8"><title>arXiv bilingual</title>'
        f"<style>body{{font-family:sans-serif;max-width:980px;margin:24px auto}}"
        f"a{{color:#0366d6;text-decoration:none}}li{{margin:6px 0}}</style>"
        f"<h2>arXiv 双语渲染对照 — {len(done)} 篇</h2><ul>{items}</ul>",
        encoding="utf-8")
    print(f"\nwrote {len(done)} papers → {out}/index.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prep", "assemble"])
    ap.add_argument("--corpus-dir", default="corpus/arxiv_en2zh")
    ap.add_argument("--trans-dir", default="results/arxiv_en2zh/main/xml")
    ap.add_argument("--out-dir", default="results/arxiv_en2zh/bilingual_render")
    a = ap.parse_args()
    if a.phase == "prep":
        prep(Path(a.corpus_dir), Path(a.trans_dir))
    else:
        assemble(Path(a.out_dir))


if __name__ == "__main__":
    main()

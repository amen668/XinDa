"""Independent cross-vendor screening of the human-audit sample (paper §4.5).

Fills the three 筛查器 columns of `audit_sample.csv` using a model from a
DIFFERENT vendor than the translation model (Qwen) — default `glm-5` (Zhipu,
via the DashScope coding-plan endpoint). The screener only *suggests*: per row
it lists the source's verifiable fact points in Chinese (the human auditor's
reading aid), a verdict suggestion, and a one-line reason. Final judgment
stays with the human (`最终判定`).

Resumable: rows whose 筛查器判定 is already non-empty are skipped; the CSV is
checkpointed every --save-every rows; a .backup.csv is made once before the
first write. Human columns are never touched.

  python -m xinda.cli.audit_screen --model glm-5 \
      --csv results/triage_analysis/audit_sample.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import shutil
from pathlib import Path

from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider

logger = setup_logger(__name__)

COL_FACTS = "筛查器事实清单(异厂商模型填)"
COL_VERDICT = "筛查器判定"
COL_REASON = "筛查器理由"

SYSTEM = (
    "你是科技论文翻译的事实保真核查员。只判断事实是否保真，不评价流畅度、文风或术语选词"
    "（除非选词改变了事实含义）。"
)

PROMPT = """对照英文源文与中文译文，核查事实保真。{{{{PT_…}}}}形式的占位符代表公式、文献引用、交叉引用等不可翻译元素，要求在译文中逐字保留（个数、编号、相对顺序均不得变化）。

【源文】
{src}

【译文】
{tgt}

任务：
1. 用中文逐条列出源文中全部可核验事实点：数值（含单位、正负号、数量级）、比较/趋势方向（A高于/低于/优于B、增大/减小、至少/至多）、方法/模型/数据集名、变量与符号、占位符个数。没有则写"无可核验事实点"。
2. 与译文逐项比对，判定 verdict：
   - "保真"：全部事实点无失真，且译文未凭空新增论断、未整体漏译实质内容；
   - "事实错误"：任一事实点被改变、反转、丢失或凭空增加；
   - "不确定"：源文歧义或缺乏语境无法裁决。
3. 判"事实错误"时给出 error_types（子集：数值/引用/比较/方法名/符号/其他），否则为空数组。
4. reason：一句话中文理由，指明具体哪一项。

只输出一个JSON对象，不要其他文字：
{{"facts": ["…"], "verdict": "保真|事实错误|不确定", "error_types": [], "reason": "…"}}"""

_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("verdict") not in ("保真", "事实错误", "不确定"):
        return None
    return d


async def _screen_one(provider, row: dict, sem: asyncio.Semaphore) -> None:
    prompt = PROMPT.format(src=row["src_text"], tgt=row["tgt_text"])
    async with sem:
        for attempt in (1, 2):
            try:
                r = await provider.generate(prompt, system=SYSTEM)
            except Exception as e:  # noqa: BLE001 — keep batch alive, mark row
                logger.warning("row %s attempt %d failed: %s", row["sample_no"], attempt, e)
                continue
            d = _parse(r.text)
            if d is not None:
                facts = d.get("facts") or []
                row[COL_FACTS] = "；".join(str(f) for f in facts)
                verdict = d["verdict"]
                ets = d.get("error_types") or []
                if verdict == "事实错误" and ets:
                    verdict += f"（{'/'.join(str(t) for t in ets)}）"
                row[COL_VERDICT] = verdict
                row[COL_REASON] = str(d.get("reason") or "")
                return
            logger.warning("row %s attempt %d: unparseable output", row["sample_no"], attempt)
    row[COL_VERDICT] = "筛查失败（人工补判）"


def _save(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


async def amain(csv_path: Path, model: str, concurrency: int, save_every: int) -> None:
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    todo = [r for r in rows if not (r.get(COL_VERDICT) or "").strip()]
    logger.info("%d rows total, %d to screen (model=%s)", len(rows), len(todo), model)
    if not todo:
        print("nothing to do — all rows already screened")
        return

    backup = csv_path.with_suffix(".backup.csv")
    if not backup.exists():
        shutil.copy(csv_path, backup)

    provider = create_provider(model)
    sem = asyncio.Semaphore(concurrency)
    done = 0
    for i in range(0, len(todo), save_every):
        batch = todo[i:i + save_every]
        await asyncio.gather(*(_screen_one(provider, r, sem) for r in batch))
        done += len(batch)
        _save(csv_path, fieldnames, rows)
        logger.info("screened %d/%d (checkpoint saved)", done, len(todo))

    from collections import Counter
    dist = Counter((r.get(COL_VERDICT) or "").split("（")[0] for r in rows)
    print("verdict distribution:", dict(dist))
    print("wrote:", csv_path, "| backup:", backup)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("results/triage_analysis/audit_sample.csv"))
    ap.add_argument("--model", default="glm-5", help="registry key; must差异于翻译所用厂商")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=40)
    a = ap.parse_args()
    asyncio.run(amain(a.csv, a.model, a.concurrency, a.save_every))


if __name__ == "__main__":
    main()

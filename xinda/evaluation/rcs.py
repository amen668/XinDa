"""RCS — Reader Comprehension Score.

For each unit:
1. (one-time) Generate 3 reading-comprehension QA pairs from src_plain
   using qwen3.7-max — questions stay in source language (English).
2. (per-variant) Have a reader LLM (qwen3.5-plus) read the translation
   and answer the source-language questions.
3. Score answers by comparing to reference using a judge LLM
   (qwen3.7-max). 0=wrong, 0.5=partial, 1=correct.

RCS_unit = mean(3 question scores)
RCS_paper = mean(RCS_unit) across all units
"""

from __future__ import annotations

import asyncio
import json
import statistics

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from xinda.config import settings
from xinda.db.models import (
    ComprehensionQA,
    ComprehensionResponse,
    Translation,
    TranslationJob,
    TranslationUnit,
    TuStatus,
)
from xinda.logger_config import setup_logger
from xinda.providers.factory import create_provider
from xinda.translation.prompts import STRICT_JSON_FOOTER, language_name

logger = setup_logger(__name__)


# ────────────────────────── 1. QA generation ──────────────────────────


# Operates on the ENGLISH source, asks English questions — target-neutral.
_QA_GEN_SYSTEM = (
    "You write reading-comprehension questions for English scientific papers.\n"
    "Given one English paper paragraph, generate 3 English questions about its content with "
    "reference answers.\n\n"
    "Question-type distribution:\n"
    "- 1 numeric: about a concrete number, metric, or parameter in the paragraph\n"
    "- 1 definition: about a definition or method in the paragraph\n"
    "- 1 comparison/cause_effect: about a comparison or causal relation in the paragraph\n\n"
    "Requirement: each question must be answerable from THIS paragraph alone (not from other "
    "parts of the paper).\n"
    "Answers must be short (one sentence, under 30 words) and unambiguous.\n"
    "Output a JSON object with field `qas`, an array of objects each with "
    "question/reference_answer/qa_type fields.\n"
    + STRICT_JSON_FOOTER
)


_QA_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "qas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "reference_answer": {"type": "string"},
                    "qa_type": {
                        "type": "string",
                        "enum": ["numeric", "definition", "comparison", "cause_effect"],
                    },
                },
                "required": ["question", "reference_answer", "qa_type"],
                "additionalProperties": False,
            },
            "minItems": 3, "maxItems": 3,
        }
    },
    "required": ["qas"],
    "additionalProperties": False,
}


async def generate_qa_for_paper(session: AsyncSession, paper_id: int) -> int:
    """Generate QA pairs for every unit of a paper. Idempotent."""
    # Skip if some QA already exists
    n_existing = await session.scalar(
        select(func.count(ComprehensionQA.id))
        .join(TranslationUnit, ComprehensionQA.unit_id == TranslationUnit.id)
        .where(TranslationUnit.paper_id == paper_id)
    ) or 0
    if n_existing > 0:
        return n_existing

    units = (
        await session.execute(
            select(TranslationUnit)
            .where(TranslationUnit.paper_id == paper_id)
            .order_by(TranslationUnit.ord)
        )
    ).scalars().all()

    provider = create_provider(settings.model_fact_extract)  # SOTA model for QA gen
    sem = asyncio.Semaphore(settings.max_concurrency)

    async def one(unit: TranslationUnit) -> list[ComprehensionQA]:
        if not unit.src_plain or len(unit.src_plain) < 60:
            return []
        async with sem:
            try:
                tr = await provider.generate(
                    prompt=f"<SOURCE>{unit.src_plain}</SOURCE>",
                    system=_QA_GEN_SYSTEM,
                    json_schema=_QA_GEN_SCHEMA,
                )
            except Exception as e:
                logger.warning("QA gen failed for unit %d: %s", unit.id, e)
                return []
        s = (tr.text or "").strip()
        if s.startswith("```"):
            s = s.strip("`").lstrip("json").strip()
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return []
        qas = obj.get("qas") if isinstance(obj, dict) else None
        if not isinstance(qas, list):
            return []
        out = []
        for q in qas:
            # qa_type column is VARCHAR(20); the model sometimes returns combined
            # labels like "comparison/cause_effect" (23 chars). Normalise to the
            # first label and cap length.
            qtype = (q.get("qa_type") or "").split("/")[0].strip()[:20] or None
            out.append(ComprehensionQA(
                unit_id=unit.id,
                question=q["question"],
                question_lang="en",
                reference_answer=q["reference_answer"],
                qa_type=qtype,
                generated_by=provider.model_name,
            ))
        return out

    results = await asyncio.gather(*(one(u) for u in units))
    total = 0
    for batch in results:
        for q in batch:
            session.add(q)
            total += 1
    await session.commit()
    logger.info("QA gen: %d pairs for paper %d", total, paper_id)
    return total


# ────────────────────────── 2. answer + score ──────────────────────────


def _reader_system(target_name: str) -> str:
    return (
        f"You are a reader. Below is a {target_name} translation of a scientific-paper "
        "paragraph, then a few questions in English.\n"
        "Answer briefly in English based ONLY on the translation (if it cannot be answered "
        "from the translation, answer 'unanswerable').\n"
        "Output a JSON object with field `answer`.\n"
        + STRICT_JSON_FOOTER
    )


_READER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


# Compares English reference vs English reader answer — target-neutral.
_SCORER_SYSTEM = (
    "You grade reading-comprehension answers. Given a reference answer and a reader's answer, "
    "judge the reader's correctness.\n"
    "Scoring: 1 = fully correct, 0.5 = partially correct, 0 = wrong or no answer.\n"
    "Output a JSON object with field `correctness` (number 0/0.5/1) and `reasoning` (brief).\n"
    + STRICT_JSON_FOOTER
)


_SCORER_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["correctness"],
    "additionalProperties": False,
}


async def score_translation(
    session: AsyncSession, job_id: int, paper_id: int,
) -> dict:
    """For one job: have reader LLM answer QA against this job's translation,
    then score against reference. Persist comprehension_responses + RCS."""
    qas = (
        await session.execute(
            select(ComprehensionQA, TranslationUnit)
            .join(TranslationUnit, ComprehensionQA.unit_id == TranslationUnit.id)
            .where(TranslationUnit.paper_id == paper_id)
        )
    ).all()
    if not qas:
        return {"rcs_paper": None, "answered": 0}

    job_lang = (
        await session.execute(
            select(TranslationJob.language).where(TranslationJob.id == job_id)
        )
    ).scalar_one_or_none()
    reader_system = _reader_system(language_name(job_lang) if job_lang else "the target language")

    # Latest translation per unit for this job
    trans = (
        await session.execute(
            select(Translation).where(Translation.job_id == job_id)
            .order_by(Translation.unit_id, Translation.pass_no.desc())
        )
    ).scalars().all()
    latest: dict[int, Translation] = {}
    for t in trans:
        latest.setdefault(t.unit_id, t)

    reader = create_provider(settings.model_qa_reader)
    scorer = create_provider(settings.model_qa_judge)
    sem = asyncio.Semaphore(settings.max_concurrency)

    async def one(qa: ComprehensionQA, unit: TranslationUnit) -> ComprehensionResponse | None:
        t = latest.get(unit.id)
        if t is None or not t.tgt_plain or t.status == TuStatus.pending:
            return None
        async with sem:
            # answer
            try:
                ans_tr = await reader.generate(
                    prompt=(
                        f"<TRANSLATION>{t.tgt_plain}</TRANSLATION>\n"
                        f"<QUESTION>{qa.question}</QUESTION>"
                    ),
                    system=reader_system,
                    json_schema=_READER_SCHEMA,
                )
            except Exception as e:
                logger.warning("RCS answer failed: %s", e)
                return None
            s = (ans_tr.text or "").strip()
            if s.startswith("```"):
                s = s.strip("`").lstrip("json").strip()
            try:
                ans_obj = json.loads(s)
                answer = ans_obj.get("answer", "")
            except json.JSONDecodeError:
                answer = ""

            # score
            try:
                sc_tr = await scorer.generate(
                    prompt=(
                        f"<QUESTION>{qa.question}</QUESTION>\n"
                        f"<REFERENCE>{qa.reference_answer}</REFERENCE>\n"
                        f"<READER_ANSWER>{answer}</READER_ANSWER>"
                    ),
                    system=_SCORER_SYSTEM,
                    json_schema=_SCORER_SCHEMA,
                )
            except Exception as e:
                logger.warning("RCS score failed: %s", e)
                return None
            s = (sc_tr.text or "").strip()
            if s.startswith("```"):
                s = s.strip("`").lstrip("json").strip()
            try:
                sc_obj = json.loads(s)
                correctness = float(sc_obj.get("correctness", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                correctness = 0.0

        return ComprehensionResponse(
            qa_id=qa.id, translation_id=t.id, answer=answer,
            correctness=correctness,
            responder_model=reader.model_name,
            judge_model=scorer.model_name,
        )

    # Idempotent resume: skip QA already scored for this job, so a re-run after
    # a crash/quota-stop continues instead of duplicating responses.
    answered_qa_ids = set(
        (
            await session.execute(
                select(ComprehensionResponse.qa_id)
                .join(Translation, ComprehensionResponse.translation_id == Translation.id)
                .where(Translation.job_id == job_id)
            )
        ).scalars().all()
    )
    todo = [(qa, u) for qa, u in qas if qa.id not in answered_qa_ids]

    # Process in chunks and commit each, so partial progress survives a failure
    # (a single hung/slow call no longer risks losing the whole batch).
    chunk = max(settings.max_concurrency * 2, 8)
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        results = await asyncio.gather(*(one(qa, u) for qa, u in batch))
        for r in results:
            if r is not None:
                session.add(r)
        await session.commit()
        logger.info("RCS: scored %d/%d (job %d)",
                    min(i + chunk, len(todo)), len(todo), job_id)

    # Aggregate over ALL responses for this job (incl. any prior partial run).
    rows = (
        await session.execute(
            select(ComprehensionResponse.correctness, Translation.unit_id)
            .join(Translation, ComprehensionResponse.translation_id == Translation.id)
            .where(
                Translation.job_id == job_id,
                ComprehensionResponse.correctness.is_not(None),
            )
        )
    ).all()
    correct = [c for c, _ in rows]
    per_unit: dict[int, list[float]] = {}
    for c, uid in rows:
        per_unit.setdefault(uid, []).append(c)
    for unit_id, scores in per_unit.items():
        t = latest.get(unit_id)
        if t is not None:
            t.rcs_unit = statistics.mean(scores)
    await session.commit()

    rcs_paper = statistics.mean(correct) if correct else None
    return {"rcs_paper": rcs_paper, "answered": len(correct)}

"""SQLAlchemy 2.0 async ORM matching arxiv_translation_hub.sql v3 schema."""

from __future__ import annotations

import datetime as dt
import enum
from typing import Any

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    TIMESTAMP,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ENUM as PGEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from xinda.db.engine import Base


# ──────────────────────────── ENUM types ────────────────────────────────────

class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"
    partial = "partial"


class PipelineStage(str, enum.Enum):
    acquire = "acquire"
    convert = "convert"
    extract = "extract"
    fact_extract = "fact_extract"
    glossary = "glossary"
    translate = "translate"
    fact_verify = "fact_verify"
    cross_doc_verify = "cross_doc_verify"
    refine = "refine"
    coherence = "coherence"
    apply = "apply"
    render = "render"
    evaluate = "evaluate"


class UnitKind(str, enum.Enum):
    title = "title"
    section_heading = "section_heading"
    abstract = "abstract"
    paragraph = "paragraph"
    caption = "caption"
    item = "item"


class TuStatus(str, enum.Enum):
    pending = "pending"
    translated = "translated"
    refined = "refined"
    accepted = "accepted"
    fallback = "fallback"


class RunVariant(str, enum.Enum):
    full = "full"
    no_glossary = "no_glossary"
    no_context = "no_context"
    no_fact_anchor = "no_fact_anchor"
    no_fact_verify = "no_fact_verify"
    no_cross_doc = "no_cross_doc"
    no_coherence = "no_coherence"
    no_external_glossary = "no_external_glossary"
    no_retry = "no_retry"
    baseline_qwen_turbo = "baseline_qwen_turbo"
    baseline_qwen_plus = "baseline_qwen_plus"
    baseline_qwen37_max = "baseline_qwen37_max"


class ClaimType(str, enum.Enum):
    numeric = "numeric"
    citation = "citation"
    comparison = "comparison"
    method_name = "method_name"
    symbol = "symbol"


class DriftType(str, enum.Enum):
    verified = "verified"
    missing = "missing"
    numeric_drift = "numeric_drift"
    citation_swap = "citation_swap"
    comparison_flip = "comparison_flip"
    symbol_drift = "symbol_drift"
    method_drift = "method_drift"
    partial = "partial"


# SQLAlchemy PGEnum (create_type=False because SQL file already creates them)
_job_status_t = PGEnum(JobStatus, name="job_status", create_type=False)
_pipeline_stage_t = PGEnum(PipelineStage, name="pipeline_stage", create_type=False)
_unit_kind_t = PGEnum(UnitKind, name="unit_kind", create_type=False)
_tu_status_t = PGEnum(TuStatus, name="tu_status", create_type=False)
_run_variant_t = PGEnum(RunVariant, name="run_variant", create_type=False)
_claim_type_t = PGEnum(ClaimType, name="claim_type", create_type=False)
_drift_type_t = PGEnum(DriftType, name="drift_type", create_type=False)


# ──────────────────────────── 1. papers ─────────────────────────────────────

class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[str | None] = mapped_column(Text)
    source_abstract: Mapped[str | None] = mapped_column(Text)
    main_category: Mapped[str | None] = mapped_column(String(50))
    categories: Mapped[str | None] = mapped_column(String(255))
    field: Mapped[str | None] = mapped_column(String(100))
    published: Mapped[dt.date | None] = mapped_column(Date)
    updated: Mapped[dt.date | None] = mapped_column(Date)
    pdf_url: Mapped[str | None] = mapped_column(Text)
    # arXiv/CC license URL + whether it permits redistributing a translation
    # (CC0/CC-BY/CC-BY-SA → permissive; see arxiv_meta.classify_license)
    license: Mapped[str | None] = mapped_column(Text)
    license_label: Mapped[str | None] = mapped_column(String(40))
    license_permissive: Mapped[bool | None] = mapped_column(Boolean)
    source_path: Mapped[str | None] = mapped_column(Text)
    download_status: Mapped[str] = mapped_column(String(20), default="pending")
    extract_status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[dt.datetime] = mapped_column(TIMESTAMP, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )

    sections: Mapped[list["Section"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    units: Mapped[list["TranslationUnit"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["TranslationJob"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    glossary_terms: Mapped[list["GlossaryTerm"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


# ──────────────────────────── 2. sections ───────────────────────────────────

class Section(Base):
    __tablename__ = "sections"
    __table_args__ = (UniqueConstraint("paper_id", "xpath"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("sections.id", ondelete="CASCADE")
    )
    ord: Mapped[int] = mapped_column(Integer, nullable=False)
    depth: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    xpath: Mapped[str] = mapped_column(Text, nullable=False)
    heading_src: Mapped[str | None] = mapped_column(Text)

    paper: Mapped["Paper"] = relationship(back_populates="sections")
    parent: Mapped["Section | None"] = relationship(
        "Section", remote_side="Section.id", backref="children"
    )
    units: Mapped[list["TranslationUnit"]] = relationship(back_populates="section")


# ──────────────────────────── 3. translation_units ──────────────────────────

class TranslationUnit(Base):
    __tablename__ = "translation_units"
    __table_args__ = (UniqueConstraint("paper_id", "xpath"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    section_id: Mapped[int | None] = mapped_column(
        ForeignKey("sections.id", ondelete="SET NULL")
    )
    ord: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[UnitKind] = mapped_column(_unit_kind_t, nullable=False)
    xpath: Mapped[str] = mapped_column(Text, nullable=False)
    src_text: Mapped[str] = mapped_column(Text, nullable=False)
    src_plain: Mapped[str] = mapped_column(Text, nullable=False)
    placeholders: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    special_chars: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    paper: Mapped["Paper"] = relationship(back_populates="units")
    section: Mapped["Section | None"] = relationship(back_populates="units")
    claims: Mapped[list["VerifiableClaim"]] = relationship(
        back_populates="unit", cascade="all, delete-orphan"
    )
    translations: Mapped[list["Translation"]] = relationship(
        back_populates="unit", cascade="all, delete-orphan"
    )


# ──────────────────────────── 4. verifiable_claims ──────────────────────────

class VerifiableClaim(Base):
    __tablename__ = "verifiable_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    unit_id: Mapped[int] = mapped_column(
        ForeignKey("translation_units.id", ondelete="CASCADE"), nullable=False
    )
    claim_type: Mapped[ClaimType] = mapped_column(_claim_type_t, nullable=False)
    span_start: Mapped[int | None] = mapped_column(Integer)
    span_end: Mapped[int | None] = mapped_column(Integer)
    surface_form: Mapped[str] = mapped_column(Text, nullable=False)
    normalized: Mapped[str] = mapped_column(Text, nullable=False)
    claim_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict
    )
    extracted_by: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

    unit: Mapped["TranslationUnit"] = relationship(back_populates="claims")
    verifications: Mapped[list["ClaimVerification"]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )


# ──────────────────────────── 5. translation_jobs ───────────────────────────

class TranslationJob(Base):
    __tablename__ = "translation_jobs"
    __table_args__ = (
        UniqueConstraint("paper_id", "language", "variant", "config_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    language: Mapped[str] = mapped_column(String(10), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), default="qianwen", nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    refine_model: Mapped[str | None] = mapped_column(String(100))
    variant: Mapped[RunVariant] = mapped_column(
        _run_variant_t, default=RunVariant.full, nullable=False
    )
    config_hash: Mapped[str] = mapped_column(String(12), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        _job_status_t, default=JobStatus.pending, nullable=False
    )
    last_stage: Mapped[PipelineStage | None] = mapped_column(_pipeline_stage_t)
    start_time: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP)
    end_time: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP)
    error_msg: Mapped[str | None] = mapped_column(Text)
    output_dir: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

    paper: Mapped["Paper"] = relationship(back_populates="jobs")
    translations: Mapped[list["Translation"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    cross_doc_drifts: Mapped[list["CrossDocDrift"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    evaluation_run: Mapped["EvaluationRun | None"] = relationship(
        back_populates="job", uselist=False, cascade="all, delete-orphan"
    )
    rendered_files: Mapped[list["RenderedFile"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    eval_samples: Mapped[list["EvalSample"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


# ──────────────────────────── 6. translations ───────────────────────────────

class Translation(Base):
    __tablename__ = "translations"
    __table_args__ = (UniqueConstraint("job_id", "unit_id", "pass_no"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("translation_jobs.id", ondelete="CASCADE"), nullable=False
    )
    unit_id: Mapped[int] = mapped_column(
        ForeignKey("translation_units.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[TuStatus] = mapped_column(
        _tu_status_t, default=TuStatus.pending, nullable=False
    )
    pass_no: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    tgt_text: Mapped[str | None] = mapped_column(Text)
    tgt_plain: Mapped[str | None] = mapped_column(Text)
    comet_score: Mapped[float | None] = mapped_column(Float)
    xcomet_score: Mapped[float | None] = mapped_column(Float)
    ppa_unit: Mapped[float | None] = mapped_column(Float)
    fps_unit: Mapped[float | None] = mapped_column(Float)
    rcs_unit: Mapped[float | None] = mapped_column(Float)
    glossary_hits: Mapped[Any | None] = mapped_column(JSONB)
    cached_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    fresh_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer)
    batch_request_id: Mapped[str | None] = mapped_column(String(100))
    refined_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP)

    job: Mapped["TranslationJob"] = relationship(back_populates="translations")
    unit: Mapped["TranslationUnit"] = relationship(back_populates="translations")
    verifications: Mapped[list["ClaimVerification"]] = relationship(
        back_populates="translation", cascade="all, delete-orphan"
    )
    comprehension_responses: Mapped[list["ComprehensionResponse"]] = relationship(
        back_populates="translation", cascade="all, delete-orphan"
    )


# ──────────────────────────── 7. claim_verifications ────────────────────────

class ClaimVerification(Base):
    __tablename__ = "claim_verifications"
    __table_args__ = (UniqueConstraint("translation_id", "claim_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    translation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("translations.id", ondelete="CASCADE"),
        nullable=False,
    )
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("verifiable_claims.id", ondelete="CASCADE"), nullable=False
    )
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False)
    drift: Mapped[DriftType] = mapped_column(
        _drift_type_t, default=DriftType.verified, nullable=False
    )
    tgt_surface: Mapped[str | None] = mapped_column(Text)
    tgt_normalized: Mapped[str | None] = mapped_column(Text)
    drift_magnitude: Mapped[float | None] = mapped_column(Float)
    verifier: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

    translation: Mapped["Translation"] = relationship(back_populates="verifications")
    claim: Mapped["VerifiableClaim"] = relationship(back_populates="verifications")


# ──────────────────────────── 8. cross_doc_drifts ───────────────────────────

class CrossDocDrift(Base):
    __tablename__ = "cross_doc_drifts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("translation_jobs.id", ondelete="CASCADE"), nullable=False
    )
    drift_type: Mapped[str | None] = mapped_column(String(30))
    unit_ids: Mapped[list[int] | None] = mapped_column(ARRAY(Integer))
    surface_forms: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    severity: Mapped[str | None] = mapped_column(String(10))
    description: Mapped[str | None] = mapped_column(Text)
    detected_by: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

    job: Mapped["TranslationJob"] = relationship(back_populates="cross_doc_drifts")


# ──────────────────────────── 9. glossary_terms ─────────────────────────────

class GlossaryTerm(Base):
    __tablename__ = "glossary_terms"
    __table_args__ = (UniqueConstraint("paper_id", "language", "src_term"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    language: Mapped[str] = mapped_column(String(10), nullable=False)
    src_term: Mapped[str] = mapped_column(Text, nullable=False)
    tgt_term: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str | None] = mapped_column(String(20))
    definition: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    # which external term bank grounded tgt_term (None = LLM-only)
    grounding_source: Mapped[str | None] = mapped_column(String(20))
    source_unit_id: Mapped[int | None] = mapped_column(
        ForeignKey("translation_units.id")
    )
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

    paper: Mapped["Paper"] = relationship(back_populates="glossary_terms")


# ──────────────────────────── 10. evaluation_runs ───────────────────────────

class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("translation_jobs.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    comet_mean: Mapped[float | None] = mapped_column(Float)
    comet_median: Mapped[float | None] = mapped_column(Float)
    comet_p10: Mapped[float | None] = mapped_column(Float)
    xcomet_mean: Mapped[float | None] = mapped_column(Float)
    ppa: Mapped[float | None] = mapped_column(Float)
    ppa_ordered: Mapped[float | None] = mapped_column(Float)
    mfr: Mapped[float | None] = mapped_column(Float)
    mfr_ordered: Mapped[float | None] = mapped_column(Float)
    tcr: Mapped[float | None] = mapped_column(Float)
    fps_paper: Mapped[float | None] = mapped_column(Float)
    fps_numeric: Mapped[float | None] = mapped_column(Float)
    fps_citation: Mapped[float | None] = mapped_column(Float)
    fps_comparison: Mapped[float | None] = mapped_column(Float)
    fps_method: Mapped[float | None] = mapped_column(Float)
    fps_symbol: Mapped[float | None] = mapped_column(Float)
    rcs_paper: Mapped[float | None] = mapped_column(Float)
    drift_numeric_count: Mapped[int | None] = mapped_column(Integer)
    drift_citation_count: Mapped[int | None] = mapped_column(Integer)
    drift_comparison_count: Mapped[int | None] = mapped_column(Integer)
    drift_method_count: Mapped[int | None] = mapped_column(Integer)
    drift_symbol_count: Mapped[int | None] = mapped_column(Integer)
    drift_missing_count: Mapped[int | None] = mapped_column(Integer)
    pass1_units: Mapped[int | None] = mapped_column(Integer)
    refined_units: Mapped[int | None] = mapped_column(Integer)
    fallback_units: Mapped[int | None] = mapped_column(Integer)
    total_units: Mapped[int | None] = mapped_column(Integer)
    total_claims: Mapped[int | None] = mapped_column(Integer)
    total_prompt_tok: Mapped[int | None] = mapped_column(Integer)
    total_cached_tok: Mapped[int | None] = mapped_column(Integer)
    total_completion_tok: Mapped[int | None] = mapped_column(Integer)
    cost_cny: Mapped[float | None] = mapped_column(Float)  # translation token cost (DashScope list price)
    wallclock_sec: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

    job: Mapped["TranslationJob"] = relationship(back_populates="evaluation_run")


# ──────────────────────────── 11. rendered_files ────────────────────────────

class RenderedFile(Base):
    __tablename__ = "rendered_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("translation_jobs.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str | None] = mapped_column(String(20))
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

    job: Mapped["TranslationJob"] = relationship(back_populates="rendered_files")


# ──────────────────────────── 12. eval_samples ──────────────────────────────

class EvalSample(Base):
    __tablename__ = "eval_samples"
    __table_args__ = (UniqueConstraint("job_id", "unit_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("translation_jobs.id", ondelete="CASCADE"), nullable=False
    )
    unit_id: Mapped[int] = mapped_column(
        ForeignKey("translation_units.id", ondelete="CASCADE"), nullable=False
    )
    sampling_kind: Mapped[str | None] = mapped_column(String(30))

    job: Mapped["TranslationJob"] = relationship(back_populates="eval_samples")
    judgments: Mapped[list["EvalJudgment"]] = relationship(
        back_populates="sample", cascade="all, delete-orphan"
    )


# ──────────────────────────── 13. eval_judgments ────────────────────────────

class EvalJudgment(Base):
    __tablename__ = "eval_judgments"
    __table_args__ = (
        UniqueConstraint("sample_id", "judge_model", "protocol", "run_no"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sample_id: Mapped[int] = mapped_column(
        ForeignKey("eval_samples.id", ondelete="CASCADE"), nullable=False
    )
    judge_model: Mapped[str] = mapped_column(String(100), nullable=False)
    protocol: Mapped[str] = mapped_column(String(20), nullable=False)
    run_no: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)
    fluency: Mapped[float | None] = mapped_column(Float)
    adequacy: Mapped[float | None] = mapped_column(Float)
    terminology: Mapped[float | None] = mapped_column(Float)
    structure: Mapped[float | None] = mapped_column(Float)
    rubric_score: Mapped[float | None] = mapped_column(Float)
    xcomet_score: Mapped[float | None] = mapped_column(Float)
    raw_response: Mapped[Any | None] = mapped_column(JSONB)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer)
    batch_request_id: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

    sample: Mapped["EvalSample"] = relationship(back_populates="judgments")
    mqm_errors: Mapped[list["EvalMqmError"]] = relationship(
        back_populates="judgment", cascade="all, delete-orphan"
    )


class EvalMqmError(Base):
    __tablename__ = "eval_mqm_errors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    judgment_id: Mapped[int] = mapped_column(
        ForeignKey("eval_judgments.id", ondelete="CASCADE"), nullable=False
    )
    category: Mapped[str | None] = mapped_column(String(40))
    severity: Mapped[str | None] = mapped_column(String(10))
    span_text: Mapped[str | None] = mapped_column(Text)
    explanation: Mapped[str | None] = mapped_column(Text)

    judgment: Mapped["EvalJudgment"] = relationship(back_populates="mqm_errors")


# ──────────────────────────── 14. fact_traps ────────────────────────────────

class FactTrap(Base):
    __tablename__ = "fact_traps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"), nullable=False)
    unit_id: Mapped[int] = mapped_column(
        ForeignKey("translation_units.id"), nullable=False
    )
    trap_type: Mapped[str | None] = mapped_column(String(30))
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    trapped_text: Mapped[str] = mapped_column(Text, nullable=False)
    trap_metadata: Mapped[Any | None] = mapped_column(JSONB)
    expected_detection: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )


# ──────────────────────────── 15-16. comprehension_qa + responses ───────────

class ComprehensionQA(Base):
    __tablename__ = "comprehension_qa"
    __table_args__ = (UniqueConstraint("unit_id", "question"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    unit_id: Mapped[int] = mapped_column(
        ForeignKey("translation_units.id", ondelete="CASCADE"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    question_lang: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    reference_answer: Mapped[str] = mapped_column(Text, nullable=False)
    qa_type: Mapped[str | None] = mapped_column(String(20))
    generated_by: Mapped[str | None] = mapped_column(String(100))

    responses: Mapped[list["ComprehensionResponse"]] = relationship(
        back_populates="qa", cascade="all, delete-orphan"
    )


class ComprehensionResponse(Base):
    __tablename__ = "comprehension_responses"
    __table_args__ = (UniqueConstraint("qa_id", "translation_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    qa_id: Mapped[int] = mapped_column(
        ForeignKey("comprehension_qa.id", ondelete="CASCADE"), nullable=False
    )
    translation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("translations.id", ondelete="CASCADE"),
        nullable=False,
    )
    answer: Mapped[str | None] = mapped_column(Text)
    correctness: Mapped[float | None] = mapped_column(Float)
    responder_model: Mapped[str | None] = mapped_column(String(100))
    judge_model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

    qa: Mapped["ComprehensionQA"] = relationship(back_populates="responses")
    translation: Mapped["Translation"] = relationship(
        back_populates="comprehension_responses"
    )


class FhrResult(Base):
    """Fidelity Honesty Rate per (system, language, trap_type).

    The headline experiment table: one row per (translation system × language ×
    trap type), plus a trap_type=NULL overall row. `system` names the condition
    compared in the paper (naive | fidelity | fact_anchor | <external model>).
    """

    __tablename__ = "fhr_results"
    __table_args__ = (UniqueConstraint("system", "language", "trap_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    system: Mapped[str] = mapped_column(String(40), nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False)
    trap_type: Mapped[str | None] = mapped_column(String(30))  # NULL = overall
    total: Mapped[int | None] = mapped_column(Integer)
    faithful: Mapped[int | None] = mapped_column(Integer)
    fhr: Mapped[float | None] = mapped_column(Float)
    judged: Mapped[bool] = mapped_column(Boolean, default=False)
    model_used: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=dt.datetime.utcnow
    )

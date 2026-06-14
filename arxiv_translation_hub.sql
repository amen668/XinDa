-- ============================================================================
-- foundation-translator schema v3 (Fidelity-First)
-- ============================================================================
-- 14 tables + 6 ENUM types
-- Apply with: psql -d arxiv_translation_hub -f arxiv_translation_hub.sql
-- ============================================================================

-- ============================================================================
-- ENUM types
-- ============================================================================

CREATE TYPE job_status AS ENUM (
    'pending', 'running', 'success', 'failed', 'partial'
);

CREATE TYPE pipeline_stage AS ENUM (
    'acquire', 'convert', 'extract', 'fact_extract', 'glossary',
    'translate', 'fact_verify', 'cross_doc_verify', 'refine',
    'coherence', 'apply', 'render', 'evaluate'
);

CREATE TYPE unit_kind AS ENUM (
    'title', 'section_heading', 'abstract', 'paragraph', 'caption', 'item'
);

CREATE TYPE tu_status AS ENUM (
    'pending', 'translated', 'refined', 'accepted', 'fallback'
);

CREATE TYPE run_variant AS ENUM (
    'full',
    'no_glossary', 'no_context', 'no_fact_anchor', 'no_fact_verify',
    'no_cross_doc', 'no_coherence', 'no_external_glossary', 'no_retry',
    'baseline_qwen_turbo', 'baseline_qwen_plus', 'baseline_qwen37_max'
);

CREATE TYPE claim_type AS ENUM (
    'numeric', 'citation', 'comparison', 'method_name', 'symbol'
);

CREATE TYPE drift_type AS ENUM (
    'verified', 'missing',
    'numeric_drift', 'citation_swap', 'comparison_flip',
    'symbol_drift', 'method_drift', 'partial'
);


-- ============================================================================
-- 1. papers — paper metadata
-- ============================================================================

CREATE TABLE papers (
    id                  SERIAL PRIMARY KEY,
    arxiv_id            VARCHAR(50) UNIQUE NOT NULL,
    title               TEXT NOT NULL,
    authors             TEXT,
    source_abstract     TEXT,
    main_category       VARCHAR(50),
    categories          VARCHAR(255),
    field               VARCHAR(100),
    published           DATE,
    updated             DATE,
    pdf_url             TEXT,
    license             TEXT,           -- arXiv/CC license URL
    license_label       VARCHAR(40),    -- CC0 / CC-BY / CC-BY-SA / arXiv-nonexclusive / …
    license_permissive  BOOLEAN,        -- TRUE iff publishing a translation is permitted
    source_path         TEXT,
    download_status     VARCHAR(20) DEFAULT 'pending',
    extract_status      VARCHAR(20) DEFAULT 'pending',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_papers_field ON papers(field);


-- ============================================================================
-- 2. sections — document section tree (parent_id self-reference)
-- ============================================================================

CREATE TABLE sections (
    id          SERIAL PRIMARY KEY,
    paper_id    INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    parent_id   INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,
    depth       SMALLINT NOT NULL,
    xpath       TEXT NOT NULL,
    heading_src TEXT,
    UNIQUE (paper_id, xpath)
);

CREATE INDEX idx_sections_paper_ord ON sections(paper_id, depth, ord);


-- ============================================================================
-- 3. translation_units — paragraph-grained translation source
-- ============================================================================

CREATE TABLE translation_units (
    id              SERIAL PRIMARY KEY,
    paper_id        INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id      INTEGER REFERENCES sections(id) ON DELETE SET NULL,
    ord             INTEGER NOT NULL,
    kind            unit_kind NOT NULL,
    xpath           TEXT NOT NULL,
    src_text        TEXT NOT NULL,                            -- 带 {{PT_*}} 占位符
    src_plain       TEXT NOT NULL,                            -- 去占位符,COMET/token 计数用
    placeholders    JSONB NOT NULL DEFAULT '{}'::jsonb,
    special_chars   JSONB NOT NULL DEFAULT '{}'::jsonb,
    char_count      INTEGER NOT NULL,
    UNIQUE (paper_id, xpath)
);

CREATE INDEX idx_units_paper_ord ON translation_units(paper_id, ord);
CREATE INDEX idx_units_section   ON translation_units(section_id);


-- ============================================================================
-- 4. verifiable_claims — Fact-Anchor Protocol core table (C1)
-- ============================================================================

CREATE TABLE verifiable_claims (
    id              SERIAL PRIMARY KEY,
    unit_id         INTEGER NOT NULL REFERENCES translation_units(id) ON DELETE CASCADE,
    claim_type      claim_type NOT NULL,
    span_start      INTEGER,                                  -- char offset in src_plain
    span_end        INTEGER,
    surface_form    TEXT NOT NULL,                            -- raw substring, e.g. "93.4%"
    normalized      TEXT NOT NULL,                            -- canonical, e.g. "93.4" or "Vaswani2017"
    metadata        JSONB DEFAULT '{}'::jsonb,
    extracted_by    VARCHAR(100),
    confidence      REAL DEFAULT 1.0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_claims_unit ON verifiable_claims(unit_id);
CREATE INDEX idx_claims_type ON verifiable_claims(claim_type);


-- ============================================================================
-- 5. translation_jobs — one row per pipeline run for a (paper, lang, variant)
-- ============================================================================

CREATE TABLE translation_jobs (
    id              SERIAL PRIMARY KEY,
    paper_id        INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    language        VARCHAR(10) NOT NULL,                     -- 'zh' | 'fr' | 'es'
    provider        VARCHAR(50) NOT NULL DEFAULT 'qianwen',
    model_name      VARCHAR(100) NOT NULL,                    -- pass-1 model
    refine_model    VARCHAR(100),                             -- NULL if no_retry
    variant         run_variant NOT NULL DEFAULT 'full',
    config_hash     CHAR(12) NOT NULL,
    status          job_status NOT NULL DEFAULT 'pending',
    last_stage      pipeline_stage,                           -- resumability cursor
    start_time      TIMESTAMP,
    end_time        TIMESTAMP,
    error_msg       TEXT,
    output_dir      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (paper_id, language, variant, config_hash)
);

CREATE INDEX idx_jobs_paper_variant ON translation_jobs(paper_id, variant, language);
CREATE INDEX idx_jobs_status        ON translation_jobs(status);


-- ============================================================================
-- 6. translations — translated text per (job, unit, pass)
-- ============================================================================

CREATE TABLE translations (
    id                      BIGSERIAL PRIMARY KEY,
    job_id                  INTEGER NOT NULL REFERENCES translation_jobs(id) ON DELETE CASCADE,
    unit_id                 INTEGER NOT NULL REFERENCES translation_units(id) ON DELETE CASCADE,
    status                  tu_status NOT NULL DEFAULT 'pending',
    pass_no                 SMALLINT NOT NULL DEFAULT 1,
    model_used              VARCHAR(100) NOT NULL,
    tgt_text                TEXT,                             -- with {{PT_*}}
    tgt_plain               TEXT,                             -- placeholder-stripped
    comet_score             REAL,
    xcomet_score            REAL,                             -- headline metric
    ppa_unit                REAL,
    fps_unit                REAL,                             -- segment Fidelity Preservation Score
    rcs_unit                REAL,                             -- segment Reader Comprehension Score
    glossary_hits           JSONB,
    cached_prompt_tokens    INTEGER DEFAULT 0,                -- DashScope cache hits
    fresh_prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens       INTEGER DEFAULT 0,
    elapsed_ms              INTEGER,
    batch_request_id        VARCHAR(100),                     -- DashScope Batch API correlation
    refined_at              TIMESTAMP,
    UNIQUE (job_id, unit_id, pass_no)
);

CREATE INDEX idx_tr_job_status ON translations(job_id, status);
CREATE INDEX idx_tr_comet      ON translations(job_id, comet_score);
CREATE INDEX idx_tr_fps        ON translations(job_id, fps_unit);


-- ============================================================================
-- 7. claim_verifications — Fact-Verify outcome per (translation, claim) (C2)
-- ============================================================================

CREATE TABLE claim_verifications (
    id              SERIAL PRIMARY KEY,
    translation_id  BIGINT NOT NULL REFERENCES translations(id) ON DELETE CASCADE,
    claim_id        INTEGER NOT NULL REFERENCES verifiable_claims(id) ON DELETE CASCADE,
    verified        BOOLEAN NOT NULL,
    drift           drift_type NOT NULL DEFAULT 'verified',
    tgt_surface     TEXT,
    tgt_normalized  TEXT,
    drift_magnitude REAL,
    verifier        VARCHAR(100),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (translation_id, claim_id)
);

CREATE INDEX idx_verif_drift  ON claim_verifications(drift) WHERE verified = FALSE;
CREATE INDEX idx_verif_trans  ON claim_verifications(translation_id);


-- ============================================================================
-- 8. cross_doc_drifts — whole-paper inconsistencies (1M context capability)
-- ============================================================================

CREATE TABLE cross_doc_drifts (
    id              SERIAL PRIMARY KEY,
    job_id          INTEGER NOT NULL REFERENCES translation_jobs(id) ON DELETE CASCADE,
    drift_type      VARCHAR(30),
    unit_ids        INTEGER[],
    surface_forms   TEXT[],
    severity        VARCHAR(10),
    description     TEXT,
    detected_by     VARCHAR(100),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cross_drifts_job ON cross_doc_drifts(job_id);


-- ============================================================================
-- 9. glossary_terms — per-(paper, language) terminology table
-- ============================================================================

CREATE TABLE glossary_terms (
    id              SERIAL PRIMARY KEY,
    paper_id        INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    language        VARCHAR(10) NOT NULL,
    src_term        TEXT NOT NULL,
    tgt_term        TEXT NOT NULL,
    kind            VARCHAR(20),                              -- 'acronym' | 'proper_noun' | 'technical_term'
    definition      TEXT,
    confidence      REAL DEFAULT 1.0,
    grounding_source VARCHAR(20),                             -- term bank that grounded tgt (NULL = LLM-only)
    source_unit_id  INTEGER REFERENCES translation_units(id),
    locked          BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (paper_id, language, src_term)
);

CREATE INDEX idx_glossary_paper_lang ON glossary_terms(paper_id, language);


-- ============================================================================
-- 10. evaluation_runs — aggregated per-job metrics
-- ============================================================================

CREATE TABLE evaluation_runs (
    id                      SERIAL PRIMARY KEY,
    job_id                  INTEGER NOT NULL UNIQUE REFERENCES translation_jobs(id) ON DELETE CASCADE,
    comet_mean              REAL,
    comet_median            REAL,
    comet_p10               REAL,
    xcomet_mean             REAL,                             -- headline
    ppa                     REAL,
    ppa_ordered             REAL,
    mfr                     REAL,
    mfr_ordered             REAL,
    tcr                     REAL,
    fps_paper               REAL,
    fps_numeric             REAL,
    fps_citation            REAL,
    fps_comparison          REAL,
    fps_method              REAL,
    fps_symbol              REAL,
    rcs_paper               REAL,
    drift_numeric_count     INTEGER,
    drift_citation_count    INTEGER,
    drift_comparison_count  INTEGER,
    drift_method_count      INTEGER,
    drift_symbol_count      INTEGER,
    drift_missing_count     INTEGER,
    pass1_units             INTEGER,
    refined_units           INTEGER,
    fallback_units          INTEGER,
    total_units             INTEGER,
    total_claims            INTEGER,
    total_prompt_tok        INTEGER,
    total_cached_tok        INTEGER,
    total_completion_tok    INTEGER,
    cost_cny                REAL,
    wallclock_sec           REAL,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================================
-- 11. rendered_files — output artifact paths
-- ============================================================================

CREATE TABLE rendered_files (
    id              SERIAL PRIMARY KEY,
    job_id          INTEGER NOT NULL REFERENCES translation_jobs(id) ON DELETE CASCADE,
    kind            VARCHAR(20),                              -- 'xml_src' | 'xml_tgt' | 'html_src' | 'html_tgt' | 'html_bilingual'
    storage_path    TEXT NOT NULL,
    size_bytes      INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rendered_job ON rendered_files(job_id);


-- ============================================================================
-- 12. eval_samples — stratified eval-sample registry for LLM-as-judge
-- ============================================================================

CREATE TABLE eval_samples (
    id              SERIAL PRIMARY KEY,
    job_id          INTEGER NOT NULL REFERENCES translation_jobs(id) ON DELETE CASCADE,
    unit_id         INTEGER NOT NULL REFERENCES translation_units(id) ON DELETE CASCADE,
    sampling_kind   VARCHAR(30),                              -- 'stratified_section' | 'random' | 'low_comet' | 'high_comet'
    UNIQUE (job_id, unit_id)
);


-- ============================================================================
-- 13. eval_judgments — LLM-as-judge outputs (RUBRIC-MQM / G-Eval / xCOMET)
-- ============================================================================

CREATE TABLE eval_judgments (
    id                  SERIAL PRIMARY KEY,
    sample_id           INTEGER NOT NULL REFERENCES eval_samples(id) ON DELETE CASCADE,
    judge_model         VARCHAR(100) NOT NULL,
    protocol            VARCHAR(20) NOT NULL,                 -- 'rubric_mqm' | 'g_eval' | 'xcomet'
    run_no              SMALLINT NOT NULL DEFAULT 1,
    fluency             REAL,
    adequacy            REAL,
    terminology         REAL,
    structure           REAL,
    rubric_score        REAL,
    xcomet_score        REAL,
    raw_response        JSONB,
    elapsed_ms          INTEGER,
    batch_request_id    VARCHAR(100),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (sample_id, judge_model, protocol, run_no)
);

CREATE INDEX idx_judgments_sample ON eval_judgments(sample_id);


CREATE TABLE eval_mqm_errors (
    id              SERIAL PRIMARY KEY,
    judgment_id     INTEGER NOT NULL REFERENCES eval_judgments(id) ON DELETE CASCADE,
    category        VARCHAR(40),
    severity        VARCHAR(10),
    span_text       TEXT,
    explanation     TEXT
);

CREATE INDEX idx_mqm_errors_judgment ON eval_mqm_errors(judgment_id);


-- ============================================================================
-- 14. fact_traps — adversarial fact-injection dataset (C3 stress test)
-- ============================================================================

CREATE TABLE fact_traps (
    id                  SERIAL PRIMARY KEY,
    paper_id            INTEGER NOT NULL REFERENCES papers(id),
    unit_id             INTEGER NOT NULL REFERENCES translation_units(id),
    trap_type           VARCHAR(30),                          -- 'numeric_subtle' | 'citation_year_swap' | 'comparison_reversal' | 'method_substitution' | 'symbol_change'
    original_text       TEXT NOT NULL,
    trapped_text        TEXT NOT NULL,
    trap_metadata       JSONB,
    expected_detection  BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_traps_type ON fact_traps(trap_type);


-- ============================================================================
-- 15. comprehension_qa — RCS evaluation: questions + reference answers
-- ============================================================================

CREATE TABLE comprehension_qa (
    id                  SERIAL PRIMARY KEY,
    unit_id             INTEGER NOT NULL REFERENCES translation_units(id) ON DELETE CASCADE,
    question            TEXT NOT NULL,
    question_lang       VARCHAR(10) NOT NULL DEFAULT 'en',
    reference_answer    TEXT NOT NULL,
    qa_type             VARCHAR(20),                          -- 'numeric' | 'definition' | 'comparison' | 'cause_effect'
    generated_by        VARCHAR(100),
    UNIQUE (unit_id, question)
);

CREATE INDEX idx_qa_unit ON comprehension_qa(unit_id);


-- ============================================================================
-- 16. comprehension_responses — reader-LLM answers + judge scores
-- ============================================================================

CREATE TABLE comprehension_responses (
    id                  SERIAL PRIMARY KEY,
    qa_id               INTEGER NOT NULL REFERENCES comprehension_qa(id) ON DELETE CASCADE,
    translation_id      BIGINT NOT NULL REFERENCES translations(id) ON DELETE CASCADE,
    answer              TEXT,
    correctness         REAL,                                 -- 0.0-1.0
    responder_model     VARCHAR(100),
    judge_model         VARCHAR(100),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (qa_id, translation_id)
);

CREATE INDEX idx_resp_translation ON comprehension_responses(translation_id);


-- ============================================================================
-- 17. fhr_results — Fidelity Honesty Rate per (system, language, trap_type)
--     The headline experiment: does a system faithfully keep deliberately
--     altered facts (high FHR) or "sycophantically" correct them (low FHR)?
--     `system` distinguishes the conditions compared in the paper:
--       naive | fidelity | fact_anchor | <external model id> | <pipeline variant>
--     trap_type NULL = overall (over the verbatim-verifiable anchor types).
-- ============================================================================

CREATE TABLE fhr_results (
    id                  SERIAL PRIMARY KEY,
    system              VARCHAR(40) NOT NULL,
    language            VARCHAR(10) NOT NULL,
    trap_type           VARCHAR(30),                          -- NULL = overall
    total               INTEGER,
    faithful            INTEGER,
    fhr                 REAL,
    judged              BOOLEAN DEFAULT FALSE,                -- TRUE if LLM-judged (comparison)
    model_used          VARCHAR(100),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (system, language, trap_type)
);

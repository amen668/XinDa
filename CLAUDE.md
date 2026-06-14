# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: XinDa（信达）

**`xinda/`** is the "Fidelity-First" v3 framework (named for 信达雅 — 信/fidelity first):
a 13-stage, DB-backed, resumable translation + evaluation pipeline driven by CLI entry
points. Config is `xinda/config.py` (Pydantic, env-driven). The old v1 FastAPI service
(`legacy/`, had committed API keys) was **deleted from the tree** pre-open-sourcing; it
survives only in pre-rename git history. `test_extract_parity.py` skips gracefully when
`legacy/` is absent. Schema is `arxiv_translation_hub.sql` (14 tables, 6 enums).

## Commands

```bash
pip install -r requirements.txt

# Apply v3 schema (must run before anything touches the DB; enums are created here and
# the ORM maps them with create_type=False, so the types must pre-exist)
psql -d arxiv_translation_hub -f arxiv_translation_hub.sql

# Smoke tests = the real way to exercise the pipeline end-to-end on one paper
python -m xinda.cli.extract_smoke 2503.15129          # Acquire→Convert→Extract
python -m xinda.cli.translate_smoke 2503.15129 zh     # full 13 stages

# Evaluation matrix (N papers × langs × variants) and result export
python -m xinda.cli.benchmark --papers paper_ids.txt --langs zh,fr,es --variants all
python -m xinda.cli.benchmark --export-only --out-dir results/

# One-off dataset/report builders + standalone metric runners (operate on an existing job_id)
python -m xinda.cli.build_fact_traps --per-type 60
python -m xinda.cli.meta_eval [--wmt24 wmt24_samples.csv]
python -m xinda.cli.extended_eval <job_id> --lang zh [--rcs] [--judges]  # FPS/TCR/RCS/judge backfill
python -m xinda.cli.run_fhr                                               # translate fact-traps → FHR
python -m xinda.cli.refacts <job_id> zh                                   # re-run ONLY FactExtract+FactVerify
python -m xinda.cli.review_triage <job_id>                                # quality-gated human-review triage + cost saving
python -m xinda.cli.eval_comparison_verifier --lang zh                    # comparison-verifier precision/recall on traps
python -m xinda.cli.compare_baselines <arxiv_id> zh --job <id>            # ours vs naive/abstract differentiation table
python -m xinda.cli.structure_break <arxiv_id> zh                         # raw-LaTeX struct survival: ours vs naive vs google
python -m xinda.cli.multi_llm_benchmark <arxiv_id> zh                     # multi-VENDOR QE benchmark (set vendor key envs)
python -m xinda.cli.filter_licenses candidates.txt --out paper_ids.txt    # keep only CC0/CC-BY/CC-BY-SA (translation = derivative work)

# The only real test (pure-function parity, no DB); needs a LaTeXML XML under workspace/
pytest xinda/tests/test_extract_parity.py -v
```

### Docker (the supported way to run — host LaTeXML/torch setup is painful)

```bash
docker compose run --rm app python -m xinda.cli.translate_smoke 2503.15129 zh
# Neural QE needs a GPU + torch; it lives in a SEPARATE service (Dockerfile.qe), not the lean app image:
docker compose run --rm qe  python -m xinda.cli.neural_qe <job_id>
```

`db` (Postgres, schema auto-applied) + `app` (lean: LaTeXML/texlive, no torch) + `qe` (torch
2.7/cu128 + unbabel-comet, GPU passthrough). The `app` image needs `numpy`/`scipy`/`krippendorff`
(in `requirements.docker.txt`) for `meta_eval`. Code dirs are bind-mounted, so edits take effect
without a rebuild; changing `requirements*.txt` needs `docker compose build`.

Requirements: `latexmlc`/`latexmlpost` (LaTeXML) and `magick` (ImageMagick) on PATH; a
PostgreSQL instance; `DASHSCOPE_API_KEY` for any stage that calls the model. **Neural QE**
(`comet.py`/`xcomet.py`) needs torch + HF-gated `Unbabel/XCOMET-XL` + `wmt22-cometkiwi-da`
(set `HF_TOKEN`, accept both licenses); on Blackwell GPUs (RTX 50-series, sm_120) torch must be
**≥2.7 / cu128** or you get "no kernel image is available for execution on the device".

### Configuration

`xinda/config.py` is a Pydantic `BaseSettings` reading from env / `.env`
(see `.env.example`). No hard-coded secrets. Key vars:
- `DASHSCOPE_API_KEY` — Qwen access via DashScope's OpenAI-compatible endpoint.
- `DATABASE_URL` — async DSN (`postgresql+asyncpg://…`).
- `LATEXMLC_PATH` / `LATEXMLPOST_PATH` / `MAGICK_PATH` — default to bare binary names.
- `DASHSCOPE_OPENAI_BASE_URL` — defaults to the intl/Singapore endpoint.

Model assignment is a **single-model policy**: every stage uses `_DEFAULT_MODEL`
(`config.py`) for reproducibility. To escalate refine/judge quality, change
`model_refine` / `model_judge_*` there — no call-site changes needed.
**`_DEFAULT_MODEL = "qwen3.7-max"`** (NOT `qwen3.5-plus`): on the CN endpoint
`qwen3.5-plus` **hangs indefinitely** on the detailed structured-extraction prompts
(FactExtract/Glossary) — the constrained decoder trickles bytes so the client's
read-timeout never fires; `providers/qianwen.generate()` now also wraps every call
in a hard `asyncio.wait_for(300s)`. `max_concurrency=10` (qwen3.7-max RPM=120 has headroom).

## Architecture

### Pipeline = idempotent stages over a shared context

The core is an **Orchestrator** (`pipeline/orchestrator.py`) walking a list of **Stages**.
Each stage implements `is_done(ctx, session)` (consults the **DB**, not the context) and
`run(ctx, session)`. The orchestrator skips done stages, marks
`translation_jobs.last_stage` after each success, and on crash leaves the job at the previous
`last_stage` — so **re-running with the same `job_id` resumes from where it stopped**.
`StageError` + a stage's `recoverable` flag decide whether a failure halts the run.

`PipelineContext` (`pipeline/context.py`) is the in-memory state threaded through stages
(paths, `paper_id`, `job_id`, `config`). It is a cache, not the source of truth — durable
state lives in Postgres, which is what makes resumability work.

The canonical **13-stage order** (`evaluation/benchmark.py:full_stage_list`, mirrored in the
smoke CLIs and the `pipeline_stage` enum):

```
Acquire → Convert → Extract → FactExtract → GlossaryBuild → FirstPassTranslate
  → FactVerify → CrossDocFactVerify → Refine → Coherence → ApplyXML → Render → Evaluate
```

Stages live in `pipeline/stages/`. Flow: download arXiv source → `latexmlc` TeX→XML →
extract translatable units (with placeholder tokenization) → extract verifiable claims and
build a glossary → batched first-pass translation → fact/cross-doc verification → refine
flagged units → **whole-doc discourse harmonization** → write translations back into XML →
`latexmlpost` XML→HTML + bilingual render → compute evaluation metrics.

**Coherence** (`pipeline/stages/coherence.py`, toggle `use_coherence`) is the discourse sibling
of CrossDocFactVerify: instead of *detecting* fact drift it *repairs* cross-paragraph
terminology/connective/reference drift, feeding the whole translated paper (placeholders intact)
to a 1M-context model and writing only edited units back as a higher `pass_no`. FirstPassTranslate
also carries cross-batch context — `<PREV_TRANSLATION>` (backward) **and `<NEXT_SOURCE>` (forward,
for cataphora/terms introduced downstream)** in `translation/prompts.variable_suffix`.

### Variants, config hashing, and the ablation matrix

`PipelineConfig` (`pipeline/config.py`) is the per-run knob set (language, models, feature
toggles `use_glossary/use_context/use_fact_anchor/use_fact_verify/use_cross_doc/use_coherence/
use_external_glossary/use_retry`, `use_cache`/`use_batch`, thresholds). `config_hash()` is a
12-char blake2b of the effective config and is part of the `translation_jobs` unique key
`(paper_id, language, variant, config_hash)` — so the same paper×lang re-run with different
toggles produces a **distinct job row** instead of conflicting. `variants_for(lang)` returns the
12 evaluation variants (`full` + 8 single-feature ablations + 3 bare baselines); these match the
`run_variant` enum. `benchmark.run_one` reuses an existing successful job with the same
`config_hash` rather than re-running.

### Provider abstraction

`providers/factory.create_provider` routes everything to `QianwenProvider`
(`providers/qianwen.py`) — Google was removed in v3. The provider calls Qwen through the
`openai` SDK pointed at DashScope's `compatible-mode/v1`, which gives JSON-schema structured
outputs and `cached_tokens` reporting (the design assumes DashScope **Context Cache** for the
stable prompt prefix and **Batch API** for cost — see `translation/prompts.py`). Rate limits
come from per-model RPM/TPM tables keyed by base alias (dated snapshot suffix stripped), fed
into the token-bucket `translation/rate_limit.RateLimiter`. Batching is three-level
(count → chars → tokens) in `translation/batching.py`.

### Glossary grounding (external term banks)

`GlossaryBuild` extracts terms with an LLM (high recall, but the target renderings are the
model's own ungrounded "standard translation"). When `use_external_glossary` is on,
`translation/glossary_grounding.py` overrides the LLM rendering with an authoritative term bank,
in priority order: **Jiqizhixin AI Terminology DB** (en→zh, auto-downloaded once to
`data/glossaries/jiqizhixin_all.md`), **Microsoft Terminology** (drop a `*.tbx` from
[Microsoft Learn](https://learn.microsoft.com/en-us/globalization/reference/microsoft-terminology)
into `data/glossaries/`), then **Wikidata** (live API — needs a User-Agent header or 403; only
for proper nouns/acronyms, guarded by exact English-label match). A grounded term gets
`glossary_terms.grounding_source` set + `confidence=1.0` + `locked=True`. `data/glossaries/` is
gitignored and bind-mounted into the `app` container. The glossary is still **per-paper** (no
persistent cross-paper bank yet).

### The placeholder contract (load-bearing)

`translation/placeholders.py` defines `PRESERVE_TAGS` / `SKIP_TAGS` / the `{{PT_<TAG>_<n>}}`
token format / `SPECIAL_CHARS` whitespace tokens. Inline LaTeXML elements that must not be
translated (`Math`, `cite`, `ref`, `bibref`, …) are serialized to tokens during **extract**
and reinserted during **apply**; the **prompt** tells the model to leave them untouched.
This module is the shared contract between `pipeline/stages/extract.py`,
`pipeline/stages/apply.py`, and `translation/prompts.py` — changing one set without the
others silently corrupts output.

### Evaluation suite (`xinda/evaluation/`)

Reference-free and fidelity-focused, all computed from DB rows and persisted to
`evaluation_runs`:
- **COMET-Kiwi** (`comet.py`) and **xCOMET** (`xcomet.py`, the headline QE metric, local GPU,
  run via `cli/neural_qe` in the `qe` container). Note xCOMET-XL scores en→zh scientific text on
  a much lower scale than COMET-Kiwi (observed median ≈0.47 on good translations); `xcomet_threshold`
  was recalibrated from 0.75 → **0.25** so the Refine gate isn't dominated by a miscalibrated metric.
- **PPA / MFR** (`metrics.py`) — annotation- and math-element preservation; v3 adds *ordered*
  (Kendall-tau) variants alongside the legacy set-intersection ones.
- **FPS** (`fps.py`) — Fidelity Preservation Score from `claim_verifications`, with per-claim-
  type breakdowns; **fact anchors** (`translation/fact_anchors.py`) normalize 5 claim types
  (numeric/citation/comparison/method_name/symbol) and detect drift. **Load-bearing nuance**:
  the re-extract-then-match verifier is brittle cross-lingually — a claim that was *correctly
  translated* re-extracts to a target-language surface the English-normalized matcher can't
  align, scoring a false `missing`. `fact_anchors.anchor_preserved` fixes this for
  **language-invariant anchors** (numeric/citation/symbol verbatim, method_name via glossary).
  `comparison` (semantic) is handled by the **cross-lingual comparison verifier**
  (`evaluation/comparison_verify.py`) — an LLM extracts source+target comparison tuples
  `(A, direction, B)` (entities normalized to English), then **deterministic** code aligns
  entities (anchors+glossary) and decides `preserved/reversed/dropped/weakened`. It is wired into
  `fact_verify._verify_one` (the comparison branch) and `fact_traps.judge_comparison_faithful`;
  validate its precision/recall via `cli/eval_comparison_verifier` against the synthetic
  `comparison_reversal` traps. Any new cross-lingual fact check must route comparison through it.
- **Cost** (`evaluation/cost.py`) — DashScope ¥-per-token price table → `evaluation_runs.cost_cny`
  per job; `benchmark` exports it + a per-language/variant `cost_summary.csv` (the降本 evidence).
- **Baselines** (`evaluation/baselines.py` + `cli/compare_baselines.py`) — the differentiation
  table: ours vs **naive-LLM** (same model, no placeholder contract → corrupts math/citations) vs
  Google (best-effort, free endpoint 429s) vs abstract-only (coverage), across placeholder/math
  preservation · coverage · FPS · cost. Isolated from the core pipeline.
- **RCS** (`rcs.py`) — Reader Comprehension Score via generated QA + reader/judge LLMs.
- **TCR** (`tcr.py`) — Terminology Consistency Rate across a paper's glossary terms.
- **Fact-Trap / FHR** (`fact_traps.py`, built by `cli/build_fact_traps.py`) — checks the
  system *preserves* deliberately altered facts instead of "helpfully" correcting them.
- **LLM-as-judge**: `judge_rubric.py` (RUBRIC-MQM) and `judge_geval.py` (G-Eval), with
  `meta_eval.py` proving judge trustworthiness (judge-vs-xCOMET correlation, self-consistency,
  inter-judge agreement) — run via `cli/meta_eval.py`. Needs `numpy`/`scipy`/`krippendorff`.
  Self-consistency requires RUBRIC-MQM stored **one row per run** (`run_no`), not the aggregated
  median (see `cli/backfill_judge_runs`).

### Data model (`db/models.py` ↔ `arxiv_translation_hub.sql` v3)

SQLAlchemy 2.0 async ORM over 14 tables and 6 Postgres enums. The enums (`job_status`,
`pipeline_stage`, `unit_kind`, `tu_status`, `run_variant`, `claim_type`, `drift_type`) are
created by the SQL file and mapped with `create_type=False`, so **the schema must be applied
before the ORM boots**. Engine/session factory is `db/engine.py`; each stage gets its own
`async_session()`. Notable tables: `papers` → `sections`/`translation_units` (units carry
`placeholders`/`special_chars` JSONB) → `verifiable_claims`; `translation_jobs` (one per
config_hash) → `translations` (per unit per pass, with token accounting) →
`claim_verifications`, `cross_doc_drifts`, `glossary_terms`, `evaluation_runs`,
`rendered_files`, and the eval-sampling tables (`eval_samples`/`eval_judgments`/
`eval_mqm_errors`, `comprehension_qa`/`comprehension_responses`, `fact_traps`).

## Gotchas

- The new API layer (`xinda/api/`) is a stub — there is **no v3 HTTP server
  yet**. Drive the pipeline through the CLI modules.
- All inputs/outputs are local files under `workspace/`, `static/input/`, `static/downloads/`
  (gitignored). Paths stored in the DB (`output_dir`, `storage_path`) are local FS paths;
  there is no object-storage abstraction.
- `is_done` consults the DB, so a half-written workspace won't be detected as complete — but a
  committed `last_stage` will skip a stage even if you deleted its files. Resumability is
  keyed on DB state, not on disk.
- The `README.md` was rewritten for open-sourcing (2026-06) and now describes the v3 system.
- **DashScope quota/region**: a China-mainland key is rejected on the intl endpoint
  (set `DASHSCOPE_OPENAI_BASE_URL` to `https://dashscope.aliyuncs.com/compatible-mode/v1`). The
  free tier is **per dated-snapshot** — if `qwen3.x-plus-<date>` returns 403
  `AllocationQuota.FreeTierOnly`, the **bare alias** (e.g. `qwen3.5-plus`) often still has quota.
  The OpenAI client sets `timeout`/`max_retries` (a missing timeout once deadlocked a whole
  `asyncio.gather` with the process idle at 0% CPU). **BUT** `qwen3.5-plus` itself hangs forever on
  the detailed structured-extraction prompts (read-timeout never trips a trickling constrained
  decoder) — that's why `_DEFAULT_MODEL = qwen3.7-max` and `generate()` adds a hard
  `asyncio.wait_for(300s)`. Diagnose a hang: `docker stats` NET≈0 + `/proc/1/wchan`=do_epoll_wait
  + established-but-silent :443 sockets.
- **Refine's retry gate is threshold-driven**, not buggy: it fires when `fps_unit < fps_threshold`
  OR `xcomet_score < xcomet_threshold`. If neural QE hasn't run, `xcomet_score` is NULL and only
  the FPS arm is active. Re-running on the same `job_id` resumes; refined units land as a higher
  `pass_no` that ApplyXML then picks up.

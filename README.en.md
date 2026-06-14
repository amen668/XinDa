# XinDa（信达）— Fidelity-First Scholarly Document Translation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-required-336791.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED.svg)](docker-compose.yml)

[中文](README.md) | **English**

**XinDa（信达）** takes its name from Yan Fu's translation triad *信 (fidelity) · 达
(readability) · 雅 (elegance)*: **fidelity first, full-text readability next**. The
framework targets full-text machine translation and quality evaluation of scientific
papers, with one central thesis — dismantling the three engineering bottlenecks of
full-text multilingual publishing at once:

- **Cost (P1):** a domestic, non-frontier model + context caching/batching brings
  full-text translation of one paper to roughly ¥0.03;
- **Structure (P2):** a **placeholder contract** tokenizes untranslatable inline
  elements (formulas, citations, cross-references, …) before they reach the model and
  restores them verbatim afterward — structure preservation is a **constructive
  guarantee**, not best-effort probability;
- **Quality (P3):** fidelity verification (fact anchors + a cross-lingual comparison
  verifier) plus gated, selective human review, compressing sentence-by-sentence human
  checking down to a small set of flagged units.

The same contract drives two scholarly XML dialects — **LaTeXML** (arXiv LaTeX) and
**JATS** (the journal-publishing standard) — i.e. it supports both the arXiv en→zh leg
and the Chinese-journal JATS zh→en leg.

## Architecture at a glance

A 13-stage, resumable pipeline (Postgres records state; re-running the same `job_id`
auto-resumes from the last checkpoint):

```
Acquire → Convert → Extract → FactExtract → GlossaryBuild → FirstPassTranslate
  → FactVerify → CrossDocFactVerify → Refine → Coherence → ApplyXML → Render → Evaluate
```

Evaluation suite (`xinda/evaluation/`): structural fidelity PPA/MFR (with order-sensitive
variants), factual fidelity FPS / fact-traps, the cross-lingual comparison verifier,
COMET-Kiwi / xCOMET neural QE, QA comprehension (RCS), terminology consistency (TCR),
LLM judges (RUBRIC-MQM / G-Eval) + judge meta-evaluation, cost accounting, and gated
human-review triage.

## Quick start

Docker is recommended (host-side LaTeXML/torch setup is fiddly):

```bash
cp .env.example .env        # fill in DASHSCOPE_API_KEY, etc.
docker compose up -d db     # Postgres; schema applied automatically
docker compose run --rm app python -m xinda.cli.translate_smoke 2503.15129 zh
# Neural QE needs a GPU, in the separate qe service:
docker compose run --rm qe python -m xinda.cli.neural_qe <job_id>
```

DB-free single-paper JATS translation and structural benchmark:

```bash
python -m xinda.cli.jats_translate <jats.xml> en          # translate a whole JATS paper
python -m xinda.cli.jats_benchmark --corpus-dir corpus/jats_zh2en --lang en \
    --format jats --systems contract,naive --out-dir results/demo
```

See `xinda/cli/` for more entry points (corpus building, license filtering, the
benchmark matrix, judge meta-evaluation, etc.).

## Configuration

`xinda/config.py` (Pydantic `BaseSettings`, reads `.env`; see `.env.example`). No
hard-coded secrets. Key variables: `DASHSCOPE_API_KEY`, `DATABASE_URL`,
`LATEXMLC_PATH` / `LATEXMLPOST_PATH` / `MAGICK_PATH`.

## Data & licensing

The experimental corpus includes only papers under **CC0 / CC-BY / CC-BY-SA** licenses
(translation is a derivative work); licenses are verified per paper against arXiv
OAI-PMH and PMC metadata, with the manifest in each corpus directory's `manifest.csv`.
The data directory layout is documented in [`DATA.md`](DATA.md).

The full corpus and complete results are large and reproducible, so they are not
committed; the repository ships **`examples/`** instead: bilingual rendered comparison
samples (3 papers per leg, synchronized scrolling, with formulas/figures), the aggregate
metric CSVs behind the paper's tables, and the full corpus manifests for both legs (with
per-paper license annotation). See [`examples/README.md`](examples/README.md).

## Tests

```bash
pytest xinda/tests/ -v   # pure-function parity tests, no DB required
```

## License

[MIT](LICENSE) © 2026 menjinliang. Corpus and translated outputs under `examples/` are
derivative works of CC-licensed source papers and are redistributed with attribution
under the corresponding CC license.

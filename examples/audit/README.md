# Human-audit evidence package

This directory preserves historical audit and review-gate records for the
English-to-Chinese experiment. It is intentionally a
small audit package rather than a redistribution of the complete source corpus
or all translated full texts.

## Current interpretation (2026-11-15)

The previously reported `7 / 311 = 2.3%` residual factual-error estimate is withdrawn.
The records contain disputed author judgments and are not an independent expert gold
standard. For example, HR8799 and V342 Peg identify the same object, while interpreting
“Number of departures” requires its matrix-nestedness context. The original fields
remain unchanged for traceability; they must not be treated as verified error labels.

All 320 sampled units received tool screening. The prescribed human review covered
71 flagged-stratum units and 50 randomly selected units from the 249 unflagged-stratum
units, totaling 121. One additional record had been judged before stratification.
The author used machine translation to assist a single-person review. This evidence
cannot estimate the overall residual error rate, and it does not establish professional
publication quality. See the updated [research package](../../research/revision_20261115/README.md).

## Historical sampling and labels

- Population: 1,854 units unflagged by the final gate across 10 jobs; unflagged is not verified correct.
- Sampling: stratified by job with proportional allocation, `seed=42`, `n=320`.
- The 71 abnormal-stratum records were all adjudicated. Fifty of the 249
  double-green records were randomly selected for adjudication.
- Final adjudicated records: 62 faithful, 51 initially marked factual error,
  and 9 uncertain.
- Post-adjudication attribution of the 51 apparent errors: 42 newline-token
  differences, one unrestored formula/symbol placeholder, seven residual
  factual errors, and one record judged before stratification and explicitly
  excluded from the estimate.
- The historical calculation excluded nine uncertain records from the 320-unit denominator.
  That estimate is no longer endorsed, for the reasons above.

The original decision rubric is retained in `audit_判读标准.md` as a historical document.
The CSV retains source/translation units and original decision fields so that the
record counts can be checked without silently replacing past judgments.

## Gate summaries

- `gate_by_job.txt`: final gate results for 10 jobs and 2,230 units; 376 units
  were flagged (16.9%).
- `gate_pass1_vs_latest.txt`: first-pass versus refined-gate totals.

## Corpus manifests

The exact paper-time corpus manifests are in `../corpus_manifests/`:

- 92 arXiv papers used in the reported experiment. The separate
  `2503.15129` smoke-test document is not part of the paper corpus.
- 25 PMC JATS articles used in the reported Chinese-to-English experiment.

The manifests provide source identifiers and license/provenance information.
The full papers and complete translated corpora are not redistributed here.

## SHA-256

```text
504f2cad58764e1b8476d96676620b92835d445712a61ed0630f387044fa787a  examples/audit/audit_sample.csv
96c18c06e62609bbd7ccbf3c697273a186b92ba586a0dfebedab3900768fc4b4  examples/audit/audit_判读标准.md
cb531ff6ed4c55f8e348d90c8471d8f55f341d8c3ecf192b70d36c56178aa82c  examples/audit/gate_by_job.txt
9bf8ddd617443d1a49dd46464fe18b407618cc32610e675c73920f6b4c705bd2  examples/audit/gate_pass1_vs_latest.txt
27742adb69138f2d35e960af92ee62b121f336f7ed3e379c920f215766b3dd82  examples/corpus_manifests/arxiv_en2zh_manifest.csv
c26cfcf85acb6a68e4ed2f90de83890370d9c9fa1774990592a0944cefcbc1ae  examples/corpus_manifests/jats_zh2en_manifest.csv
```


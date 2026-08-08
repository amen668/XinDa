# Human-audit evidence package

This directory contains the records behind the residual-error and review-gate
figures reported for the English-to-Chinese experiment. It is intentionally a
small audit package rather than a redistribution of the complete source corpus
or all translated full texts.

## Sampling and adjudication

- Population: 1,854 automatically released translation units from 10 jobs.
- Sampling: stratified by job with proportional allocation, `seed=42`, `n=320`.
- The 71 abnormal-stratum records were all adjudicated. Fifty of the 249
  double-green records were randomly selected for adjudication.
- Final adjudicated records: 62 faithful, 51 initially marked factual error,
  and 9 uncertain.
- Post-adjudication attribution of the 51 apparent errors: 42 newline-token
  differences, one unrestored formula/symbol placeholder, seven residual
  factual errors, and one record judged before stratification and explicitly
  excluded from the estimate.
- Reported residual factual-error estimate: `7 / 311 = 2.3%`; the nine uncertain
  records are excluded from the denominator.

The complete decision rubric and caveats are in `audit_判读标准.md`. The CSV
retains the source/translation units and all decision fields so that the
reported counts can be independently checked.

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
8496c2846686f2dfb1c8039b84fcd99e1f0fe7ca98675b0fe3557113773907aa  examples/audit/audit_sample.csv
de755550bc07a8d27f4175694735d2c7e2fa4ec0e3518d29fa935962e32a8556  examples/audit/audit_判读标准.md
cb531ff6ed4c55f8e348d90c8471d8f55f341d8c3ecf192b70d36c56178aa82c  examples/audit/gate_by_job.txt
9bf8ddd617443d1a49dd46464fe18b407618cc32610e675c73920f6b4c705bd2  examples/audit/gate_pass1_vs_latest.txt
7587c99cb9feea56c0e818e61d6a3fdaa6db5f27056427dae7e692d737751df5  examples/corpus_manifests/arxiv_en2zh_manifest.csv
ad2425895a8b9e2039fe5ee7c73da2d11b247daf6d392efab38e7c64197b75a6  examples/corpus_manifests/jats_zh2en_manifest.csv
```


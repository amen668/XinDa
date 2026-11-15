"""Offline CSV aggregation only. No model, network or database access.

Run: python 复算旧实验.py
Historical audit labels are summarized, NOT certified as ground truth.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import statistics

BASE = Path(__file__).resolve().parents[1]
P = BASE / 'data/legacy'
AUDIT = BASE.parents[1] / 'examples/audit'

def read(name):
    with (P / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

eligible = {r['论文标识'] for r in read('原实验样本与许可.csv') if r['方向'] == '英译中'}
assert len(eligible) == 92
groups = defaultdict(list)
for row in read('五模型逐篇结构.csv'):
    if row['pmcid'] in eligible:
        groups[row['model']].append(row)
models = []
for model, rows in groups.items():
    item = {'model': model, 'n': len(rows)}
    for key in ('n_ann', 'ann_kept', 'n_math', 'math_kept', 'fresh_tok', 'completion_tok'):
        item[key] = sum(int(r[key]) for r in rows)
    item['ann_pct'] = item['ann_kept'] / item['n_ann'] * 100
    item['math_pct'] = item['math_kept'] / item['n_math'] * 100
    item['wellformed_recorded'] = sum(r['wellformed'] == 'True' for r in rows)
    models.append(item)
    assert item['n'] == 92 and item['n_ann'] == 170172 and item['n_math'] == 46405

controlled = read('英译中40组核查与问答逐项.csv')
kiwi = read('英译中40组COMET逐项.csv')
assert len(controlled) == len(kiwi) == 40
assert len({r['source_sha256'] for r in controlled}) == 34
control = {
    'n_pairs': len(controlled),
    'source_groups': len({r['source_sha256'] for r in controlled}),
    'control_verdicts': dict(Counter(r['fid_good'] for r in controlled)),
    'altered_verdicts': dict(Counter(r['fid_bad'] for r in controlled)),
    'qa_control_mean': statistics.mean(float(r['rcs_good']) for r in controlled),
    'qa_altered_mean': statistics.mean(float(r['rcs_bad']) for r in controlled),
    'kiwi_control_mean': statistics.mean(float(r['kiwi_good']) for r in kiwi),
    'kiwi_altered_mean': statistics.mean(float(r['kiwi_bad']) for r in kiwi),
    'kiwi_altered_not_lower': sum(float(r['kiwi_bad']) >= float(r['kiwi_good']) for r in kiwi),
}

audit = list(csv.DictReader((AUDIT / 'audit_sample.csv').open(encoding='utf-8-sig', newline='')))
stage_key = next(k for k in audit[0] if k.startswith('两阶段复核'))
attribution_key = next(k for k in audit[0] if k.startswith('事实/结构归因'))
audit_info = {
    'sample_rows': len(audit),
    'sampling_strata': dict(Counter(r[stage_key] for r in audit)),
    'historical_attributions': dict(Counter(r[attribution_key] for r in audit)),
    'caution': 'Historical labels contain disputed/mistaken cases. No residual factual error rate is calculated.',
}
assert len(audit) == 320

# Parse only the ten article rows in the stored SQL-style aggregation.
gate_rows = []
for line in (AUDIT / 'gate_by_job.txt').read_text(encoding='utf-8').splitlines():
    cols = [c.strip() for c in line.split('|')]
    if len(cols) == 10 and cols[0].isdigit():
        gate_rows.append(cols)
gate = {
    'articles': len(gate_rows),
    'units': sum(int(r[1]) for r in gate_rows),
    'flagged': sum(int(r[2]) for r in gate_rows),
    'source_chars': sum(int(r[8]) for r in gate_rows),
    'flagged_chars': sum(int(r[9]) for r in gate_rows),
}
gate['unflagged'] = gate['units'] - gate['flagged']
gate['flagged_pct'] = 100 * gate['flagged'] / gate['units']
gate['unflagged_chars_pct'] = 100 * (gate['source_chars'] - gate['flagged_chars']) / gate['source_chars']
assert gate['articles'] == 10 and gate['units'] == 2230 and gate['flagged'] == 376
print(json.dumps({'five_models': models, 'controlled_en2zh': control,
                  'gate_aggregate': gate, 'historical_audit': audit_info},
                 ensure_ascii=False, indent=2))

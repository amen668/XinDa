"""仅用 Python 标准库，读取同一补充材料目录内的既有 ZIP，不访问网络。"""
from pathlib import Path
import json
import zipfile

HERE = Path(__file__).resolve().parents[1] / 'data/history'
def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))
files = read('文件可用性.json')['jobs']
units = {r['job_id']: r for r in read('逐任务单元核验.json')['rows']}
structures = {r['id']: r for r in read('双语文件结构核验.json')['rows']}
recorded = read('分析汇总.json')['latest_per_paper']

latest = {}
for task in sorted(files, key=lambda r: (r['created_at'], r['id'])):
    latest[task['arxiv_id']] = task['id']
selected_units = [units[k] for k in latest.values()]
selected_structures = [structures[k] for k in latest.values()]
state = json.loads((HERE / '任务状态匿名汇总.json').read_text(encoding='utf-8'))
result = {
    'snapshot_utc': state['snapshot_utc'],
    'task_status_counts': state['task_status_counts'],
    'task_total': sum(state['task_status_counts'].values()),
    'success_tasks_with_saved_files': len(files),
    'latest_success_per_arxiv_id': len(latest),
    'units': sum(r.get('units', 0) for r in selected_units),
    'fallback_units': sum(r.get('status_fallback', 0) for r in selected_units),
    'articles_with_fallback': sum(r.get('status_fallback', 0) > 0 for r in selected_units),
    'placeholder_mismatch_units': sum(r.get('units_token_mismatch', 0) for r in selected_units),
    'articles_with_placeholder_mismatch': sum(r.get('units_token_mismatch', 0) > 0 for r in selected_units),
    'math_source': sum(r.get('math_source', 0) for r in selected_structures),
    'math_matched': sum(r.get('math_matched', 0) for r in selected_structures),
    'articles_with_math': sum(r.get('math_source', 0) > 0 for r in selected_structures),
    'articles_with_all_math_matched': sum(
        r.get('math_source', 0) > 0 and
        r.get('math_source', 0) == r.get('math_matched', 0) == r.get('math_target', 0)
        for r in selected_structures),
    'supsub_source': sum(r.get('supsub_source', 0) for r in selected_structures),
    'supsub_matched': sum(r.get('supsub_matched', 0) for r in selected_structures),
}
result['math_element_percent'] = 100 * result['math_matched'] / result['math_source']
result['math_article_percent'] = 100 * result['articles_with_all_math_matched'] / result['articles_with_math']
result['supsub_percent'] = 100 * result['supsub_matched'] / result['supsub_source']
assert result['success_tasks_with_saved_files'] == state['task_status_counts']['success']
assert result['latest_success_per_arxiv_id'] == recorded['n']
for output, source in [('units', 'units'), ('fallback_units', 'status_fallback'),
                       ('placeholder_mismatch_units', 'units_token_mismatch'),
                       ('math_source', 'math_source'), ('math_matched', 'math_matched'),
                       ('supsub_source', 'supsub_source'), ('supsub_matched', 'supsub_matched')]:
    assert result[output] == recorded['totals'][source], output
assert result['articles_with_fallback'] == recorded['with_fallback']
assert result['articles_with_placeholder_mismatch'] == recorded['with_token_mismatch']
assert result['articles_with_math'] == recorded['with_math']
assert result['articles_with_all_math_matched'] == recorded['with_math'] - recorded['with_math_mismatch']
print(json.dumps(result, ensure_ascii=False, indent=2))

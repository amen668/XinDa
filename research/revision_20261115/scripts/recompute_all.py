"""Verify archived bytes and reproduce saved-data results without model calls."""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
import argparse
import json
import os
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]

def run(script):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUTF8'] = '1'
    result = subprocess.run([sys.executable, str(script)], cwd=script.parent,
                            env=env, encoding='utf-8', capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{script.name} failed:\n{result.stderr}\n{result.stdout}')
    return json.loads(result.stdout)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, default=BASE / '.recompute')
    args = parser.parse_args()
    hashes = json.loads((BASE / 'checksums.json').read_text(encoding='utf-8'))
    for name, expected in hashes['sha256'].items():
        assert sha256((BASE / name).read_bytes()).hexdigest() == expected, name
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    destination = work / 'chinese50'
    destination.mkdir(exist_ok=True)
    with ZipFile(BASE / 'data/chinese50.zip') as archive:
        for member in archive.infolist():
            path = (destination / member.filename).resolve()
            if not path.is_relative_to(destination) or '\\' in member.filename:
                raise ValueError('Unsafe archive path')
        expected = {name: json.loads(archive.read('experiment/' + name)) for name in
                    ['summary.json', 'numeric_summary.json', 'challenge_summary.json',
                     'table_summary.json', '篇级与预处理核验.json']}
        archive.extractall(destination)
    experiment = destination / 'experiment'
    for name in ['analyze.py', 'analyze_tables_zh.py', 'audit_article_structure.py']:
        run(experiment / name)
    for name, value in expected.items():
        reproduced = json.loads((experiment / name).read_text(encoding='utf-8'))
        assert reproduced == value, name
    legacy = run(BASE / 'scripts/recompute_legacy.py')
    history = run(BASE / 'scripts/recompute_history.py')
    # Manuscript-relevant invariants, independent of JSON formatting and rounding.
    zh = expected['summary.json']
    assert [zh[k]['returned'] for k in ['placeholder', 'raw_xml', 'plain']] == [3200, 3218, 3168]
    assert zh['placeholder']['formula_matched'] == zh['placeholder']['formula_source'] == 143
    assert zh['placeholder']['supsub_matched'] == 33 and zh['placeholder']['supsub_source'] == 2388
    assert history['latest_success_per_arxiv_id'] == 1560
    assert history['math_matched'] == 528315 and history['math_source'] == 529978
    assert legacy['gate_aggregate']['units'] == 2230 and legacy['gate_aggregate']['flagged'] == 376
    report = {
        'offline_only': True,
        'package_hashes_verified': len(hashes['sha256']),
        'chinese_summary_files_identical': list(expected),
        'chinese_articles': 50,
        'chinese_structure': zh,
        'legacy': legacy,
        'history': history,
        'limitations': 'Recomputes saved records and structural matches, not model reruns, expert accuracy labels, human work time, or actual billing.',
    }
    (work / 'verification_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'chinese_articles': 50,
                      'chinese_summary_files_identical': len(expected),
                      'legacy_five_models': len(legacy['five_models']),
                      'legacy_gate_units': 2230, 'legacy_gate_flagged': 376,
                      'historical_articles': 1560, 'network_or_model_calls': 0}, ensure_ascii=False))

if __name__ == '__main__':
    main()

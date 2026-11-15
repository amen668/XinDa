"""Curate a publication-safe offline package from the locally retained supplement.

This tool does not call models, download articles, or read environment files.
The input location is supplied by the operator and is never stored in the output.
"""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from hashlib import sha256
from io import StringIO
import argparse
import csv
import json
import re
from lxml import etree

BASE = Path(__file__).resolve().parents[1]
PREFIX = 'paper/全文英译重构_20261111/实验/'

def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode('utf-8')

def write(rel, data):
    target = BASE / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data if isinstance(data, bytes) else data.encode('utf-8'))

def csv_bytes(rows):
    out = StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')

def check_public(name, data):
    if name.endswith(('.png', '.pdf')):
        raise ValueError('Publication figures are outside this package: ' + name)
    text = data.decode('utf-8', errors='ignore')
    checks = {
        'absolute Windows workspace/home path': r'(?i)\b[A-Z]:[\\/](?:Users|Project|Workspace)[\\/]',
        'absolute Unix home path': r'/(?:home|root)/[A-Za-z0-9_.-]+/',
        'private credential filename': r'(?i)privateenv|\.env\.production',
        'private-key block': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
        'API-token pattern': r'\bsk-[A-Za-z0-9_-]{24,}',
        'GitHub-token pattern': r'\bgh[pousr]_[A-Za-z0-9]{30,}',
    }
    for label, pattern in checks.items():
        if re.search(pattern, text):
            raise ValueError(f'Public-content scan failed: {name}: {label}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('supplement', type=Path)
    parser.add_argument('--source-search', type=Path, action='append', default=[])
    args = parser.parse_args()
    supplement = args.supplement
    archive = supplement / '中文实验数据与代码.zip'
    origins = []
    members = {}
    with ZipFile(archive) as z:
        manifest = json.loads(z.read(PREFIX + 'manifest.json'))
        source_by_id = {}
        for row in manifest:
            original = row['path'].replace('\\', '/')
            raw = z.read(original)
            assert sha256(raw).hexdigest() == row['sha256']
            assert 'creativecommons.org/licenses/by/' in row['license']
            source_by_id[row['id']] = raw
            members['source_xml/' + row['id'] + '.xml'] = raw
        # Retain saved responses/outputs unchanged. Only relocate manifests and
        # adapt the offline analysis entry points to the public directory layout.
        scripts = {'analyze.py', 'analyze_tables_zh.py', 'audit_article_structure.py'}
        for name in z.namelist():
            if not name.startswith(PREFIX):
                continue
            short = name[len(PREFIX):]
            if short.startswith('服务器核验/') or short in {'run_fulltext.py', 'run_challenge.py', 'summarize_server.py'}:
                continue
            if name.endswith('/'):
                continue
            raw = z.read(name)
            data = raw
            if short == 'manifest.json' or short.startswith('sources/'):
                value = json.loads(raw)
                rows = value if isinstance(value, list) else [value]
                for row in rows:
                    row['path'] = 'source_xml/' + row['id'] + '.xml'
                data = dump(value)
            elif short in scripts:
                text = raw.decode('utf-8')
                text = text.replace('ROOT=P.parents[2]', 'ROOT=P.parent')
                data = text.encode('utf-8')
            public_name = 'experiment/' + short
            members[public_name] = data
            origins.append({'path': 'data/chinese50.zip!/' + public_name,
                            'original_sha256': sha256(raw).hexdigest(),
                            'public_sha256': sha256(data).hexdigest(),
                            'change': 'path relocation only' if raw != data else 'unchanged'})
        frozen = ['xinda/__init__.py', 'xinda/formats/__init__.py',
                  'xinda/formats/profiles.py', 'xinda/formats/xml_units.py',
                  'xinda/translation/placeholders.py']
        for name in frozen:
            members[name] = z.read(name)
        members['xinda/translation/__init__.py'] = b''
        members['LICENSE'] = z.read('LICENSE')

        # Keep the original index as evidence; add a complete attribution index
        # derived from available, licensed JATS metadata (name/string-name).
        original_index = z.read('来源署名与改编说明.csv')
        attributions = list(csv.DictReader(StringIO(original_index.decode('utf-8-sig'))))
        needed = {r['PMCID'] for r in attributions} - source_by_id.keys()
        for root in args.source_search:
            if not needed:
                break
            for file in root.rglob('PMC*.xml'):
                if file.stem in needed:
                    raw = file.read_bytes()
                    tree = etree.fromstring(raw)
                    license_text = ''.join(etree.tostring(e, encoding='unicode') for e in tree.xpath('//article-meta/permissions/license'))
                    if 'creativecommons.org/licenses/by/' in license_text:
                        source_by_id[file.stem] = raw
                        needed.remove(file.stem)
        updated = []
        for row in attributions:
            assert 'creativecommons.org/licenses/by/' in row['许可链接'], row['PMCID']
            raw = source_by_id.get(row['PMCID'])
            if raw:
                tree = etree.fromstring(raw)
                names = []
                for author in tree.xpath('//article-meta/contrib-group/contrib[@contrib-type="author" or (not(@contrib-type) and parent::contrib-group[@content-type="author"])]'):
                    name = author.find('.//name')
                    if name is None: name = author.find('.//string-name')
                    if name is None: name = author.find('.//collab')
                    if name is not None:
                        names.append(' '.join(' '.join(name.itertext()).split()))
                assert names, row['PMCID']
                row['作者'] = '; '.join(names)
                row['署名核对'] = 'Derived from JATS article-meta author names; article XML unchanged.'
                row['署名来源XML_SHA256'] = sha256(raw).hexdigest()
            else:
                assert re.sub(r'[;\s]', '', row['作者']), row['PMCID']
                row['署名核对'] = 'Retained from archived attribution index.'
                row['署名来源XML_SHA256'] = ''
            updated.append(row)
        write('data/chinese_attribution_original.csv', original_index)
        write('data/chinese_attribution.csv', csv_bytes(updated))
        for name in ['文件可用性.json', '逐任务单元核验.json', '双语文件结构核验.json', '分析汇总.json']:
            write('data/history/' + name, z.read(PREFIX + '服务器核验/' + name))

    for name, data in members.items():
        check_public(name, data)
    target = BASE / 'data/chinese50.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=9) as out:
        for name in sorted(members):
            info = ZipInfo(name, date_time=(2026, 11, 15, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            out.writestr(info, members[name], compresslevel=9)

    legacy = supplement / '原实验记录'
    for name in ['五模型逐篇结构.csv', '五模型92篇复算.json', '英译中40组COMET逐项.csv',
                 '英译中两输入逐篇结构.csv', '英译中原始XML逐篇结构.csv',
                 '原25篇中文两输入逐篇结构.csv', '原25篇中文原始XML逐篇结构.csv',
                 '原实验样本与许可.csv']:
        write('data/legacy/' + name, (legacy / name).read_bytes())
    original = list(csv.DictReader((legacy / '英译中40组核查与问答逐项.csv').open(encoding='utf-8-sig')))
    selected = [{k: row[k] for k in ['trap', 'rcs_good', 'rcs_bad', 'fid_good', 'fid_bad', 'focus']} for row in original]
    for row, source in zip(selected, original):
        row['source_sha256'] = sha256(source['source'].encode('utf-8')).hexdigest()
    assert len({row['source_sha256'] for row in selected}) == 34
    write('data/legacy/英译中40组核查与问答逐项.csv', csv_bytes(selected))
    write('methods/legacy_prompts.txt', (legacy / '原实验提示词模板_非逐次日志.txt').read_bytes())
    write('methods/controlled_examples_limits.txt', (supplement / '受控样例核验说明.txt').read_bytes())
    write('data/history/任务状态匿名汇总.json', (supplement / '历史批量运行记录/任务状态匿名汇总.json').read_bytes())

    # Reuse the repository audit exports instead of publishing duplicate text.
    text = (legacy / '复算旧实验.py').read_text(encoding='utf-8')
    text = text.replace("P = Path(__file__).resolve().parent", "BASE = Path(__file__).resolve().parents[1]\nP = BASE / 'data/legacy'\nAUDIT = BASE.parents[1] / 'examples/audit'")
    text = text.replace("audit = read('历史抽样320条_保留原标注.csv')", "audit = list(csv.DictReader((AUDIT / 'audit_sample.csv').open(encoding='utf-8-sig', newline='')))")
    text = text.replace("(P / '复检逐任务分流计数.txt')", "(AUDIT / 'gate_by_job.txt')")
    text = text.replace("'n_pairs': len(controlled),", "'n_pairs': len(controlled),\n    'source_groups': len({r['source_sha256'] for r in controlled}),")
    text = text.replace("assert len(controlled) == len(kiwi) == 40", "assert len(controlled) == len(kiwi) == 40\nassert len({r['source_sha256'] for r in controlled}) == 34")
    write('scripts/recompute_legacy.py', text)
    text = (supplement / '历史批量运行记录/离线复算历史记录.py').read_text(encoding='utf-8')
    a = text.index('HERE = '); b = text.index('\nlatest = {}')
    text = text[:a] + '''HERE = Path(__file__).resolve().parents[1] / 'data/history'
def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))
files = read('文件可用性.json')['jobs']
units = {r['job_id']: r for r in read('逐任务单元核验.json')['rows']}
structures = {r['id']: r for r in read('双语文件结构核验.json')['rows']}
recorded = read('分析汇总.json')['latest_per_paper']
''' + text[b:]
    write('scripts/recompute_history.py', text)

    # Preserve method implementation snapshots separately from executable entry
    # points. No online experiment launcher or credential-loading code is copied.
    for file in (supplement / '原实验方法源码').rglob('*.py'):
        rel = file.relative_to(supplement / '原实验方法源码').as_posix()
        write('methods/legacy_source/' + rel, file.read_bytes())
    write('data/archive_provenance.json', dump({
        'source_archive_sha256': sha256(archive.read_bytes()).hexdigest(),
        'archive_members': len(members),
        'fulltext_articles': 50, 'attribution_entries': len(updated),
        'unchanged_saved_outputs': 'All saved translation XML, batch responses, pairs and challenge responses remain byte-for-byte unchanged.',
        'relocations': 'manifest and source metadata paths point to source_xml; three offline scripts resolve the new archive root.',
        'omitted': ['manuscripts, review correspondence, manuscript-editing scripts and publication figures',
                    'online generation entry points and private environment paths',
                    'unused application modules, duplicate source XML and historical full-text translations'],
        'entries': origins,
    }))
    print(json.dumps({'archive_bytes': target.stat().st_size, 'archive_members': len(members),
                      'attribution_entries': len(updated), 'author_index_unresolved': len(needed)}, ensure_ascii=False))

if __name__ == '__main__':
    main()

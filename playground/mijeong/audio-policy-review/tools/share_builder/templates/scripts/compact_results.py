"""Compact sharing results; keep summaries, per-episode outcomes and replay inputs."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from update_inventory import sha

EXPANDED='V14_pilot_20261016_034601'
SMOKE='V14_smoke_20261016_032646'
INPUTS=['selection_config.json','selection_samples.npz','replay_manifest.json','audio_preflight.json','audio_preflight.csv']
for model in ['efficientat','mobilenetv4','yamnet']:
    INPUTS += [f'replay_{model}.npz',f'runtime_{model}.json']
STATIC=['framework_static_comparison.csv','framework_static_per_class.csv','framework_threshold_diagnostics.csv']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--apply',action='store_true',help='Apply the reviewed plan; default is read-only')
    args=parser.parse_args();root=args.root.resolve();results=(root/'results').resolve()
    if not results.is_dir():raise FileNotFoundError(results)
    expanded=results/'original'/EXPANDED;smoke=results/'original'/SMOKE
    for name in ['selection_config.json','replay_manifest.json',*[f'replay_{m}.npz' for m in ['efficientat','mobilenetv4','yamnet']]]:
        if not (expanded/name).is_file():raise FileNotFoundError(expanded/name)
    files=[p for p in results.rglob('*') if p.is_file()]
    before_bytes=sum(p.stat().st_size for p in files)
    plan={}
    def add(path,reason,replacement=None):
        if not path.is_file():return
        resolved=path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_relative_to(results) or resolved==results:
            raise ValueError('Unsafe cleanup target: '+str(path))
        row=dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=sha(path),reason=reason)
        if replacement is not None:
            if sha(path)!=sha(replacement):raise ValueError('Different contents: '+str(path))
            row['retained_at']=replacement.relative_to(root).as_posix()
        else:
            row['regenerate']='python scripts/replay_cached.py --stage STAGE --timing TIMING; see results/README.md'
        plan[row['path']]=row

    for path in files:
        if path.name.endswith('_trace.csv'):add(path,'regenerable_trace')
    for name in INPUTS:
        path=results/'followup'/name;canonical=expanded/name
        if path.is_file() and canonical.is_file() and sha(path)==sha(canonical):add(path,'exact_duplicate_input',canonical)
    reference=results/'checks/v14_reference'
    if reference.is_dir():
        for path in reference.iterdir():
            if not path.is_file() or path.name=='verification.json':continue
            for candidate in [expanded/path.name,smoke/path.name]:
                if candidate.is_file() and sha(path)==sha(candidate):
                    add(path,'exact_duplicate_reference',candidate);break
    for name in STATIC:
        a,b=smoke/name,expanded/name
        if a.is_file() and b.is_file() and sha(a)==sha(b):add(a,'exact_duplicate_static_comparison',b)
    manifest=root/'manifests/results_cleanup.json'
    summary=dict(before_files=len(files),before_bytes=before_bytes,removed_files=len(plan),
                  removed_bytes=sum(r['bytes'] for r in plan.values()),
                  after_files=len(files)-len(plan),after_bytes=before_bytes-sum(r['bytes'] for r in plan.values()))
    print(json.dumps(dict(apply=args.apply,**summary,plan=list(plan.values())),ensure_ascii=False,indent=2))
    if not args.apply or not plan:return
    # Verify retained copies and all resolved targets before making any deletion.
    for relative,row in plan.items():
        path=(root/relative).resolve(strict=True)
        if not path.is_relative_to(results) or not path.is_file():raise ValueError('Unsafe target')
        if sha(path)!=row['sha256']:raise ValueError('Input changed while preparing cleanup')
        if 'retained_at' in row:
            retained=root/row['retained_at']
            if row['retained_at'] in plan or sha(retained)!=row['sha256']:raise ValueError('Retained copy unavailable')
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=str(results),**summary,removed=list(plan.values()))
    (root/'manifests').mkdir(exist_ok=True)
    manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    for relative in plan:(root/relative).unlink()
    inventory=root/'manifests/source_inventory.csv'
    if inventory.is_file():
        with inventory.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)
        for row in rows:
            if row['destination'] in plan:row['status']='removed_share_'+plan[row['destination']]['reason']
        with inventory.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    build=root/'manifests/build_summary.json'
    if build.is_file():
        data=json.loads(build.read_text(encoding='utf-8'))
        data['results_compaction']=summary
        build.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Applied. Refresh hashes: python scripts/update_inventory.py')

if __name__=='__main__':main()

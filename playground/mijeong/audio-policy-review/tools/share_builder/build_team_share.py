"""Collect a new, reviewable GitHub sharing snapshot; never push or modify sources."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone

KIT = Path(__file__).resolve().parent
MEDIA = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aac'}
WEIGHTS = {'.pt', '.pth', '.keras', '.h5', '.hdf5', '.onnx', '.tflite', '.ckpt'}
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.venv', 'node_modules'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def reparse(path):
    return path.is_symlink() or bool(getattr(path.lstat(), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 1024))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path.home()/'Desktop/audio_experiments_DEF')
    parser.add_argument('--review-root', type=Path, default=Path.home()/'Desktop/audio-policy-review-20261015T164106Z-1-001/audio-policy-review')
    parser.add_argument('--codex-outputs', type=Path, default=Path.home()/'Documents/Codex/2026-10-16/r/outputs')
    parser.add_argument('--dataset-root', type=Path, default=Path('D:/Sound_Dataset_aug'))
    parser.add_argument('--conda-root', type=Path, default=Path.home()/'anaconda3')
    parser.add_argument('--include-embedding-cache', action='store_true', help='Also copy regenerable embedding caches locally; Git ignores them')
    parser.add_argument('--skip-environment-capture', action='store_true')
    parser.add_argument('--keep-full-results', action='store_true', help='Keep duplicate inputs and regenerable trace CSV files')
    args = parser.parse_args()
    dest, project, review, outputs = [p.expanduser().resolve() for p in [args.destination,args.project_root,args.review_root,args.codex_outputs]]
    for source in [project, review/'audio_policy_pilot_v1_4', outputs/'conda-setup', KIT/'templates']:
        if not source.is_dir(): raise FileNotFoundError(source)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f'Destination must be new or empty; no files were overwritten: {dest}')
    for source in [project, review, KIT]:
        if dest == source or source in dest.parents:
            raise ValueError(f'Destination cannot be inside a source directory: {source}')
    dest.mkdir(parents=True, exist_ok=True)
    inventory, missing = [], []

    def copy_file(source, relative, reason='included'):
        source = source.resolve()
        target = dest/relative
        row = dict(source=str(source), destination=Path(relative).as_posix(), bytes=source.stat().st_size, status=reason, sha256='')
        if source.suffix.lower() in MEDIA:
            row['status']='omitted_audio_media'
        elif 'embedding_cache' in Path(relative).parts and not args.include_embedding_cache:
            row['status']='omitted_regenerable_embedding_cache'
        elif source.name == '.env' or source.name.startswith('.env.') and source.name != '.env.example':
            row['status']='omitted_local_environment_file'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            row['sha256']=sha(target)
            if 'embedding_cache' in Path(relative).parts:row['status']='copied_but_gitignored_embedding_cache'
        inventory.append(row)

    def copy_tree(source, relative, excluded_top=()):
        if not source.exists():missing.append(str(source));return
        for current, dirs, files in os.walk(source, followlinks=False):
            current=Path(current)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not reparse(current/d)
                              and not (current==source and d in excluded_top))
            for name in sorted(files):
                path=current/name
                if name.endswith(('.pyc','.pyo')) or reparse(path):continue
                copy_file(path,Path(relative)/path.relative_to(source))

    print('Collecting legacy project, all trained checkpoints, and existing results...',flush=True)
    copy_tree(project,Path('project'),excluded_top={'policy_results'})
    copy_tree(project/'policy_results',Path('results/original'))
    copy_tree(review/'audio_policy_pilot_v1_4',Path('src/policy_v1_4'))
    copy_tree(review,Path('archive/initial_review'),excluded_top={'audio_policy_pilot_v1_4'})
    mappings = {
        'conda-setup':'environments',
        'selection-replay-v1_3':'archive/policy_v1_3',
        'selection-replay-S03_smoke':'results/codex/S03_smoke',
        'v1_4_reference_check':'results/checks/v14_reference',
        'v1_4_smoke_review':'reports/v14_smoke',
        'v1_4_expanded_next':'reports/v14_expanded',
        'v1_4_experiment_analysis':'reports/v14_detailed',
    }
    for name,relative in mappings.items():
        copy_tree(outputs/name,Path(relative),excluded_top={'experiment'} if name=='v1_4_expanded_next' else ())
    copy_tree(outputs/'v1_4_expanded_next/experiment',Path('results/followup'))
    for name in ['데이터셋_연결확인.md','dataset_connection_audit.json','S02_결과검토.md','v1_4_실행단계.md']:
        source=outputs/name
        if source.is_file():copy_file(source,Path('reports/history')/name)
        else:missing.append(str(source))
    analysis_work=outputs.parent/'work'
    for name in ['review_v14_results.py','advance_v14.py','finalize_v14_advance.py','deep_review_v14.py','inspect_selection_result.py','inspect_audio_roots.py']:
        source=analysis_work/name
        if source.is_file():copy_file(source,Path('archive/analysis_scripts')/name)
    copy_tree(KIT/'templates',Path('.'))
    for name in ['build_team_share.py','New-TeamShare.ps1']:
        copy_file(KIT/name,Path('tools/share_builder')/name)
    copy_tree(KIT/'templates',Path('tools/share_builder/templates'))

    observed={}
    if not args.skip_environment_capture:
        capture = "import importlib.metadata as m,json,sys,platform; p=sorted({(d.metadata.get('Name',d.name),d.version) for d in m.distributions()}); print(json.dumps(dict(python=sys.version,platform=platform.platform(),packages=[dict(name=n,version=v) for n,v in p])))"
        for env_name in ['audio-policy','audio-yamnet']:
            executable=args.conda_root/'envs'/env_name/('python.exe' if os.name=='nt' else 'bin/python')
            if not executable.is_file():observed[env_name]='not found';continue
            process=subprocess.run([str(executable),'-c',capture],capture_output=True,text=True,encoding='utf-8',timeout=90)
            if process.returncode:
                observed[env_name]='capture failed: '+process.stderr[-500:];continue
            info=json.loads(process.stdout)
            env_dir=dest/'environments/observed';env_dir.mkdir(parents=True,exist_ok=True)
            (env_dir/(env_name+'.packages.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
            (env_dir/(env_name+'.freeze.txt')).write_text('\n'.join(p['name']+'=='+p['version'] for p in info['packages'])+'\n',encoding='utf-8')
            observed[env_name]=dict(python=info['python'].split()[0],packages=len(info['packages']))

    manifest_dir=dest/'manifests';manifest_dir.mkdir(exist_ok=True)
    with (manifest_dir/'source_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['source','destination','bytes','status','sha256']);w.writeheader();w.writerows(inventory)
    weights=[]
    for row in inventory:
        if row['sha256'] and Path(row['destination']).suffix.lower() in WEIGHTS:
            weights.append(dict(path=row['destination'],bytes=row['bytes'],sha256=row['sha256'],
                                current_v14='runs/E_predefined_specialists/nonexpert_070/' in row['destination'],
                                role='vendor_pretrained_backbone' if 'third_party' in row['destination'] else 'trained_checkpoint'))
    (manifest_dir/'models.json').write_text(json.dumps(weights,ensure_ascii=False,indent=2),encoding='utf-8')
    paths=dict(original_project=str(project),original_review=str(review),original_codex_outputs=str(outputs),
               dataset_root=str(args.dataset_root),metadata_original_root='/home/plass/YAMNet/aug_data',
               repository_project='project',repository_policy='src/policy_v1_4',
               original_results='results/original',followup_results='results/followup')
    (manifest_dir/'path_map.json').write_text(json.dumps(paths,ensure_ascii=False,indent=2),encoding='utf-8')
    build=dict(created_utc=datetime.now(timezone.utc).isoformat(),destination_at_build=str(dest),
               included_files=sum(bool(r['sha256']) for r in inventory),
               omitted_embedding_files=sum(r['status']=='omitted_regenerable_embedding_cache' for r in inventory),
               omitted_embedding_bytes=sum(r['bytes'] for r in inventory if r['status']=='omitted_regenerable_embedding_cache'),
               trained_checkpoints=sum(w['role']=='trained_checkpoint' for w in weights),
               model_bytes=sum(w['bytes'] for w in weights),observed_environments=observed,
               missing_optional_sources=missing,raw_dataset_copied=False,git_push_performed=False)
    (manifest_dir/'build_summary.json').write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.keep_full_results:
        subprocess.run([sys.executable,'-B',str(dest/'scripts/compact_results.py'),'--root',str(dest),'--apply'],check=True)
    process=subprocess.run([sys.executable,'-B',str(dest/'scripts/update_inventory.py'),'--root',str(dest)],check=True)
    print(json.dumps(build,ensure_ascii=False,indent=2),flush=True)
    print('Created:',dest,flush=True)

if __name__=='__main__':main()

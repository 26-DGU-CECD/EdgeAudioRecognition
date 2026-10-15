"""Replay existing predictions in a fresh local folder; no WAV/GPU inference."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['baseline','adaptive','temporal','wait_all','bounded'],default='adaptive')
    parser.add_argument('--timing',choices=['audio_clock','independent_worker_sim'],default='independent_worker_sim')
    parser.add_argument('--source',type=Path,default=ROOT/'results/original/V14_pilot_20261016_034601')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.stage=='bounded' and args.timing!='independent_worker_sim':parser.error('Use independent_worker_sim for the bounded comparison')
    source=args.source.resolve();out=(args.out or ROOT/'.local/replays'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new or empty output folder')
    required=['selection_config.json','replay_manifest.json']+[f'replay_{m}.npz' for m in ['efficientat','mobilenetv4','yamnet']]
    for name in required:
        p=source/name
        if not p.is_file():raise FileNotFoundError(p)
        with p.open('rb') as f:
            if f.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('Git LFS object not downloaded; run git lfs pull')
    out.mkdir(parents=True,exist_ok=True)
    for name in required:shutil.copy2(source/name,out/name)
    table={
        'baseline':(['F0','F1','F2'],['R0','R1','R2'],['I0','I1','I2'],['T0'],'wait_all'),
        'adaptive':(['F1','F2'],['R0','R2'],['I1','I2','I3'],['T0'],'wait_all'),
        'temporal':(['F1','F2'],['R0','R2'],['I2','I3'],['T0','T1','T2'],'wait_all'),
        'wait_all':(['F1','F2'],['R0','R2'],['I2'],['T0'],'wait_all'),
        'bounded':(['F1','F2'],['R0','R2'],['I2'],['T0'],'bounded')}
    fusion,rules,inputs,temporal,async_policy=table[args.stage]
    tag=args.stage+('_fifo' if args.timing=='independent_worker_sim' else '_audio')
    command=[sys.executable,'-B',str(ROOT/'src/policy_v1_4/evaluate_framework_replay.py'),'--out',str(out),'--fusions',*fusion,'--rules',*rules,'--inputs',*inputs,'--temporal',*temporal,'--timing',args.timing,'--async-policy',async_policy,'--result-tag',tag]
    subprocess.run(command,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    print('New result folder:',out)

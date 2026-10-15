"""Check a clone's exact bytes, including Git LFS downloads."""
import argparse
import json
from pathlib import Path
from update_inventory import sha

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    root=args.root.resolve();manifest=json.loads((root/'manifests/files.sha256.json').read_text(encoding='utf-8'))
    bad=[];pointers=[]
    for row in manifest['files']:
        p=root/row['path']
        if not p.is_file():bad.append(dict(path=row['path'],reason='missing'));continue
        with p.open('rb') as f: prefix=f.read(80)
        if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
            pointers.append(row['path']);continue
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:bad.append(dict(path=row['path'],reason='hash mismatch'))
    print(json.dumps(dict(checked=len(manifest['files']),mismatches=bad,lfs_pointers=pointers),ensure_ascii=False,indent=2))
    if pointers:print('Run git lfs install, then git lfs pull inside your clone.')
    raise SystemExit(1 if bad or pointers else 0)

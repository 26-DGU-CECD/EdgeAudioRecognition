"""Refresh sharing hashes after intentional edits or adding new result files."""
import argparse
import hashlib
import json
import os
from pathlib import Path

LFS={'.pt','.pth','.keras','.h5','.hdf5','.onnx','.tflite','.ckpt','.npz'}
SKIP={'.git','.local','__pycache__','.pytest_cache','.mypy_cache','.venv','node_modules','embedding_cache'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def collect(root):
    result=[]
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in SKIP and not (Path(directory)/d).is_symlink())
        for name in sorted(files):
            p=Path(directory)/name;relative=p.relative_to(root).as_posix()
            if relative=='manifests/files.sha256.json' or name.endswith(('.pyc','.pyo')) or name.startswith('.env') or relative=='config/paths.local.json':continue
            if p.is_symlink() or p.suffix.lower() in {'.wav','.flac','.mp3','.m4a','.ogg','.aac'}:continue
            result.append(dict(path=relative,bytes=p.stat().st_size,sha256=sha(p),git_lfs=p.suffix.lower() in LFS))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    root=args.root.resolve();rows=collect(root)
    too_large=[r for r in rows if r['bytes']>=100*1024**2 and not r['git_lfs']]
    if too_large:raise SystemExit('Non-LFS files at/above 100 MiB: '+str([r['path'] for r in too_large]))
    value=dict(files=rows,file_count=len(rows),total_bytes=sum(r['bytes'] for r in rows),lfs_bytes=sum(r['bytes'] for r in rows if r['git_lfs']))
    (root/'manifests').mkdir(exist_ok=True)
    (root/'manifests/files.sha256.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in value.items() if k!='files'}))

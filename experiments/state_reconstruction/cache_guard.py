"""Keep cached features and predictions tied to the inputs that produced them."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import uuid

PACKAGES=('numpy','pandas','scipy','scikit-learn','lightgbm','xgboost','catboost')


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def inputs(study,data):
    source={}
    for directory in ['.','scratchboard','scratchboard/reference','deep_features','merge_check']:
        for path in sorted((study/directory).glob('*.py')):
            source[str(path.relative_to(study))]=file_hash(path)
    for name in ['requirements.txt']:
        path=study/name
        if path.exists():source[name]=file_hash(path)
    versions={}
    for name in PACKAGES:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'data':{name:file_hash(data/name) for name in ['train.jsonl','type_chart_gen2.csv']},
            'source':source,'packages':versions}


def artifacts(study):
    paths=[]
    for name in ['scratchboard','deep_features']:
        path=study/name/'cache'
        if path.exists() and any(path.iterdir()):paths.append(path)
    for name in ['scratchboard','deep_features','merge_check']:
        folder=study/name/'results'
        for path in sorted(folder.glob('*.npz')):
            paths.append(path)
            metadata=path.with_suffix('.json')
            if metadata.exists():paths.append(metadata)
        paths.extend(sorted(folder.glob('test_probabilities.csv')))
        paths.extend(sorted(folder.glob('full_*_importance.csv')))
    for name in ['scratchboard','deep_features']:
        paths.extend(sorted((study/name).glob('submission*.csv')))
    return list(dict.fromkeys(paths))


def has_prediction_cache(study):
    return any((study/name/'cache/full_fit.pkl.gz').exists() for name in ['scratchboard','deep_features']) or any(
        (study/name/'results/test_probabilities.csv').exists() for name in ['scratchboard','deep_features'])


def validate_cache(study,data,action,refresh=False):
    study,data=Path(study),Path(data)
    manifest_path=study/'cache_manifest.json'
    current={'format':1,'inputs':inputs(study,data),'test_sha256':file_hash(data/'test.jsonl') if action=='predict' else None}
    found=artifacts(study);previous=None;reason=None
    if manifest_path.exists():
        try:
            previous=json.loads(manifest_path.read_text())
            if not isinstance(previous,dict):raise ValueError('Expected an object')
        except (ValueError,OSError):reason='The cache manifest cannot be read.'
    elif found:reason='Existing cached files have no input manifest.'
    if isinstance(previous,dict):
        if previous.get('format')!=1 or previous.get('inputs')!=current['inputs']:
            reason='The data, source or installed packages changed.'
        elif action=='predict' and previous.get('test_sha256')!=current['test_sha256']:
            if has_prediction_cache(study):
                reason='The test input does not match the prediction cache.'
        if action!='predict':current['test_sha256']=previous.get('test_sha256')
    if reason and found and not refresh:
        raise RuntimeError(reason+' Run again with --refresh-cache to archive generated files and rebuild them. The current files were left in place.')
    archived=None
    if refresh and found:
        tag=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
        archived=study/'cache_archive'/tag
        archived.mkdir(parents=True)
        if manifest_path.exists():shutil.copy2(manifest_path,archived/'cache_manifest.json')
        for path in found:
            destination=archived/path.relative_to(study)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(path),str(destination))
            if path.name=='cache':path.mkdir()
        # Training-only rebuilds must not bless an earlier test input.
        if action!='predict':current['test_sha256']=None
    elif isinstance(previous,dict) and previous.get('inputs')!=current['inputs'] and action!='predict':
        current['test_sha256']=None
    temporary=manifest_path.with_name(manifest_path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(current,indent=2,sort_keys=True)+'\n')
    temporary.replace(manifest_path)
    return archived

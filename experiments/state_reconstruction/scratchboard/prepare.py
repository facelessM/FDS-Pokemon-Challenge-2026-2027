"""Build fold-local experiment inputs; test labels/data are never used here."""
import argparse
import gzip
import json
import pickle
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from data_views import load, describe, observation_features, identity_features, sparse_tokens
from team_context import TeamContext
from tactics import TacticalContext
from reference.data import build_species_table, build_move_vocab, build_top_species, load_type_chart
from reference.features import build_features

ROOT=Path(__file__).resolve().parent
BASE_GROUPS=['starter','state_t30','sacrifice','move_usage','move_effects','alive_species']

def save_cache(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with gzip.open(temp,'wb',compresslevel=2) as f:pickle.dump(value,f,pickle.HIGHEST_PROTOCOL)
    temp.replace(path)

def subset(values,indices):return [values[int(i)] for i in indices]

def numeric(rows):
    frame=pd.DataFrame(rows).fillna(0).astype(np.float32)
    assert np.isfinite(frame.to_numpy()).all()
    return frame

def encodings(train_views, y, groups, target_views):
    """Training rows exclude their own outcome and duplicate-timeline outcomes."""
    n=len(y);train={};valid={}
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=99).split(np.zeros(n),y,groups))
    for key,config,mode in [('record',{},'base'),('hier',dict(enriched=True,moves=True,graph=True),'hierarchy')]:
        parts=[]
        for a,b in splits:
            model=TeamContext(**config).fit(subset(train_views,a),y[a])
            part=model.transform(subset(train_views,b),mode);part.index=b;parts.append(part)
        train[key]=pd.concat(parts).sort_index().astype(np.float32)
        model=TeamContext(**config).fit(train_views,y)
        valid[key]=(model.transform(target_views,mode) if target_views else pd.DataFrame(columns=train[key].columns)).astype(np.float32)
    post_columns=[c for c in train['hier'] if not c.startswith(('p1_item_','p1_lead_','p1_skill','p2_expected_skill','diff_skill','skill_probability'))]
    train['post']=train['hier'][post_columns];valid['post']=valid['hier'][post_columns]
    return train,valid

def fold_features(battles,views,y,groups,tr,va,chart,static):
    bt=subset(battles,tr);vt=subset(views,tr);vv=subset(views,va)
    context={'species_table':build_species_table(bt),'move_vocab':build_move_vocab(bt),'top_species':build_top_species(bt)}
    # No outcomes occur in these numeric features or in TacticalContext's outputs.
    base=build_features(battles,BASE_GROUPS,context).drop(columns='battle_id').astype(np.float32)
    left={k:v.iloc[tr].reset_index(drop=True) for k,v in static.items()}
    right={k:v.iloc[va].reset_index(drop=True) for k,v in static.items()}
    left['base']=base.iloc[tr].reset_index(drop=True);right['base']=base.iloc[va].reset_index(drop=True)
    e1,e2=encodings(vt,y[tr],groups[tr],vv);left.update(e1);right.update(e2)
    tactical=TacticalContext().fit(bt,vt,chart,context['top_species'])
    for name,func in [('roster',lambda b,v:tactical.roster_features(v)),('threat',tactical.threat_features)]:
        feat=numeric([func(b,v) for b,v in zip(battles,views)])
        left[name]=feat.iloc[tr].reset_index(drop=True);right[name]=feat.iloc[va].reset_index(drop=True)
    return dict(train=left,valid=right,tr=tr,va=va,ytr=y[tr],yva=y[va])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='../upload');ap.add_argument('--seed',type=int,default=20261026);ap.add_argument('--fold',type=int)
    args=ap.parse_args();start=time.time();battles=load(Path(args.data)/'train.jsonl');views=[describe(b) for b in battles]
    y=np.array([int(b['player_won']) for b in battles]);groups=np.array([v['group'] for v in views])
    static={'obs':numeric([observation_features(b,v) for b,v in zip(battles,views)]),'cats':pd.DataFrame([identity_features(b,v) for b,v in zip(battles,views)])}
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    # Per-row tokens are raw observations, not a fitted vocabulary.
    if not (cache/'raw_tokens.pkl.gz').exists():
        save_cache(cache/'raw_tokens.pkl.gz',[sparse_tokens(v) for v in views])
    if not (cache/'train_meta.npz').exists():
        np.savez(cache/'train_meta.npz',y=y,groups=groups,ids=[b['battle_id'] for b in battles])
    chart=load_type_chart(args.data)
    splits=StratifiedGroupKFold(5,shuffle=True,random_state=args.seed).split(np.zeros(len(y)),y,groups)
    for fold,(tr,va) in enumerate(splits):
        if args.fold is not None and args.fold!=fold:continue
        path=cache/f'{args.seed}_{fold}.pkl.gz'
        if path.exists():print(f'exists {path.name}',flush=True);continue
        assert not set(groups[tr])&set(groups[va])
        features=fold_features(battles,views,y,groups,tr,va,chart,static)
        save_cache(path,features)
        print(json.dumps(dict(seed=args.seed,fold=fold,ntrain=len(tr),nvalid=len(va),features={k:v.shape[1] for k,v in features['train'].items()},elapsed=round(time.time()-start,1))),flush=True)

if __name__=='__main__':main()

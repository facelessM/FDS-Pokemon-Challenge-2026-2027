"""Merge the remaining legacy representation into the reconstructed-state model."""
import argparse,gzip,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scratchboard'))
from reference.models import lgbm_final,xgb_final

MODELS=['deep_hierarchy_lgb','deep_hierarchy_xgb','deep_hierarchy_cat']
NEW=['exits','resources','switches','entryko','sets']

def read(path):
    with gzip.open(path,'rb') as f:return pickle.load(f)

def matrices(base,extra,categorical=False):
    old=['base','hier','obs','threat']+(['cats'] if categorical else [])
    frames=[]
    for split,index in [('train','tr'),('valid','va')]:
        X=pd.concat([base[split][name] for name in old]+[extra[name].iloc[base[index]].reset_index(drop=True) for name in NEW],axis=1)
        assert X.columns.is_unique and not {'battle_id','player_won'}&set(X.columns)
        frames.append(X)
    return frames

def make_model(name,cat_columns=None):
    if name.endswith('_lgb'):return lgbm_final()
    if name.endswith('_xgb'):return xgb_final()
    return CatBoostClassifier(iterations=700,depth=5,learning_rate=.04,l2_leaf_reg=10,random_seed=0,thread_count=2,verbose=False,allow_writing_files=False,cat_features=cat_columns)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--models',nargs='+',choices=MODELS,default=MODELS);args=ap.parse_args()
    out=ROOT/'merge_check/results';out.mkdir(exist_ok=True)
    y=np.load(ROOT/'scratchboard/cache/train_meta.npz')['y'];p={n:np.zeros(len(y)) for n in args.models if not (out/f'{args.seed}_{n}.npz').exists()};scores={n:[] for n in p};foldids=np.full(len(y),-1,int)
    for k in range(5):
        if not p:break
        base=read(ROOT/f'scratchboard/cache/{args.seed}_{k}.pkl.gz');extra=read(ROOT/f'deep_features/cache/{args.seed}_{k}.pkl.gz')
        extra['entryko']=read(ROOT/f'deep_features/cache/aux_entry_{args.seed}_{k}.pkl.gz');extra['sets']=read(ROOT/f'deep_features/cache/sets_{args.seed}_{k}.pkl.gz')
        foldids[base['va']]=k
        for name in p:
            X,Z=matrices(base,extra,categorical=name.endswith('_cat'));cats=list(base['train']['cats'].columns) if name.endswith('_cat') else None
            est=make_model(name,cats).fit(X,base['ytr']);p[name][base['va']]=est.predict_proba(Z)[:,1]
            scores[name].append(float(accuracy_score(base['yva'],p[name][base['va']]>=.5)))
            print(json.dumps(dict(seed=args.seed,fold=k,model=name,features=X.shape[1],accuracy=scores[name][-1])),flush=True)
    for name,q in p.items():
        assert (foldids>=0).all()
        report=dict(seed=args.seed,model=name,accuracy=float(accuracy_score(y,q>=.5)),log_loss=float(log_loss(y,q)),folds=scores[name])
        np.savez_compressed(out/f'{args.seed}_{name}.npz',p=q,fold=foldids)
        (out/f'{args.seed}_{name}.json').write_text(json.dumps(report,indent=2));print('FINAL '+json.dumps(report),flush=True)

if __name__=='__main__':main()

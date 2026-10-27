"""Fixed classical-model experiments with grouped out-of-fold predictions."""
import argparse
import gzip
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score
from reference.models import lgbm_final,xgb_final

ROOT=Path(__file__).resolve().parent
CONFIGS={
 'base_lgb':('lgb',['base','record']),
 'base_xgb':('xgb',['base','record']),
 'obs_lgb':('lgb',['base','record','obs']),
 'roster_lgb':('lgb',['base','record','roster']),
 'posterior_lgb':('lgb',['base','post']),
 'threat_lgb':('lgb',['base','record','threat']),
 'hierarchy_lgb':('lgb',['base','hier']),
 'combined_lgb':('lgb',['base','post','obs','threat']),
 'combined_xgb':('xgb',['base','post','obs','threat']),
 'no_obs_lgb':('lgb',['base','post','threat']),
 'no_obs_xgb':('xgb',['base','post','threat']),
 'no_post_lgb':('lgb',['base','record','obs','threat']),
 'no_post_xgb':('xgb',['base','record','obs','threat']),
 'no_threat_lgb':('lgb',['base','post','obs']),
 'no_threat_xgb':('xgb',['base','post','obs']),
 'cat_base':('cat',['base','record','cats']),
 'cat_combined':('cat',['base','hier','obs','threat','cats']),
 'sparse_logreg':('sparse',['base','record']),
}

def matrix(parts,names):
    X=pd.concat([parts[n] for n in names],axis=1)
    assert X.columns.is_unique
    return X

def fit_predict(name,fold,raw_tokens=None):
    kind,names=CONFIGS[name];X=matrix(fold['train'],names);V=matrix(fold['valid'],names);y=fold['ytr']
    if kind=='lgb':model=lgbm_final()
    elif kind=='xgb':model=xgb_final()
    elif kind=='cat':
        from catboost import CatBoostClassifier
        cats=list(fold['train']['cats'].columns)
        model=CatBoostClassifier(iterations=700,depth=5,learning_rate=.04,l2_leaf_reg=10,random_seed=0,thread_count=2,verbose=False,allow_writing_files=False,cat_features=cats)
    elif kind=='sparse':
        vec=DictVectorizer();T=vec.fit_transform([raw_tokens[i] for i in fold['tr']]);U=vec.transform([raw_tokens[i] for i in fold['va']])
        scaler=StandardScaler();X=hstack([csr_matrix(scaler.fit_transform(X)),T],format='csr');V=hstack([csr_matrix(scaler.transform(V)),U],format='csr')
        model=LogisticRegression(C=.05,solver='liblinear',max_iter=1000,random_state=0)
    model.fit(X,y)
    return model.predict_proba(V)[:,1]

def metrics(y,p):return dict(accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p)),auc=float(roc_auc_score(y,p)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20261026);ap.add_argument('--models',nargs='+',default=list(CONFIGS));args=ap.parse_args()
    meta=np.load(ROOT/'cache/train_meta.npz');y=meta['y'];result_dir=ROOT/'results';result_dir.mkdir(exist_ok=True)
    raw=None
    if 'sparse_logreg' in args.models:
        with gzip.open(ROOT/'cache/raw_tokens.pkl.gz','rb') as f:raw=pickle.load(f)
    outputs={};started=time.time()
    for name in args.models:
        path=result_dir/f'{args.seed}_{name}.npz'
        if path.exists():print('exists '+path.name,flush=True);continue
        outputs[name]={'p':np.zeros(len(y)),'fold':np.full(len(y),-1,dtype=int),'scores':[]}
    for k in range(5):
        # Preparation may run concurrently; wait for a completely written cache.
        deadline=time.time()+1800
        while True:
            try:
                with gzip.open(ROOT/f'cache/{args.seed}_{k}.pkl.gz','rb') as f:fold=pickle.load(f)
                break
            except (FileNotFoundError,EOFError):
                if time.time()>deadline:raise
                time.sleep(2)
        for name,out in outputs.items():
            checkpoint=result_dir/f'{args.seed}_{name}_fold{k}.npy'
            if checkpoint.exists():p=np.load(checkpoint)
            else:
                p=fit_predict(name,fold,raw);np.save(checkpoint,p)
            out['p'][fold['va']]=p;out['fold'][fold['va']]=k
            score=metrics(fold['yva'],p);out['scores'].append(score)
            print(json.dumps(dict(model=name,seed=args.seed,fold=k,**score,elapsed=round(time.time()-started,1))),flush=True)
    for name,out in outputs.items():
        assert (out['fold']>=0).all()
        np.savez_compressed(result_dir/f'{args.seed}_{name}.npz',p=out['p'],fold=out['fold'])
        report=dict(model=name,seed=args.seed,**metrics(y,out['p']),folds=out['scores'])
        (result_dir/f'{args.seed}_{name}.json').write_text(json.dumps(report,indent=2))
        print('FINAL '+json.dumps(report),flush=True)

if __name__=='__main__':main()

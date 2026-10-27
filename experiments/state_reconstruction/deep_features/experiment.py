import argparse,gzip,json,pickle,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scratchboard'))
from reference.models import lgbm_final,xgb_final

CONFIGS={
 'exits_lgb':('lgb',['base','record'],['exits']),
 'resources_lgb':('lgb',['base','record'],['resources']),
 'switches_lgb':('lgb',['base','record'],['switches']),
 'quality_lgb':('lgb',['base','record'],['quality']),
 'order_lgb':('lgb',['base','record'],['order']),
 'state_lgb':('lgb',['base','record'],['exits','resources','switches']),
 'prior_state_lgb':('lgb',['base','post','obs','threat'],['exits','resources','switches']),
 'state_xgb':('xgb',['base','record'],['exits','resources','switches']),
 'prior_state_xgb':('xgb',['base','post','obs','threat'],['exits','resources','switches']),
 'prior_resources_lgb':('lgb',['base','post','obs','threat'],['resources']),
 'prior_switches_lgb':('lgb',['base','post','obs','threat'],['switches']),
 'prior_exits_lgb':('lgb',['base','post','obs','threat'],['exits']),
 'ko_lgb':('lgb',['base','record'],['ko']),
 'ko_xgb':('xgb',['base','record'],['ko']),
 'prior_ko_lgb':('lgb',['base','post','obs','threat'],['ko']),
 'prior_ko_xgb':('xgb',['base','post','obs','threat'],['ko']),
 'entryko_lgb':('lgb',['base','record'],['entryko']),
 'entryko_xgb':('xgb',['base','record'],['entryko']),
 'prior_entryko_lgb':('lgb',['base','post','obs','threat'],['entryko']),
 'prior_entryko_xgb':('xgb',['base','post','obs','threat'],['entryko']),
 'deep_ko_lgb':('lgb',['base','post','obs','threat'],['exits','resources','switches','ko']),
 'deep_ko_xgb':('xgb',['base','post','obs','threat'],['exits','resources','switches','ko']),
 'sets_lgb':('lgb',['base','record'],['sets']),
 'sets_xgb':('xgb',['base','record'],['sets']),
 'prior_sets_lgb':('lgb',['base','post','obs','threat'],['sets']),
 'prior_sets_xgb':('xgb',['base','post','obs','threat'],['sets']),
 'deep_entry_lgb':('lgb',['base','post','obs','threat'],['exits','resources','switches','entryko']),
 'deep_entry_xgb':('xgb',['base','post','obs','threat'],['exits','resources','switches','entryko']),
 'deep_entry_sets_lgb':('lgb',['base','post','obs','threat'],['exits','resources','switches','entryko','sets']),
 'deep_entry_sets_xgb':('xgb',['base','post','obs','threat'],['exits','resources','switches','entryko','sets']),
}

def load_wait(path):
    limit=time.time()+1200
    while True:
        try:
            with gzip.open(path,'rb') as f:return pickle.load(f)
        except FileNotFoundError:
            if time.time()>limit:raise
            time.sleep(2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20261026);ap.add_argument('--models',nargs='+',default=list(CONFIGS));args=ap.parse_args()
    out=ROOT/'deep_features/results';out.mkdir(exist_ok=True);y=np.load(ROOT/'scratchboard/cache/train_meta.npz')['y'];pred={n:np.zeros(len(y)) for n in args.models if not (out/f'{args.seed}_{n}.npz').exists()};scores={n:[] for n in pred}
    for k in range(5):
        base=load_wait(ROOT/f'scratchboard/cache/{args.seed}_{k}.pkl.gz');extra=load_wait(ROOT/f'deep_features/cache/{args.seed}_{k}.pkl.gz')
        if any('ko' in CONFIGS[n][2] for n in pred):extra['ko']=load_wait(ROOT/f'deep_features/cache/aux_{args.seed}_{k}.pkl.gz')
        if any('entryko' in CONFIGS[n][2] for n in pred):extra['entryko']=load_wait(ROOT/f'deep_features/cache/aux_entry_{args.seed}_{k}.pkl.gz')
        if any('sets' in CONFIGS[n][2] for n in pred):extra['sets']=load_wait(ROOT/f'deep_features/cache/sets_{args.seed}_{k}.pkl.gz')
        for name,p in pred.items():
            kind,old,new=CONFIGS[name]
            X=pd.concat([base['train'][n] for n in old]+[extra[n].iloc[base['tr']].reset_index(drop=True) for n in new],axis=1)
            Z=pd.concat([base['valid'][n] for n in old]+[extra[n].iloc[base['va']].reset_index(drop=True) for n in new],axis=1)
            assert X.columns.is_unique
            model=lgbm_final() if kind=='lgb' else xgb_final();model.fit(X,base['ytr']);p[base['va']]=model.predict_proba(Z)[:,1]
            scores[name].append(float(accuracy_score(base['yva'],p[base['va']]>=.5)))
            print(json.dumps({'model':name,'seed':args.seed,'fold':k,'accuracy':scores[name][-1]}),flush=True)
    for name,p in pred.items():
        result={'model':name,'seed':args.seed,'accuracy':float(accuracy_score(y,p>=.5)),'log_loss':float(log_loss(y,p)),'folds':scores[name]}
        np.savez_compressed(out/f'{args.seed}_{name}.npz',p=p);(out/f'{args.seed}_{name}.json').write_text(json.dumps(result,indent=2));print('FINAL '+json.dumps(result),flush=True)

if __name__=='__main__':main()

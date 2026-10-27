import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parents[1]

def predictions(seed):
    pred={}
    for p in (ROOT/'deep_features/results').glob(f'{seed}_*.npz'):
        d=np.load(p)
        if 'p' in d:pred[p.stem[len(str(seed))+1:]]=d['p']
    for name in list(pred):
        if name.endswith('_lgb') and name[:-4]+'_xgb' in pred:pred[name[:-4]+'_blend']=(pred[name]+pred[name[:-4]+'_xgb'])/2
    previous={n:np.load(ROOT/f'scratchboard/results/{seed}_{n}.npz')['p'] for n in ['base_lgb','base_xgb','combined_lgb','combined_xgb']}
    pred['strict_base']=(previous['base_lgb']+previous['base_xgb'])/2
    pred['previous_combined']=(previous['combined_lgb']+previous['combined_xgb'])/2
    pred['previous_submission01']=sum(previous.values())/4
    pred['fixed_main']=np.load(ROOT/f'fix_impact_check/fixed_{seed}.npz')['p']
    for name in ['prior_state_blend','deep_ko_blend','deep_entry_blend','prior_entryko_blend','deep_entry_sets_blend']:
        if name in pred:pred[name+'_half_previous']=.5*pred[name]+.5*pred['previous_submission01']
    return pred

def main():
    seeds=list(map(int,sys.argv[1:])) or [20261026];y=np.load(ROOT/'scratchboard/cache/train_meta.npz')['y'];rows=[]
    for seed in seeds:
        pred=predictions(seed)
        for name,p in pred.items():
            d=((p>=.5)==y).astype(int)-((pred['fixed_main']>=.5)==y).astype(int)
            rows.append({'seed':seed,'model':name,'accuracy':accuracy_score(y,p>=.5),'log_loss':log_loss(y,p),'delta_fixed':d.mean(),'corrected':sum(d==1),'regressed':sum(d==-1)})
    f=pd.DataFrame(rows);f.to_csv(ROOT/'deep_features/results/summary.csv',index=False);print(f.sort_values(['seed','accuracy'],ascending=False).to_string(index=False))

if __name__=='__main__':main()

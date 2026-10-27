"""Assess predeclared merges using existing paired out-of-fold predictions."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,log_loss

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[20261026,20261105,193]

def load(seed):
    old={n:np.load(ROOT/f'scratchboard/results/{seed}_{n}.npz')['p'] for n in ['base_lgb','base_xgb','combined_lgb','combined_xgb','cat_combined']}
    state=sum(np.load(ROOT/f'deep_features/results/{seed}_deep_entry_sets_{m}.npz')['p'] for m in ['lgb','xgb'])/2
    ref={
        'previous_01':sum(old[n] for n in ['base_lgb','base_xgb','combined_lgb','combined_xgb'])/4,
        'previous_02':(old['combined_lgb']+old['combined_xgb'])/2,
        'previous_03':.25*old['base_lgb']+.25*old['base_xgb']+.5*old['cat_combined'],
    }
    p={'state':state,**ref}
    for name,q in ref.items():
        p['state_75_'+name+'_25']=.75*state+.25*q
        p['state_50_'+name+'_50']=.5*state+.5*q
    for m in ['lgb','xgb','cat']:
        path=ROOT/f'merge_check/results/{seed}_deep_hierarchy_{m}.npz'
        if path.exists():p['hierarchy_'+m]=np.load(path)['p']
    if {'hierarchy_lgb','hierarchy_xgb'}<=p.keys():
        p['hierarchy_blend']=(p['hierarchy_lgb']+p['hierarchy_xgb'])/2
        p['hierarchy_half_state']=.5*p['hierarchy_blend']+.5*state
    if 'hierarchy_cat' in p:
        p['state_cat25']=.75*state+.25*p['hierarchy_cat']
        p['state_cat_equal']=(2*state+p['hierarchy_cat'])/3
        if 'hierarchy_blend' in p:p['hierarchy_cat_equal']=(2*p['hierarchy_blend']+p['hierarchy_cat'])/3
    return p

def bootstrap(d,groups,repeats=10000):
    _,inv=np.unique(groups,return_inverse=True);sizes=np.bincount(inv);delta=np.bincount(inv,weights=d.mean(axis=0));rng=np.random.default_rng(20261027);boot=[]
    for start in range(0,repeats,200):
        ix=rng.integers(0,len(sizes),size=(min(200,repeats-start),len(sizes)))
        boot.extend(delta[ix].sum(axis=1)/sizes[ix].sum(axis=1))
    return np.quantile(boot,[.025,.975]).tolist()

def main():
    meta=np.load(ROOT/'scratchboard/cache/train_meta.npz');y=meta['y'];groups=meta['groups'];allp={s:load(s) for s in SEEDS}
    names=sorted(set.intersection(*(set(p) for p in allp.values())));rows=[];detail={}
    for name in names:
        acc=[float(accuracy_score(y,allp[s][name]>=.5)) for s in SEEDS];ll=[float(log_loss(y,allp[s][name])) for s in SEEDS]
        delta=[acc[i]-accuracy_score(y,allp[s]['state']>=.5) for i,s in enumerate(SEEDS)]
        row=dict(model=name,seed_20261026=acc[0],seed_20261105=acc[1],seed_193=acc[2],mean_accuracy=float(np.mean(acc)),mean_log_loss=float(np.mean(ll)),delta_state=float(np.mean(delta)),positive_seeds=sum(z>0 for z in delta),worst_seed_delta=float(min(delta)))
        rows.append(row)
        detail[name]=dict(accuracies=acc,log_losses=ll,delta_by_seed=delta)
    frame=pd.DataFrame(rows).sort_values(['mean_accuracy','mean_log_loss'],ascending=[False,True]);frame.to_csv(ROOT/'merge_check/results/summary.csv',index=False)
    print(frame.to_string(index=False))
    # Exact decision guard: a sub-0.10 percentage-point mean gain is too small
    # to justify adding components after this many CV comparisons.
    base=frame[frame.model=='state'].iloc[0]
    eligible=frame[(~frame.model.str.startswith('previous'))&(~frame.model.isin(['hierarchy_lgb','hierarchy_xgb','hierarchy_cat']))&(frame.delta_state>=.001-1e-10)&(frame.positive_seeds>=2)&(frame.worst_seed_delta>=-.001-1e-10)&(frame.mean_log_loss<base.mean_log_loss)]
    complete=all('hierarchy_cat_equal' in p for p in allp.values())
    choice=str(eligible.iloc[0].model) if len(eligible) else 'state'
    if complete:
        for name in ['state_50_previous_01_50','state_50_previous_02_50','state_50_previous_03_50','hierarchy_blend','hierarchy_half_state','state_cat25','state_cat_equal','hierarchy_cat_equal',choice]:
            if 'ci95' in detail[name]:continue
            d=np.array([((allp[s][name]>=.5)==y).astype(float)-((allp[s]['state']>=.5)==y).astype(float) for s in SEEDS])
            detail[name]['ci95']=bootstrap(d,groups);detail[name]['net_correct_by_seed']=d.sum(axis=1).astype(int).tolist()
        report=dict(complete=True,selected=choice,decision='Apply the predeclared accuracy/log-loss/stability guard, with at least +0.10 percentage point mean gain to justify more components.',comparisons=detail,bootstrap_note='Paired exact-timeline battle-cluster bootstrap, all three seed results kept together. Exploratory and unadjusted for model selection.')
        (ROOT/'merge_check/results/review.json').write_text(json.dumps(report,indent=2))
        print('SELECTED '+choice,flush=True)

if __name__=='__main__':main()

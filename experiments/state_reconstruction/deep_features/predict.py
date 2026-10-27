"""Fit the frozen feature candidate and generate one Kaggle submission."""
import argparse, gzip, hashlib, json, pickle, sys
from collections import Counter
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import features as deep
from auxiliary import make_events,model as auxiliary_model,aggregate
from move_records import MoveRecords

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scratchboard'))
from data_views import load,describe,observation_features,identity_features
from prepare import fold_features,numeric,save_cache
from reference.data import load_type_chart
from reference.models import lgbm_final,xgb_final

NEW_GROUPS=['exits','resources','switches','entryko','sets']
CONFIGS={
    'deep_entry_sets_lgb':('lgb',['base','post','obs','threat'],NEW_GROUPS),
    'deep_entry_sets_xgb':('xgb',['base','post','obs','threat'],NEW_GROUPS),
    'deep_hierarchy_lgb':('lgb',['base','hier','obs','threat'],NEW_GROUPS),
    'deep_hierarchy_xgb':('xgb',['base','hier','obs','threat'],NEW_GROUPS),
    'deep_hierarchy_cat':('cat',['base','hier','obs','threat','cats'],NEW_GROUPS),
    'base_lgb':('lgb',['base','record'],[]),
    'base_xgb':('xgb',['base','record'],[]),
    'combined_lgb':('lgb',['base','post','obs','threat'],[]),
    'combined_xgb':('xgb',['base','post','obs','threat'],[]),
    'cat_combined':('cat',['base','hier','obs','threat','cats'],[]),
}

def auxiliary_features(B,V,y,groups,tr,va,stats,chart):
    X,M=make_events(B,V,stats,chart,include_entry=True)
    source=M.battle.isin(tr)&(M.side=='p1')
    for c in ['victim','attacker','attack_move','previous_own_move']:
        X[c]=pd.Categorical(X[c],categories=sorted(X.loc[source,c].unique()))
    probabilities=np.zeros(len(M))
    for a,b in StratifiedGroupKFold(2,shuffle=True,random_state=347).split(np.zeros(len(tr)),y[tr],groups[tr]):
        assert not set(groups[tr[a]])&set(groups[tr[b]])
        fit=M.battle.isin(tr[a])&(M.side=='p1');check=M.battle.isin(tr[b])
        est=auxiliary_model().fit(X.loc[fit],M.loc[fit,'target'])
        probabilities[check]=est.predict_proba(X.loc[check])[:,1]
    est=auxiliary_model().fit(X.loc[source],M.loc[source,'target']);check=M.battle.isin(va)
    probabilities[check]=est.predict_proba(X.loc[check])[:,1]
    top=[n for n,c in Counter(n for i in tr for n in V[i]['team']).most_common(12)]
    return aggregate(V,M,probabilities,top)

def set_features(V,y,groups,tr,va):
    parts=[]
    for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=99).split(np.zeros(len(tr)),y[tr],groups[tr]):
        assert not set(groups[tr[a]])&set(groups[tr[b]])
        est=MoveRecords().fit([V[i] for i in tr[a]],y[tr[a]])
        z=est.transform([V[i] for i in tr[b]]);z.index=tr[b];parts.append(z)
    est=MoveRecords().fit([V[i] for i in tr],y[tr]);z=est.transform([V[i] for i in va]);z.index=va;parts.append(z)
    return pd.concat(parts).sort_index()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rebuild-base',action='store_true');ap.add_argument('--selection',type=Path,default=ROOT/'deep_features/selection.json');args=ap.parse_args()
    selection=json.loads(args.selection.read_text())
    assert set(selection['weights'])<=set(CONFIGS)
    assert np.isclose(sum(selection['weights'].values()),1) and all(w>0 for w in selection['weights'].values())
    assert selection['threshold']==.5
    data=Path(os.environ.get('POKEMON_DATA_DIR', str(ROOT/'upload')));hashes={n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ['train.jsonl','test.jsonl','type_chart_gen2.csv']}
    assert hashes==selection['data_sha256']
    train=load(data/'train.jsonl');test=load(data/'test.jsonl')
    assert all('player_won' not in b for b in test)
    B=train+test;V=[describe(b) for b in B];n=len(train)
    y=np.array([int(b['player_won']) for b in train]+[0]*len(test));groups=np.array([v['group'] for v in V]);tr=np.arange(n);va=np.arange(n,len(B))
    chart=load_type_chart(data);cache=ROOT/'scratchboard/cache/full_fit.pkl.gz'
    if cache.exists() and not args.rebuild_base:
        with gzip.open(cache,'rb') as f:base=pickle.load(f)
        assert np.array_equal(base['tr'],tr) and np.array_equal(base['va'],va) and np.array_equal(base['ytr'],y[tr])
    else:
        static={'obs':numeric([observation_features(b,v) for b,v in zip(B,V)]),'cats':pd.DataFrame([identity_features(b,v) for b,v in zip(B,V)])}
        base=fold_features(B,V,y,groups,tr,va,chart,static);cache.parent.mkdir(exist_ok=True);save_cache(cache,base)
    print('Loaded fold-consistent baseline features',flush=True)
    stats=deep.stats_context(train)
    extra={'resources':numeric([deep.resources(b,v) for b,v in zip(B,V)])}
    for name,func in [('exits',deep.departures),('switches',deep.switches)]:extra[name]=numeric([func(b,v,stats,chart) for b,v in zip(B,V)])
    print('Built observed battle-state features',flush=True)
    extra['entryko']=auxiliary_features(B,V,y,groups,tr,va,stats,chart)
    print('Cross-fitted auxiliary event features',flush=True)
    extra['sets']=set_features(V,y,groups,tr,va)
    print('Cross-fitted team/move records',flush=True)
    old=['base','post','obs','threat'];new=['exits','resources','switches','entryko','sets']
    X=pd.concat([base['train'][g] for g in old]+[extra[g].iloc[tr].reset_index(drop=True) for g in new],axis=1)
    Z=pd.concat([base['valid'][g] for g in old]+[extra[g].iloc[va].reset_index(drop=True) for g in new],axis=1)
    assert X.columns.is_unique and X.columns.equals(Z.columns)
    assert not {'battle_id','player_won'}&set(X.columns)
    assert np.isfinite(X.to_numpy()).all() and np.isfinite(Z.to_numpy()).all()
    cache=ROOT/'deep_features/cache/full_fit.pkl.gz';cache.parent.mkdir(exist_ok=True);save_cache(cache,dict(train=X,test=Z,y=y[tr]))
    predictions={};feature_counts={}
    for name in selection['weights']:
        kind,old_groups,new_groups=CONFIGS[name]
        XX=pd.concat([base['train'][g] for g in old_groups]+[extra[g].iloc[tr].reset_index(drop=True) for g in new_groups],axis=1)
        ZZ=pd.concat([base['valid'][g] for g in old_groups]+[extra[g].iloc[va].reset_index(drop=True) for g in new_groups],axis=1)
        assert XX.columns.is_unique and XX.columns.equals(ZZ.columns)
        if kind=='lgb':est=lgbm_final()
        elif kind=='xgb':est=xgb_final()
        else:
            from catboost import CatBoostClassifier
            est=CatBoostClassifier(iterations=700,depth=5,learning_rate=.04,l2_leaf_reg=10,random_seed=0,thread_count=2,verbose=False,allow_writing_files=False,cat_features=list(base['train']['cats'].columns))
        est.fit(XX,y[tr]);predictions[name]=est.predict_proba(ZZ)[:,1];feature_counts[name]=XX.shape[1]
        importance=est.booster_.feature_importance(importance_type='gain') if kind=='lgb' else est.feature_importances_
        pd.DataFrame({'feature':XX.columns,'importance':importance}).sort_values('importance',ascending=False).to_csv(ROOT/f'deep_features/results/full_{name}_importance.csv',index=False)
        print('Fitted '+name,flush=True)
    p=sum(selection['weights'][name]*q for name,q in predictions.items())
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    ids=[b['battle_id'] for b in test];sample=pd.read_csv(sorted(data.glob('sample_submission*.csv'))[0])
    assert len(ids)==len(set(ids))==len(sample)==5000 and set(ids)==set(sample.battle_id)
    probability=pd.DataFrame({'battle_id':ids,**predictions,'blend':p}).set_index('battle_id').loc[sample.battle_id].reset_index()
    probability.to_csv(ROOT/'deep_features/results/test_probabilities.csv',index=False)
    result=pd.DataFrame({'battle_id':sample.battle_id,'player_won':(probability.blend>=selection['threshold']).astype(int)})
    output=ROOT/'deep_features'/selection.get('filename','submission_state_reconstruction.csv');result.to_csv(output,index=False)
    saved=pd.read_csv(output);assert list(saved.columns)==['battle_id','player_won'] and saved.battle_id.equals(sample.battle_id) and set(saved.player_won)<={0,1}
    manifest=dict(selection=selection,rows=len(result),columns=list(result.columns),predicted_wins=int(result.player_won.sum()),feature_count=max(feature_counts.values()),feature_counts=feature_counts,data_sha256=hashes,submission_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['deep_features','scratchboard','scratchboard/reference'] for p in (ROOT/folder).glob('*.py')})
    (ROOT/'deep_features/results/submission_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ['rows','columns','predicted_wins','feature_count','submission_sha256']},indent=2),flush=True)

if __name__=='__main__':main()

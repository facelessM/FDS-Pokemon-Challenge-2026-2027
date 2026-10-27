"""Paired, battle-clustered assessment and data-boundary checks for the frozen candidate."""
import copy, gzip, hashlib, json, pickle, sys
from collections import Counter
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from summarize import predictions
import features as deep
from auxiliary import make_events

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scratchboard'))
from data_views import load,describe
from reference.data import load_type_chart

SEEDS=[20261026,20261105,193]
CHOSEN='deep_entry_sets_blend'

def prefix_hash(b,k=10):
    sides=[]
    for side in ['p1','p2']:
        sides.append(json.dumps([(t[side+'_pokemon_state'],t[side+'_move_details']) for t in b['battle_timeline'][:k]],sort_keys=True,separators=(',',':')))
    return hashlib.sha256('\n'.join(sorted(sides)).encode()).hexdigest()

def main():
    B=load(Path(os.environ.get('POKEMON_DATA_DIR', str(ROOT/'upload')))/'train.jsonl');V=[describe(b) for b in B]
    y=np.array([int(b['player_won']) for b in B]);groups=np.array([v['group'] for v in V])
    ps={s:predictions(s) for s in SEEDS}
    candidates=['fixed_main','previous_submission01','previous_combined','prior_state_blend','prior_entryko_blend','deep_entry_blend','deep_entry_sets_blend','deep_entry_sets_blend_half_previous']
    rows=[]
    for name in candidates:
        acc=[accuracy_score(y,ps[s][name]>=.5) for s in SEEDS]
        ll=[log_loss(y,ps[s][name]) for s in SEEDS]
        rows.append(dict(model=name,accuracies=acc,mean_accuracy=float(np.mean(acc)),mean_log_loss=float(np.mean(ll))))
    _,inv=np.unique(groups,return_inverse=True);sizes=np.bincount(inv);ng=len(sizes)
    comparisons={}
    for ref in ['fixed_main','previous_submission01','deep_entry_blend']:
        d=np.array([((ps[s][CHOSEN]>=.5)==y).astype(float)-((ps[s][ref]>=.5)==y).astype(float) for s in SEEDS])
        perbattle=d.mean(axis=0);cluster=np.bincount(inv,weights=perbattle)
        rng=np.random.default_rng(51917);boot=[]
        for _ in range(50):
            draw=rng.integers(0,ng,size=(200,ng));boot.extend(cluster[draw].sum(axis=1)/sizes[draw].sum(axis=1))
        comparisons[ref]=dict(delta=float(perbattle.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),net_correct_by_seed=d.sum(axis=1).astype(int).tolist(),corrected_by_seed=(d==1).sum(axis=1).tolist(),regressed_by_seed=(d==-1).sum(axis=1).tolist())
    hashes=[prefix_hash(b) for b in B];counts=Counter(hashes);keep=np.array([counts[h]==1 for h in hashes])
    robust={name:[float(accuracy_score(y[keep],ps[s][name][keep]>=.5)) for s in SEEDS] for name in ['fixed_main','previous_submission01',CHOSEN]}
    folds=[];checks=[]
    for seed in SEEDS:
        seen=np.zeros(len(B),int)
        for k in range(5):
            with gzip.open(ROOT/f'scratchboard/cache/{seed}_{k}.pkl.gz','rb') as f:base=pickle.load(f)
            tr,va=base['tr'],base['va'];assert not set(groups[tr])&set(groups[va]);seen[va]+=1
            for ns,rs in [(2,347),(5,99)]:
                for a,b in StratifiedGroupKFold(ns,shuffle=True,random_state=rs).split(np.zeros(len(tr)),y[tr],groups[tr]):
                    assert not set(groups[tr[a]])&set(groups[tr[b]])
                    assert not set(tr[a])&set(va)
            for part in ['train','valid']:
                for name in ['base','post','obs','threat']:
                    X=base[part][name];assert X.columns.is_unique and np.isfinite(X.to_numpy()).all()
                    assert 'battle_id' not in X and 'player_won' not in X
            for name in [CHOSEN,'previous_submission01','fixed_main']:
                folds.append(dict(seed=seed,fold=k,model=name,accuracy=float(accuracy_score(y[va],ps[seed][name][va]>=.5))))
        assert (seen==1).all()
    checks.append('All 15 outer folds and their auxiliary/target-encoding inner folds separate exact timeline groups.')
    ix=np.random.default_rng(18).choice(len(B),size=80,replace=False);small=[copy.deepcopy(B[i]) for i in ix]
    changed=copy.deepcopy(small)
    for i,b in enumerate(changed):b['player_won']=not b['player_won'];b['battle_id']=-100000-i
    sv=[describe(b) for b in small];cv=[describe(b) for b in changed];assert sv==cv
    stats=deep.stats_context(B);chart=load_type_chart(Path(os.environ.get('POKEMON_DATA_DIR', str(ROOT/'upload'))))
    for func in [deep.resources,deep.quality,deep.ordered_state]:
        for a,av,b,bv in zip(small,sv,changed,cv):assert func(a,av)==func(b,bv)
    for func in [deep.departures,deep.switches]:
        for a,av,b,bv in zip(small,sv,changed,cv):assert func(a,av,stats,chart)==func(b,bv,stats,chart)
    X,M=make_events(small,sv,stats,chart,include_entry=True);Z,N=make_events(changed,cv,stats,chart,include_entry=True)
    pd.testing.assert_frame_equal(X,Z);pd.testing.assert_frame_equal(M,N)
    checks.append('80 sampled battles: all new label-free features and auxiliary event labels are invariant to winner-label and battle-ID changes.')
    report=dict(candidate=CHOSEN,seeds=SEEDS,rows=rows,paired_comparisons=comparisons,bootstrap_note='10000 paired resamples of exact-timeline battle clusters. The three seeds remain together within each battle; these are not 30000 independent battles. Intervals are exploratory and do not adjust for model selection.',prefix_duplicate_rows_excluded=int((~keep).sum()),scores_without_prefix_duplicates=robust,checks=checks)
    (ROOT/'deep_features/results/validation.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(folds).to_csv(ROOT/'deep_features/results/paired_folds.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

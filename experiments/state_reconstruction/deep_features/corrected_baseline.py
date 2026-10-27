"""Reproduce corrected deployed-team-model comparisons on the stored outer folds.

This intentionally retains the original global label-free catalog and inner
StratifiedKFold encoding. The stricter baseline is computed by scratchboard/
experiment.py instead. Neither model is fitted on outer-validation outcomes.
"""
import argparse,gzip,json,pickle,sys
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,log_loss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scratchboard'))
from data_views import load
from reference.data import build_species_table,build_move_vocab,build_top_species,build_team_index
from reference.features import build_features
from reference.models import TeamRecordModel,LgbXgbBlend

def matrix(B):
    context={'species':build_species_table(B),'move_vocab':build_move_vocab(B),'top_species':build_top_species(B),'teams':build_team_index(B)}
    return build_features(B,['starter','state_t30','sacrifice','move_usage','move_effects','alive_species','team_ids'],context).drop(columns='battle_id')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,default=[20261026,20261105,193]);args=ap.parse_args()
    B=load(Path(os.environ.get('POKEMON_DATA_DIR', str(ROOT/'upload')))/'train.jsonl');y=np.array([int(b['player_won']) for b in B]);X=matrix(B);out=ROOT/'fix_impact_check';out.mkdir(exist_ok=True)
    for seed in args.seeds:
        path=out/f'fixed_{seed}.npz'
        if path.exists():continue
        p=np.zeros(len(y));foldids=np.zeros(len(y),int)
        for k in range(5):
            with gzip.open(ROOT/f'scratchboard/cache/{seed}_{k}.pkl.gz','rb') as f:fold=pickle.load(f)
            tr,va=fold['tr'],fold['va'];est=TeamRecordModel(LgbXgbBlend).fit(X.iloc[tr],y[tr]);p[va]=est.predict_proba(X.iloc[va])[:,1];foldids[va]=k
            print(json.dumps(dict(seed=seed,fold=k,accuracy=accuracy_score(y[va],p[va]>=.5))),flush=True)
        np.savez_compressed(path,p=p,fold=foldids)
        print(json.dumps(dict(seed=seed,accuracy=accuracy_score(y,p>=.5),log_loss=log_loss(y,p))),flush=True)

if __name__=='__main__':main()

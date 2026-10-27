"""Check fitted roster, threat and set features under different Python hash seeds."""
import argparse,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scratchboard'));sys.path.insert(0,str(ROOT/'deep_features'))
from data_views import load,describe,observation_features,identity_features
from features import resources
from team_context import TeamContext
from tactics import TacticalContext
from move_records import MoveRecords
from reference.data import load_type_chart,build_top_species

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--worker',action='store_true');args=ap.parse_args()
    if args.worker:
        battles=load(args.data/'train.jsonl');views=[describe(b) for b in battles];y=np.array([int(b['player_won']) for b in battles])
        # These used to insert empty move evidence when the baseline cache was
        # absent. Their execution must not alter any fitted-context input.
        before=copy.deepcopy(views)
        for battle,view in zip(battles,views):
            observation_features(battle,view);identity_features(battle,view);resources(battle,view)
        assert views==before
        a=7000;test=views[a:];context=TeamContext(enriched=True,moves=True,graph=True).fit(views[:a],y[:a])
        records=MoveRecords().fit(views[:a],y[:a]);tactical=TacticalContext().fit(battles[:a],views[:a],load_type_chart(args.data),build_top_species(battles[:a]))
        X=pd.concat([context.transform(test,'hierarchy'),records.transform(test),pd.DataFrame([tactical.threat_features(b,v) for b,v in zip(battles[a:],test)])],axis=1).astype(np.float64)
        print(json.dumps({'rows':len(X),'columns':X.shape[1],'sha256':hashlib.sha256(X.to_numpy().tobytes()+json.dumps(list(X.columns)).encode()).hexdigest()}));return
    reports=[]
    for seed in ['1','59','101']:
        env=dict(os.environ,PYTHONHASHSEED=seed,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        result=subprocess.check_output([sys.executable,str(Path(__file__)), '--data',str(args.data.resolve()),'--worker'],env=env,text=True)
        reports.append(dict(hash_seed=seed,**json.loads(result)))
    assert len({r['sha256'] for r in reports})==1,reports
    out=ROOT/'determinism_check.json';out.write_text(json.dumps({'passed':True,'hash_seed_runs':reports},indent=2));print(out.read_text())

if __name__=='__main__':main()

import argparse,gzip,json,pickle,sys,time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import features as deep
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scratchboard'))
from data_views import load,describe
from reference.data import load_type_chart

def numeric(rows):
    X=pd.DataFrame(rows).fillna(0).astype(np.float32);assert np.isfinite(X.to_numpy()).all();return X

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=20261026);args=ap.parse_args()
    B=load(Path(os.environ.get('POKEMON_DATA_DIR', str(ROOT/'upload')))/'train.jsonl');V=[describe(b) for b in B];chart=load_type_chart(Path(os.environ.get('POKEMON_DATA_DIR', str(ROOT/'upload'))));out=ROOT/'deep_features/cache';out.mkdir(exist_ok=True)
    static={name:numeric([fn(b,v) for b,v in zip(B,V)]) for name,fn in [('resources',deep.resources),('quality',deep.quality),('order',deep.ordered_state)]}
    start=time.time()
    for k in range(5):
        path=out/f'{args.seed}_{k}.pkl.gz'
        if path.exists():continue
        with gzip.open(ROOT/f'scratchboard/cache/{args.seed}_{k}.pkl.gz','rb') as f:base=pickle.load(f)
        stats=deep.stats_context([B[i] for i in base['tr']]);groups=dict(static)
        for name,fn in [('exits',deep.departures),('switches',deep.switches)]:groups[name]=numeric([fn(b,v,stats,chart) for b,v in zip(B,V)])
        with gzip.open(path.with_suffix('.tmp'),'wb',compresslevel=2) as f:pickle.dump(groups,f,pickle.HIGHEST_PROTOCOL)
        path.with_suffix('.tmp').replace(path)
        print(json.dumps({'seed':args.seed,'fold':k,'columns':{n:X.shape[1] for n,X in groups.items()},'elapsed':round(time.time()-start,1)}),flush=True)

if __name__=='__main__':main()
